"""PyTorch H1 Euler port with frozen-checkpoint conversion and native feature layout."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def mul(q,r):
    return torch.cat((q[...,:1]*r[...,:1]-(q[...,1:]*r[...,1:]).sum(-1,keepdim=True),
                      q[...,:1]*r[...,1:]+r[...,:1]*q[...,1:]+torch.cross(q[...,1:],r[...,1:],dim=-1)), -1)

def matrix(q):
    q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-8)
    w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)

def rotate(r,v):
    return (r@v.unsqueeze(-1)).squeeze(-1)

def log_matrix(r):
    # Match scipy/JAX's dominant-diagonal matrix-to-quaternion algorithm.
    diag=r.diagonal(dim1=-2,dim2=-1)
    trace=diag.sum(-1)
    choice=torch.cat((diag,trace[...,None]),-1).argmax(-1)
    qs=[]
    for i in range(3):
        j,k=(i+1)%3,(i+2)%3
        components=[None]*4
        components[i]=1-trace+2*r[...,i,i]
        components[j]=r[...,j,i]+r[...,i,j]
        components[k]=r[...,k,i]+r[...,i,k]
        components[3]=r[...,k,j]-r[...,j,k]
        qs.append(torch.stack(components,-1))
    qs.append(torch.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],
                            r[...,1,0]-r[...,0,1],1+trace),-1))
    q=torch.stack(qs,-2).gather(-2,choice[...,None,None].expand(*choice.shape,1,4)).squeeze(-2)
    q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True).clamp_min(1e-8)
    q=torch.where(q[...,-1:]<0,-q,q)
    v=q[...,:3];r2=(v*v).sum(-1,keepdim=True);n=(r2+1e-16).sqrt()
    scale=torch.where(r2<1e-6,2+r2/3+3*r2*r2/20,2*torch.atan2(n,q[...,-1:])/n)
    return v*scale

class FlaxMLP(nn.Module):
    def __init__(self, weights, prefix, norm=False):
        super().__init__();self.norm=norm
        count=sum(k.startswith(prefix+'/Dense_') and k.endswith('/kernel') for k in weights)
        self.layers=nn.ModuleList();self.norms=nn.ModuleList()
        for i in range(count):
            w=weights[f'{prefix}/Dense_{i}/kernel'];layer=nn.Linear(*w.shape)
            layer.weight.data.copy_(torch.from_numpy(w.T.copy()));layer.bias.data.copy_(torch.from_numpy(weights[f'{prefix}/Dense_{i}/bias'].copy()))
            self.layers.append(layer)
            if norm and i<count-1:
                ln=nn.LayerNorm(w.shape[1],eps=1e-6)
                ln.weight.data.copy_(torch.from_numpy(weights[f'{prefix}/LayerNorm_{i}/scale'].copy()))
                ln.bias.data.copy_(torch.from_numpy(weights[f'{prefix}/LayerNorm_{i}/bias'].copy()))
                self.norms.append(ln)
    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x=layer(x)
            if i<len(self.layers)-1:
                if self.norm:x=F.silu(self.norms[i](x))
                else:x=F.elu(x)
        return x

class H1(nn.Module):
    def __init__(self, fixture):
        super().__init__();fixture=Path(fixture);self.meta=json.loads((fixture/'metadata.json').read_text())
        weights=dict(np.load(fixture/'weights.npz'));self.world=FlaxMLP(weights,'wp/params/MLP_0',True)
        self.is_ode=self.world.layers[-1].out_features==26
        self.encoder=FlaxMLP(weights,'pp/params/MLP_0');self.decoder=FlaxMLP(weights,'pp/params/MLP_1')
        self.critic=FlaxMLP(weights,'vp/params/MLP_0')
        for k in ['mean','std','delta_mean','delta_std']:self.register_buffer(k,torch.from_numpy(weights['norm/'+k].copy()))
        model=dict(np.load(fixture/'model.npz'));self.model_arrays=model
        for k,v in model.items():self.register_buffer('m_'+k,torch.from_numpy(v.copy()).float() if v.dtype.kind=='f' else torch.from_numpy(v.copy()).long())
        self.parents=model['body_parentid'].tolist();self.joints=[list(range(int(a),int(a+n))) if n else [] for a,n in zip(model['body_jntadr'],model['body_jntnum'])]
        self.joint_types=model['jnt_type'].tolist();self.qadr=model['jnt_qposadr'].tolist();self.vadr=model['jnt_dofadr'].tolist()
        self.site_bodies=model['site_bodyid'].tolist();self.ids=self.meta['ids'];self.rids=self.meta['reward_ids']
        self.register_buffer('body_rot',matrix(self.m_body_quat),persistent=False)
        self.register_buffer('site_rot',matrix(self.m_site_quat),persistent=False)
        identity=np.array([1,0,0,0])
        self.body_identity=np.all(model['body_quat']==identity,axis=-1).tolist()
        self.site_identity=np.all(model['site_quat']==identity,axis=-1).tolist()
        self.zero_joint_anchor=np.all(model['jnt_pos']==0,axis=-1).tolist()
        self.reward_subset=self.rids[0]==self.ids[0] and all(i in self.ids[1:] for i in self.rids[1:])
        self.register_buffer('reward_feature_indexes',torch.tensor([self.ids[1:].index(i) for i in self.rids[1:]],dtype=torch.long),persistent=False)
        self.register_buffer('obs_idx',torch.tensor(self.meta['obs_indexes']))
        self.register_buffer('low',torch.tensor(self.meta['low']));self.register_buffer('high',torch.tensor(self.meta['high']))
        self.dt=self.meta['dt']

    def features(self,x):
        q,v=x[...,:26],x[...,26:51];zero=q[...,:3]*0
        identity=torch.eye(3,device=q.device,dtype=q.dtype).expand(*q.shape[:-1],3,3)
        positions=[zero];rotations=[identity];angular=[zero];linear=[zero]
        for body in range(1,len(self.parents)):
            parent=self.parents[body];pr=rotations[parent]
            p=positions[parent]+rotate(pr,self.m_body_pos[body].expand_as(zero))
            r=pr if self.body_identity[body] else pr@self.body_rot[body]
            w=angular[parent];vel=linear[parent]+torch.cross(w,p-positions[parent],dim=-1)
            for j in self.joints[body]:
                if self.joint_types[j]==0:
                    p=q[...,:3];r=matrix(q[...,3:7]);w=rotate(r,v[...,3:6]);vel=v[...,:3]
                else:
                    assert self.joint_types[j]==3
                    axis=rotate(r,self.m_jnt_axis[j].expand_as(zero))
                    angle=q[...,self.qadr[j]]-self.m_qpos0[self.qadr[j]]
                    local=self.m_jnt_axis[j]*torch.sin(angle[...,None]/2)
                    dq=torch.cat((torch.cos(angle[...,None]/2),local),-1)
                    if self.zero_joint_anchor[j]:
                        r=r@matrix(dq)
                    else:
                        anchor=p+rotate(r,self.m_jnt_pos[j].expand_as(zero));r=r@matrix(dq)
                        newp=anchor-rotate(r,self.m_jnt_pos[j].expand_as(zero))
                        vel=vel+torch.cross(w,newp-p,dim=-1)+torch.cross(axis*v[...,self.vadr[j],None],newp-anchor,dim=-1)
                        p=newp
                    w=w+axis*v[...,self.vadr[j],None]
            positions.append(p);rotations.append(r);angular.append(w);linear.append(vel)
        # Compute only required sites, retaining the original site ordering.
        sites={}
        for site in sorted(set(self.ids+self.rids)):
            body=self.site_bodies[site];r=rotations[body];offset=rotate(r,self.m_site_pos[site].expand_as(zero))
            site_r=r if self.site_identity[site] else r@self.site_rot[site]
            sites[site]=(positions[body]+offset,site_r,angular[body],linear[body]+torch.cross(angular[body],offset,dim=-1))
        def relative(ids):
            p,r,w,vel=sites[ids[0]]
            ps=torch.stack([sites[i][0] for i in ids[1:]],-2)
            rs=torch.stack([sites[i][1] for i in ids[1:]],-3)
            ws=torch.stack([sites[i][2] for i in ids[1:]],-2)
            vs=torch.stack([sites[i][3] for i in ids[1:]],-2)
            rel=r.transpose(-1,-2).unsqueeze(-3)@rs
            ang=rotate(rel.transpose(-1,-2),ws)-w.unsqueeze(-2)
            lin=rotate(r.unsqueeze(-3),vel.unsqueeze(-2)-vs)
            return ps-p.unsqueeze(-2),log_matrix(rel),torch.cat((ang,lin),-1)
        obs=relative(self.ids)
        rew=tuple(x.index_select(-2,self.reward_feature_indexes) for x in obs) if self.reward_subset else relative(self.rids)
        return torch.cat((q,v,*(x.flatten(-2) for x in obs),*(x.flatten(-2) for x in rew)),-1)

    def obs(self,x):return torch.cat((x[...,self.obs_idx],x[...,51:219]),-1)
    def normh(self,h):return (h-self.obs(self.mean))/self.obs(self.std)
    def normg(self,g):return ((g.reshape(g.shape[0],10,217)-self.mean[2:219])/self.std[2:219]).flatten(1)
    def policy(self,h,g):return self.decoder(torch.cat((self.normh(h).flatten(1),torch.tanh(self.encoder(self.normg(g)))),-1))
    def value(self,x,g):return self.critic(torch.cat(((x-self.mean)/self.std,self.normg(g)),-1))[...,0]
    def bounded(self,raw):return (self.low+self.high)/2+(self.high-self.low)/2*torch.tanh(raw)

    def step(self,x,action,full=None):
        if full is None:full=self.features(x)
        out=self.world(torch.cat(((full-self.mean)/self.std,action),-1))
        acc=self.delta_mean[26:51]/self.dt+(self.delta_std[26:51]/self.dt)*out[...,:25]
        q,v=x[...,:26],x[...,26:51];rot=v[...,3:6]*self.dt
        a2=(rot*rot).sum(-1,keepdim=True);a=(a2+1e-16).sqrt()
        factor=torch.where(a2<1e-6,.5-a2/48+a2*a2/3840,torch.sin(a/2)/a)
        quat=mul(q[...,3:7],torch.cat((torch.cos(a/2),rot*factor),-1));quat=quat/torch.linalg.vector_norm(quat,dim=-1,keepdim=True).clamp_min(1e-8)
        q1=torch.cat((q[...,:3]+self.dt*v[...,:3],quat,q[...,7:]+self.dt*v[...,6:]),-1)
        return torch.cat((q1,v+self.dt*acc),-1),out[...,-1]-5

    def reward(self,x,ref,action):
        q=F.normalize(x[...,3:7],dim=-1,eps=1e-8);qr=F.normalize(ref[...,3:7],dim=-1,eps=1e-8)
        rw=(q*qr).sum(-1);rv=q[...,:1]*qr[...,1:]-qr[...,:1]*q[...,1:]-torch.cross(q[...,1:],qr[...,1:],dim=-1)
        angle=2*torch.atan2(((rv*rv).sum(-1)+1e-16).sqrt(),rw.abs())
        nonquat=[0,1,2]+list(range(7,26));r=self.meta['reward']
        ep=((x[...,nonquat]-ref[...,nonquat])**2).mean(-1)+angle
        ev=((x[...,26:51]-ref[...,26:51])**2).mean(-1)
        dp=(x[...,219:231]-ref[...,219:231]).square().mean(-1)
        dr=(x[...,231:243]-ref[...,231:243]).square().mean(-1)
        vel=(x[...,243:267]-ref[...,243:267]).reshape(*x.shape[:-1],4,6)
        eva=vel[...,:3].square().mean((-2,-1));evl=vel[...,3:].square().mean((-2,-1))
        result=r['_qpos_w_sum']*torch.exp(-r['_qpos_w_exp']*ep)+r['_qvel_w_sum']*torch.exp(-r['_qvel_w_exp']*ev)+r['_rpos_w_sum']*torch.exp(-r['_rpos_w_exp']*dp)+r['_rquat_w_sum']*torch.exp(-r['_rquat_w_exp']*dr)+r['_rvel_w_sum']*(torch.exp(-r['_rvel_w_exp']*eva)+torch.exp(-r['_rvel_w_exp']*evl))
        penalty=-r['_action_out_of_bounds_coeff']*(F.relu(self.low-action)+F.relu(action-self.high)).square().mean(-1)
        return (result+penalty.clamp_min(-1)).clamp_min(0)

    def wm_loss(self,b):
        dim=51 if self.is_ode else 267
        x=b['x'][:,0,:dim];total=x.new_zeros(());den=b['mask'].sum().clamp_min(1)
        for i in range(8):
            x=torch.where(b['mask'][:,i,None]>0,x,b['x'][:,i,:dim].detach())
            x,logit=self.step(x,b['a'][:,i]) if self.is_ode else self.mlp_step(x,b['a'][:,i])
            err=((x-b['x1'][:,i,:dim])/self.delta_std[:dim]).square().mean(-1)
            total=total+((err+F.binary_cross_entropy_with_logits(logit,b['absorbing'][:,i].float(),reduction='none'))*b['mask'][:,i]).sum()
        return total/den

    def mlp_step(self,x,action):
        out=self.world(torch.cat(((x-self.mean)/self.std,action),-1))
        pred=x+self.delta_mean+self.delta_std*out[...,:267]
        pred=torch.cat((pred[...,:3],F.normalize(pred[...,3:7],dim=-1,eps=1e-8),pred[...,7:]),-1)
        return pred,out[...,-1]-5

    def actor_loss(self,b,refs,first_action_delta=None,return_aux=False,saturation_weight=0.,saturation_threshold=1.):
        x=b['x'][:,0];h=b['h'];full=self.features(x[...,:51]) if self.is_ode else x
        survival=x.new_ones(len(x));total=x.new_zeros(len(x));trace=[]
        lengths=torch.as_tensor(self.meta['lengths'],device=x.device)[b['tid'][:,0].long()]
        frame=b['f'][:,0]
        for t in range(32):
            goal=refs[:,t:t+10,2:219].flatten(1)
            raw=self.policy(h,goal)
            if t==0:first_raw=raw
            action=self.bounded(raw)
            if t==0 and first_action_delta is not None:action=action+first_action_delta
            if self.is_ode:
                physical,logit=self.step(x[...,:51],action,full);xn=self.features(physical)
            else:xn,logit=self.mlp_step(x,action)
            rr=self.reward(xn,refs[:,t+1],action)*(frame+t+1<lengths)
            cont=(frame+t+1<lengths-1)*torch.sigmoid(-logit)
            if return_aux:
                with torch.no_grad():vv=self.value(x,goal)
                trace.append((x.detach(),goal.detach(),rr.detach(),cont.detach(),vv.detach(),(frame+t<lengths).detach()))
            x=xn;full=x
            total=total+.99**t*survival*rr
            survival=survival*cont
            h=torch.cat((h[:,1:],self.obs(x)[:,None]),1)
        bootstrap=self.value(x,refs[:,32:42,2:219].flatten(1))
        loss=-(total+.99**32*survival*bootstrap).mean()
        if saturation_weight:loss=loss+saturation_weight*F.relu(first_raw.abs()-saturation_threshold).square().mean()
        if return_aux:return loss,(tuple(torch.stack([t[i] for t in trace]) for i in range(6)),bootstrap.detach())
        return loss
