"""Full-state acceleration field with explicit Euler and differentiable FK reuse."""
import torch
from odemimic.g1.kinematics.rotations import heading_quat, quat_apply, quat_apply_inverse, quat_mul, quat_normalize, quat_from_rotvec, rot6d_from_quat
from odemimic.g1.world_model.state_spec import quat_apply_inverse_quat

def full_features(wm,s,bodies=None):
    b=wm.heads.robot_bodies(s) if bodies is None else bodies
    h=heading_quat(quat_normalize(s[...,3:7]))
    h=h[...,None,:].expand(*b['quat'].shape)
    pos=quat_apply_inverse(h,b['pos']-s[...,None,:3])
    rot=rot6d_from_quat(quat_mul(quat_apply_inverse_quat(h),b['quat']))
    lin=quat_apply_inverse(h,b['lin_vel'])
    ang=quat_apply_inverse(h,b['ang_vel'])/2
    return torch.cat([s,pos.flatten(-2),rot.flatten(-2),lin.flatten(-2),ang.flatten(-2)],-1)

def euler_step(wm,s,action,xi,bodies=None):
    y=wm.dynamics((full_features(wm,s,bodies)-wm.input_mean)/wm.input_std,action,wm.spec.normalize_xi(xi))
    # Scale acceleration so a control interval has the baseline velocity-increment scale.
    acceleration = (wm.velocity_delta_mean + wm.velocity_delta_std * y) / wm.spec.dt
    dv = wm.spec.dt * acceleration
    p,q,qj,v,w,qd=wm.spec.split(s)
    q=quat_normalize(q);dt=wm.spec.dt
    v1=v+dv[...,:3]
    w1=w+dv[...,3:6];qd1=qd+dv[...,6:]
    p1=p+dt*v
    q1=quat_normalize(quat_mul(q,quat_from_rotvec(dt*w)))
    qj1=qj+dt*qd
    return torch.cat([p1,q1,qj1,v1,w1,qd1],-1)
