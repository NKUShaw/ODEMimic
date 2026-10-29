"""World-model fitting on k-step replay windows (paper Eq. 1, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from odemimic.g1.numerics import checked_clip, require_finite

from odemimic.g1.data.replay_buffer import ReplayBuffer, WindowBatch
from odemimic.g1.envs.mdp.motion_library import MotionLibrary
from odemimic.g1.world_model.heads import ReferenceFrame
from odemimic.g1.training.distributed import allreduce_grads
from odemimic.g1.world_model.world_model import WorldModel


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"


def reference_at(lib: MotionLibrary, frame: torch.Tensor) -> ReferenceFrame:
    r = lib.ref(frame)
    b = lib.bodies(frame)
    return ReferenceFrame(b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w, r.root_pos_w, r.root_quat_w)


class WorldModelTrainer:
    def __init__(self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        wm.enable_compile(cfg.compile)

    @torch.no_grad()
    def fit_output_normalization(self, buffer):
        from odemimic.g1.world_model.full_state_euler import full_features
        e,t=buffer.window_mask(1).nonzero(as_tuple=True)
        assert len(e)>0
        sx=torch.zeros(281,device=buffer.s.device,dtype=torch.float64);sx2=torch.zeros_like(sx)
        sd=torch.zeros(71,device=buffer.s.device,dtype=torch.float64);sd2=torch.zeros_like(sd)
        for lo in range(0,len(e),4096):
            ee,tt=e[lo:lo+4096],t[lo:lo+4096]
            s0=buffer.s[ee,tt];s1=buffer.s[ee,(tt+1)%buffer.T]
            x=full_features(self.wm,s0).double();delta=(s1-s0).double()
            assert torch.isfinite(x).all() and torch.isfinite(delta).all()
            sx+=x.sum(0);sx2+=x.square().sum(0);sd+=delta.sum(0);sd2+=delta.square().sum(0)
        mu=sx/len(e);std=(sx2/len(e)-mu.square()).clamp_min(0).sqrt().clamp_min(.1)
        dm=sd/len(e);ds=(sd2/len(e)-dm.square()).clamp_min(0).sqrt().clamp_min(.01)
        self.wm.input_mean.copy_(mu);self.wm.input_std.copy_(std)
        self.wm.state_delta_mean.copy_(dm);self.wm.state_delta_std.copy_(ds)
        self.wm.velocity_delta_mean.copy_(dm[36:71]);self.wm.velocity_delta_std.copy_(ds[36:71])
        self.wm.normalization_fitted.fill_(True)
        print('ODE_PREPROCESSING',dict(samples=len(e),input_dim=281,loss_dim=71,velocity_frame='world_linear_body_angular',frozen=True),flush=True)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        k, n = w.frame.shape
        # Only the root pose of the references is needed for the termination features.
        r = self.lib.ref(w.frame.reshape(-1))
        refs = [ReferenceFrame(None, None, None, None, r.root_pos_w[j * n : (j + 1) * n], r.root_quat_w[j * n : (j + 1) * n]) for j in range(k)]
        return self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, refs)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        if self.wm.cfg.kind == "euler" and not bool(self.wm.normalization_fitted):
            self.fit_output_normalization(buffer)
        agg: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            loss, m = self.loss_on(w)
            self.opt.zero_grad(set_to_none=True)
            require_finite(loss,"WM loss")
            loss.backward()
            allreduce_grads(self.wm)
            gn = checked_clip(self.wm.parameters(), self.cfg.grad_clip,"WM")
            self.opt.step()
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}

    @torch.no_grad()
    def online_error(self, buffer: ReplayBuffer, n: int = 1024, k: int = 8) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)
