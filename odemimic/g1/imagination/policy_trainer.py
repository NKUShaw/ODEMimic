"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from odemimic.g1.data.replay_buffer import StartBatch

import torch
from odemimic.g1.numerics import checked_clip, require_finite

from odemimic.g1.data.replay_buffer import ReplayBuffer
from odemimic.g1.envs.mdp.motion_library import MotionLibrary
from odemimic.g1.imagination.objective import gae_targets, policy_objective
from odemimic.g1.imagination.rollout import ImagRollout, imagine
from odemimic.g1.policy.nets import Critic, Policy
from odemimic.g1.training.distributed import allreduce_grads
from odemimic.g1.world_model.world_model import WorldModel


@dataclass
class PolicyTrainerCfg:
    saturation_weight: float = 0.0
    saturation_threshold: float = 1.0
    horizon: int = 32
    batch_size: int = 4096
    microbatch: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"
    anchor_obs: bool = False  # None | "default" | "reduce-overhead" | "max-autotune-no-cudagraphs"


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        from odemimic.g1.imagination.compiled import ImagStep

        self.step_fn = ImagStep(wm, lib.default_joint_pos, compile_mode=cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr

    def _critic_inputs(self, ro: ImagRollout):
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        return feat, ro.goals, ro.e_root

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, cfg.history)
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model ------------------------------------
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        self.opt.zero_grad(set_to_none=True)
        chunks, feats = [], []
        J_sum = torch.zeros((), device=start.state.device)
        for lo in range(0, cfg.batch_size, cfg.microbatch):
            hi = min(lo + cfg.microbatch, cfg.batch_size)
            sub = StartBatch(**{f.name: (getattr(start, f.name)[:, lo:hi] if f.name.startswith("hist_") else getattr(start, f.name)[lo:hi]) for f in fields(start)})
            ro = imagine(self.wm, self.policy, self.lib, sub, cfg.horizon, cfg.future_frames, obs_noise=cfg.obs_noise, step_fn=self.step_fn, anchor_obs=cfg.anchor_obs)
            feat, goals, e_root = self._critic_inputs(ro)
            v_H = self.critic(feat[-1], goals[-1], xi_norm[lo:hi], e_root[-1]) if cfg.use_critic else None
            J = policy_objective(ro, v_H, cfg.gamma)
            weight = (hi - lo) / cfg.batch_size
            as_loss = torch.relu(ro.raw_actions.abs() - cfg.saturation_threshold).square().mean()
            ((-J / cfg.horizon + cfg.saturation_weight * as_loss) * weight).backward()
            J_sum += J.detach() * weight
            chunks.append(ImagRollout(**{f.name: getattr(ro, f.name).detach() if torch.is_tensor(getattr(ro, f.name)) else getattr(ro, f.name) for f in fields(ro)}))
            feats.append(feat.detach())
            del ro, feat, goals, e_root, v_H, J
        ro = ImagRollout(**{f.name: torch.cat([getattr(c, f.name) for c in chunks], dim=1) if f.name != "reward_terms" else {k: sum(c.reward_terms[k] * c.actions.shape[1] / cfg.batch_size for c in chunks) for k in chunks[0].reward_terms} for f in fields(ImagRollout)})
        feat = torch.cat(feats, dim=1)
        goals, e_root = ro.goals, ro.e_root
        J = J_sum
        allreduce_grads(self.policy)
        gn = checked_clip(self.policy.parameters(), cfg.grad_clip_norm,"actor")
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)

        metrics = {
            "imag_objective": float(J.detach()),
            "as_loss": float(torch.relu(ro.raw_actions.abs() - cfg.saturation_threshold).square().mean()),
            "raw_output_above_threshold": float((ro.raw_actions.abs() > cfg.saturation_threshold).float().mean()),
            "tanh_derivative_below_001": float(((1-ro.actions.square()) < 0.01).float().mean()) if self.policy.cfg.action_output == "tanh" else 0.0,
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(gn),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
        }
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v

        # ---- critic: fit to GAE targets of this rollout (targets fixed) ---------------------
        if cfg.use_critic:
            with torch.no_grad():
                feat_d, goals_d, e_root_d = feat.detach(), goals.detach(), e_root.detach()
                H1, n, _ = feat_d.shape
                xi_rep = xi_norm[None].expand(H1, n, -1)
                values = self.critic(feat_d.reshape(H1 * n, -1), goals_d.reshape(H1 * n, -1), xi_rep.reshape(H1 * n, -1), e_root_d.reshape(H1 * n, -1)).reshape(H1, n)
                targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda)  # (H, n)
                X_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])
                X_goal = goals_d[:-1].reshape(-1, goals_d.shape[-1])
                X_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])
                X_er = e_root_d[:-1].reshape(-1, 4)
                Y = targets.reshape(-1)
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    pred = self.critic(X_feat[idx], X_goal[idx], X_xi[idx], X_er[idx])
                    closs = torch.nn.functional.mse_loss(pred, Y[idx])
                    self.critic_opt.zero_grad(set_to_none=True)
                    require_finite(closs,"critic loss")
                    closs.backward()
                    allreduce_grads(self.critic)
                    checked_clip(self.critic.parameters(), cfg.critic_grad_clip,"critic")
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = float(last)
            metrics["critic_value_mean"] = float(values.mean())
            metrics["critic_target_mean"] = float(targets.mean())
        return metrics
