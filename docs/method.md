# Structured world model

The starting point is [Alchedata/supermimic-locomujoco](https://github.com/Alchedata/supermimic-locomujoco). SuperMimic alternates real simulator collection, supervised world-model fitting and policy improvement through differentiable imagination. ODEMimic studies the dynamics parameterization within that loop.

Let `s = (p, q, j, v, omega, nu)`, where `q` is a scalar-first root quaternion, `j` are joint positions, and `nu` are joint velocities. For one Euler interval:

```text
acceleration = (velocity_increment_mean + velocity_increment_std * network_output) / dt
p_next       = p + dt * v
q_next       = normalize(q * Exp(dt * omega))
j_next       = j + dt * nu
(v, omega, nu)_next = (v, omega, nu) + dt * acceleration
```

The field input is reconstructed from the physical state using differentiable FK. Successor FK features are reused at the next field call, retaining gradients. Fixed robot transforms are precomputed. H1's zero hinge anchors and overlapping body/reward feature sets are simplified without changing the state representation.

## H1

- Physical state: 26 qpos + 25 qvel = 51 dimensions.
- Full input state: physical state + 168 observation-site features + 48 reward-site features = 267 dimensions; 19 torque actions are supplied separately.
- Frozen statistics: full-state mean/std, and real state-increment mean/std.
- Field output: 25 generalized accelerations, scaled through the saved velocity-increment statistics; termination logit has prior offset −5.
- WM loss: autoregressive eight-step physical-state MSE divided by empirical state-increment std, plus termination BCE. Invalid/terminated transitions are masked.
- Actor: ten observation frames, ten reference frames, 32-step imagination and critic bootstrap. Action output is Tanh in the reported H1 comparisons.

The MLP baseline learns a 267-D state increment instead, with quaternion normalization. The H1 PyTorch learner supports both models in the same code.

## G1 ODE-preprocessing

- Physical state: 36 qpos + 35 qvel = 71 dimensions.
- Field features: the raw physical state and 14 × 15 heading-frame body features = 281 dimensions, plus 29 actions and the reproduction's domain parameters.
- Full-input mean/std uses valid warmup transitions, with a std floor of 0.1.
- Root linear-velocity increments are world-frame; root angular-velocity increments are body-frame. Output and state-increment std floors are 0.01.
- WM state loss uses the raw 71-D predicted state divided by empirical state-increment std.
- Statistics are fitted once, frozen and checkpointed.
- The MLP baseline retains the reproduction's `StateSpec` features and error vector. ODE-preprocessing is enabled for the ODE branch.

The learned termination head and analytic reward are retained. Older G1 torque-control and older ODE-processing trials are excluded from the current G1 result table.

## Optional AS

At real replay starts, add `lambda * mean(relu(abs(raw_action) - threshold)^2)` to actor loss. Its threshold is 1 in the reported runs. H1 uses `lambda=0.01`; the G1 MLP uses 0.01 and Euler uses 0.001. It is a raw-output penalty, not a replacement environment reward.

The contribution being tested is structured learned dynamics for differentiable policy improvement. The title uses “structured” for the explicit kinematic integration and acceleration prediction; no quantitative explainability result is claimed.
