# H1 protocol

Unitree H1 in LocoMuJoCo, 19 torque actions, native `MimicReward` and native absorbing conditions. One model and policy cover the 40 motion names in `configs/lafan1_40.json`.

| Setting | Value |
|---|---:|
| Parallel environments | 4096 |
| Regular collection | alternating 3/2 vector steps per iteration |
| WM window / batch / updates | 8 / 2048 / 10 |
| Actor batch / imagined horizon | 4096 / 32 |
| Critic epochs / minibatches | 5 / 4 |
| PyTorch actor microbatch | 4096 |
| Evaluation | 40 motions × 4 fixed starts |
| Evaluation horizon | 1000 controls, 10 seconds |
| Evaluation reset seed | 20261022 |

Learning rate starts at 2e-4, with the original 250k cosine schedule and 3% floor. Gamma is 0.99 and GAE lambda is 0.95. Actor Adam betas are `(0.9, 0.95)`; WM/critic use `(0.9, 0.999)`. Gradient norm limit is 1; actor elements are additionally clipped to 0.1.

Return is mean undiscounted episode sum. Error is mean relative-site position distance in meters. Failures count absorbing/numerical terminations. Natural reference endings can occur without an absorbing failure. All scores stop at the first episode end.

The completed 10k baseline/Euler table uses the historical JAX runs and saved-checkpoint continuations. The new AS continuations switch to the validated PyTorch learner at recorded checkpoints. Parameters, Adam moments/counters, normalization, curriculum and RNG are restored; live simulator/replay are refilled. Nominal collection transitions exclude warmup and replay refill. Historical microbatches and continuation arguments are retained in `results/provenance.json`.

See [the inherited reference protocol](protocol_h1_reference.md) for earlier action-output experiments and the original setup. Its historical runtime/default-budget notes are superseded by the current README and migration document.
