# Source and attribution

ODEMimic is an extension of **[Alchedata/supermimic-locomujoco](https://github.com/Alchedata/supermimic-locomujoco)**, starting from commit `de9f39191b9e8b87ff1afd2b0129dd93d4795e8c`. Its Git history is retained in this repository.

The project studies structured Neural ODE dynamics and optional anti-saturation regularization within the [SuperMimic](https://supermimic.github.io/) training loop. SuperMimic's collect–fit–improve algorithm, differentiable imagination, actor/critic setup, and the original paper belong to their respective authors.

- `supermimic/`: the inherited H1 JAX reference implementation and simulator adapter.
- `odemimic/h1/`: the current PyTorch learner, differentiable FK and checkpoint migration.
- `odemimic/g1/`: adapted from [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), revision `ba6105349069bf040f1fdad60873d13138d505a2`, with the current G1 full-state ODE processing and numerical guards. Its MIT license is included in that directory.
- [LocoMuJoCo](https://github.com/robfiras/loco-mujoco) supplies the H1 simulator and motion interface; [mjlab](https://github.com/mujocolab/mjlab) supplies the G1 simulation infrastructure.
- [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) informed reproduction checks.

Motion datasets, robot assets and external dependencies are not redistributed here and retain their upstream licenses.
