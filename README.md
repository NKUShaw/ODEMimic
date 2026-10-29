# ODEMimic

**Distilling Physics Simulators into Structured Neural ODE World Models**

PyTorch code for humanoid motion imitation with learned continuous dynamics, based on [Alchedata/supermimic-locomujoco](https://github.com/Alchedata/supermimic-locomujoco).

ODEMimic replaces SuperMimic's direct state prediction with an acceleration model. Positions and orientations are integrated from velocities, while body features are reconstructed through differentiable forward kinematics. The policy is trained through 32-step world-model rollouts.

## Method

For physical state $s_t$ and action $u_t$, the network predicts normalized velocity increments:

$$
a_\phi(s_t,u_t)
=
\frac{\mu_{\Delta v}+\sigma_{\Delta v}\odot f_\phi(\Phi(s_t),u_t)}{\Delta t}.
$$

Here, $\Phi$ constructs and normalizes full-state features, including body kinematics. The scaling statistics are fitted on simulator data and frozen during training. Euler integration advances the physical state; root orientation is updated with quaternion integration.

Anti-saturation regularization (AS) optionally penalizes policy outputs beyond the action bounds. It can be enabled independently of the ODE world model.

See [model details](docs/method.md) for the state representation, losses and preprocessing.

## Results

Mean ± sample standard deviation over seeds **42, 43 and 44**. Each seed is evaluated on **40 LAFAN1 motions × 4 fixed starts = 160 episodes**.

### Unitree H1 · 10,000 iterations

| Method | Return ↑ | Error (m) ↓ | Failures / 160 ↓ |
|---|---:|---:|---:|
| SuperMimic | 894.32 ± 29.64 | 0.0959 ± 0.0237 | 4.67 ± 7.23 |
| + ODE Euler | **971.85 ± 29.48** | **0.0715 ± 0.0024** | 4.33 ± 7.51 |

### Unitree G1 · 20,000 iterations

G1 return is multiplied by 32 for display, giving a maximum of 1,600. Raw returns are included in the result files.

| Method | Return (×32) ↑ | Error (m) ↓ | Failures / 160 ↓ |
|---|---:|---:|---:|
| SuperMimic | 1266.78 ± 38.51 | 0.0385 ± 0.0021 | 1.00 ± 1.00 |
| + AS | 1247.27 ± 53.29 | 0.0429 ± 0.0011 | **0.00 ± 0.00** |
| + ODE Euler | **1296.97 ± 44.71** | **0.0360 ± 0.0007** | **0.00 ± 0.00** |
| + ODE Euler + AS | 1293.21 ± 14.25 | 0.0363 ± 0.0008 | 0.67 ± 0.58 |

[Per-seed results](results/per_seed.csv) · [Learning curves](results/g1_learning_curves.csv) · [Evaluation records](results/evaluations)

## Running the code

Use Python 3.12 with CUDA. Install H1 and G1 dependencies in separate environments.

### H1

```bash
pip install -r requirements/h1.txt
git clone https://github.com/robfiras/loco-mujoco.git external/loco-mujoco
git -C external/loco-mujoco checkout 3921fedbca93876c0595c90c535b24ec08ff623d
pip install -e external/loco-mujoco
```

The H1 training entry point takes a prepared asset directory containing model configuration, parameters and frozen statistics. With assets at `cache/h1_euler_s42`:

```bash
python scripts/prepare_h1_cache.py --out cache/h1_references.npy
python scripts/train_h1.py --assets cache/h1_euler_s42 --gpu 0 \
  --iterations 10000 --out runs/h1_euler_s42
```

The motion list is in [configs/lafan1_40.json](configs/lafan1_40.json). See the [H1 protocol](docs/protocol_h1.md) for environment and evaluation settings.

### G1

Prepare motion CSVs retargeted to the 29-DOF G1, then convert them and launch training:

```bash
pip install -r requirements/g1.txt
pip install 'git+https://github.com/mujocolab/mjlab.git@8bb2bc9e187797689ec82c78c35019afd2fd0376'

python scripts/convert_g1_motions.py --input data/g1_csv --out data/g1_npz --input-fps 30
python scripts/train_g1.py --method euler --data data/g1_npz --seed 42 \
  --updates 20000 --gpu 0 --out runs/g1_euler_s42
python scripts/evaluate_g1.py --ckpt runs/g1_euler_s42/ckpt_update_0020000.pt \
  --data-dir data/g1_npz --out runs/g1_euler_s42/eval_20k.json
```

Use `--method baseline` for SuperMimic. For AS, add `--saturation-weight 0.001` to Euler or `--saturation-weight 0.01` to the baseline. See the [G1 protocol](docs/protocol_g1.md) for control and evaluation settings.

## Acknowledgements

This work extends [Alchedata/supermimic-locomujoco](https://github.com/Alchedata/supermimic-locomujoco), a reproduction of **SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**. We thank the authors of SuperMimic, its public reproductions, LocoMuJoCo and mjlab. See [NOTICE.md](NOTICE.md) for source attribution.

