# ODEMimic

**Distilling Physics Simulators into Structured Neural ODE World Models**

ODEMimic builds on **[Alchedata/supermimic-locomujoco](https://github.com/Alchedata/supermimic-locomujoco)**. We study an acceleration-based Neural ODE world model inside SuperMimic's collect–fit–improve loop for humanoid motion imitation.

The dynamics network predicts accelerations. Euler integration updates the physical state, and differentiable forward kinematics reconstructs body features. The policy improves through 32-step imagined rollouts. Anti-saturation regularization (AS) is an optional component.

## Results

Each row is **mean ± sample standard deviation over seeds 42, 43 and 44**, at a fixed checkpoint. Each seed is evaluated on 40 LAFAN1 motions × 4 fixed starts = 160 episodes.

### Unitree H1 · 10,000 iterations

| Method | Return ↑ | Error (m) ↓ | Failures / 160 ↓ |
|---|---:|---:|---:|
| SuperMimic | 894.32 ± 29.64 | 0.0959 ± 0.0237 | 4.67 ± 7.23 |
| + ODE Euler | **971.85 ± 29.48** | **0.0715 ± 0.0024** | 4.33 ± 7.51 |

Euler improves mean return by **8.67%** and reduces tracking error by **25.53%**. 

### Unitree G1 · 20,000 updates

| Method | Return (×32) ↑ | Error (m) ↓ | Failures / 160 ↓ |
|---|---:|---:|---:|
| SuperMimic | 1266.78 ± 38.51 | 0.0385 ± 0.0021 | 1.00 ± 1.00 |
| + AS | 1247.27 ± 53.29 | 0.0429 ± 0.0011 | **0.00 ± 0.00** |
| + ODE Euler | **1296.97 ± 44.71** | **0.0360 ± 0.0007** | **0.00 ± 0.00** |
| + ODE Euler + AS | 1293.21 ± 14.25 | 0.0363 ± 0.0008 | 0.67 ± 0.58 |

Euler improves mean return by **2.38%** and reduces tracking error by **6.35%**. AS weights are 0.01 for the MLP and 0.001 for Euler in this table. AS is not a uniform improvement over Euler alone.

[Per-seed results](results/per_seed.csv) · [G1 learning curves](results/g1_learning_curves.csv) · [Raw evaluations](results/evaluations) · [Provenance](results/provenance.json)

## Method

The physical state contains root position/orientation, joint positions, and their velocities. A learned field predicts generalized accelerations, with frozen real-data statistics defining their scale:

$$
\mathbf a_\phi = \frac{\mu_{\Delta v}+\sigma_{\Delta v}\odot f_\phi(\mathrm{Normalize}(\Phi(s)),u)}{\Delta t}.
$$

Euler updates positions using current velocities and updates velocities using the learned acceleration. Root orientation uses quaternion integration. FK is rebuilt at the successor and reused for the next field evaluation, with gradients connected throughout.

- **H1:** 51-D physical state, 267-D full-state features, 25 acceleration outputs and a termination logit.
- **G1:** 71-D physical state, 281-D full-state features, 35 acceleration outputs and a separate learned termination head. The reproduction's domain-parameter conditioning is retained.
- **AS:** `weight * mean(relu(abs(raw_action) - threshold)**2)` at real replay starts. It changes the actor objective, not the simulator reward or critic targets.

[Dynamics and preprocessing](docs/method.md) · [H1 protocol](docs/protocol_h1.md) · [G1 protocol](docs/protocol_g1.md)

## H1: PyTorch learning, unchanged simulator

The current learner runs WM, policy, critic, differentiable FK, BPTT and Adam in PyTorch. LocoMuJoCo/MJX still runs the simulator. The JAX reference backend is retained for checkpoint bootstrap and numerical checks.

Complete-iteration short tests on one L40S, excluding initial compilation:

| Method | JAX microbatch | JAX seconds | PyTorch microbatch | PyTorch seconds |
|---|---:|---:|---:|---:|
| Tanh + AS | 1024 | 1.945 | 4096 | **1.600** |
| Tanh + Euler + AS | 2048 | 2.452 | 4096 | **2.080** |

Total actor batch stays 4096. These times include real collection, ten WM updates, one actor update and twenty critic updates. Both PyTorch short tests completed 12 updates with finite values. Forward predictions, WM/actor losses, action gradients, policy gradients and restored Adam updates were checked against JAX. [Migration details](docs/pytorch_migration.md)

## Setup

Use separate environments for H1 and G1: the tested MuJoCo versions differ. Python 3.12 and CUDA are used for training. Replay requires substantial host RAM.

### H1

```bash
python3.12 -m venv .venv-h1
source .venv-h1/bin/activate
pip install -r requirements/h1.txt
git clone https://github.com/robfiras/loco-mujoco.git external/loco-mujoco
git -C external/loco-mujoco checkout 3921fedbca93876c0595c90c535b24ec08ff623d
pip install -e external/loco-mujoco
```

The 40 motion names are in `configs/lafan1_40.json`. LocoMuJoCo handles the H1 assets. An optional `--variables configs/variables.yaml` can point at local dataset/model caches.

Start a JAX reference run from scratch, or use an existing H1 checkpoint:

```bash
python scripts/train.py --method ode --solver euler --seed 42 --gpu 0 \
  --iterations 5000 --out runs/h1_euler_jax_s42

python scripts/convert_h1_checkpoint.py \
  runs/h1_euler_jax_s42/checkpoints/iter_5000.pkl --out cache/h1_euler_s42
python scripts/prepare_h1_cache.py --out cache/h1_references.npy

python scripts/train_h1.py --assets cache/h1_euler_s42 --gpu 0 \
  --iterations 10000 --out runs/h1_euler_torch_s42
```

The converter restores all three Adam states/counters, normalization, curriculum and RNG. Replay is refilled on continuation. The PyTorch learner supports H1 MLP and Euler checkpoints with Tanh actions; Midpoint remains available through the JAX reference backend. Use `--saturation-weight 0.01` on the initial reference run to create an AS run. Resume a PyTorch run with the same `--assets` plus `--resume-torch runs/.../latest.pt`.

### G1

```bash
python3.12 -m venv .venv-g1
source .venv-g1/bin/activate
pip install -r requirements/g1.txt
pip install 'git+https://github.com/mujocolab/mjlab.git@8bb2bc9e187797689ec82c78c35019afd2fd0376'

python scripts/convert_g1_motions.py --input data/g1_csv --out data/g1_npz --input-fps 30
python scripts/train_g1.py --method euler --data data/g1_npz --seed 42 \
  --updates 20000 --gpu 0 --out runs/g1_euler_s42
python scripts/evaluate_g1.py --ckpt runs/g1_euler_s42/ckpt_update_0020000.pt \
  --data-dir data/g1_npz --out runs/g1_euler_s42/eval_20k.json
```

G1 input CSVs must already be retargeted to the 29-DOF robot. The converter resamples and computes body kinematics; it does not retarget human joints. Use `--method baseline` for SuperMimic and `--saturation-weight 0.001` for the reported Euler+AS configuration.

## Layout

```text
odemimic/h1/   PyTorch learner, FK, replay and Adam conversion
odemimic/g1/   G1 world model, policy, simulator adapter and training loop
supermimic/   inherited H1 JAX reference/bootstrap implementation
scripts/      training, conversion and evaluation entry points
configs/      motion manifest
results/      fixed-budget evaluations, statistics and validation reports
docs/         method, protocol and migration notes
```

Weights, motion data and machine-specific job queues are not stored in Git. The training scripts save checkpoints and per-seed evaluation records. The historical H1 results and the current PyTorch continuation phase are distinguished in the provenance.
