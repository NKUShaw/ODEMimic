# G1 protocol

The current G1 campaign uses mjlab's **29-DOF G1** with PD position targets:

```text
target_joint_position = default_joint_position + action_scale * action
action_scale          = 0.25 * torque_limit / kp
```

PD gains, torque limits and collision geometry come from the pinned mjlab robot model. Training uses 4096 environments, 50 Hz control, ten WM batches of 2048 samples with an eight-step window, and an actor batch of 4096 with horizon 32. Actor microbatch is 4096. The long learning-rate decay schedule is retained when setting a short update budget.

The reported comparison uses Linear decoder output without an added final action clamp. Noise/execution handling is shared by real collection and imagination through the inherited G1 reproduction. AS is optional; its weight is 0.01 for MLP and 0.001 for Euler in the completed 20k table.

Evaluation uses 40 motions × 4 fixed random starts, seed 20261027, with 500 controls/10 seconds. Policy execution is deterministic and evaluation disables observation/reset noise and domain randomization. Return and tracking error stop at the first termination. Numerical failures are counted as failures.

Raw return has a nominal ceiling of 50. The README shows `32 * raw_return`, giving a display ceiling of 1600. The CSV stores both values. Error and failures are not rescaled. These are G1's own reward/control settings; the H1 table uses its native LocoMuJoCo protocol.

Source dependencies:

- mjlab `8bb2bc9e187797689ec82c78c35019afd2fd0376`.
- MuJoCo/MuJoCo-Warp 3.16.0 in the tested G1 environment.
- Initial reproduction: CuboiLeo/SuperMimic-Reproduced `ba6105349069bf040f1fdad60873d13138d505a2`.
- Our current full-state processing is documented in [method.md](method.md).

Motion CSVs must already be retargeted to the 29-DOF G1. `scripts/convert_g1_motions.py` resamples them to 50 Hz and exports joint/body trajectories for the trainer.
