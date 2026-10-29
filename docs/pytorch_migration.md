# H1 JAX → PyTorch

The migration changes the learner implementation while retaining the H1 simulator and task configuration. The network sizes, frozen data statistics, state representation, acceleration scale, quaternion integration, termination probabilities and critic bootstrap are preserved.

## Checks

For both Tanh+AS and Euler+AS, a fixed checkpoint and 32 real starts were evaluated in both frameworks. Checks cover FK features, next physical state, termination logit, raw actions, eight-step WM loss, 32-step policy loss, first-action gradient and complete policy gradient. Relative-L2 thresholds were declared as 1e-4 for forward values/losses and 1e-2 for gradients. The policy gradient also requires cosine above 0.999.

All checks passed. Separate same-gradient checks restored each of the three Optax Adam states and compared one update against PyTorch Adam. Maximum parameter differences were at most 2.4e-7. Native-reference caching matched the original lookup exactly on the checked starts.

Reports are in `results/validation/`. Raw complete-iteration timing rows are in `results/runtime/`.

## Execution

The current PyTorch learner uses microbatch 4096, with total actor batch 4096 and horizon 32. It precomputes fixed FK transforms, skips zero-anchor operations, shares overlapping site features, and reuses successor FK at the next field evaluation. It keeps Python gradient accumulation when the microbatch is smaller than the total batch.

The shared MLP forward has multiple valid variants: WM fitting, frozen-WM policy differentiation, collection and evaluation. Dynamo's recompilation limit is set to 32 to keep these variants compiled rather than fall back to eager execution.

The reported complete-iteration times include collection, ten WM updates, one actor update and twenty critic updates. They exclude initial compilation and periodic evaluation/checkpoint overhead. The separate actor-only benchmark is not a complete-iteration measurement.

## Checkpoints

`scripts/convert_h1_checkpoint.py` reads a trusted JAX optimizer checkpoint and exports weights, Adam moments/counters, frozen normalization, collection RNG and curriculum. Dense kernels are transposed; LayerNorm epsilon remains 1e-6. `scripts/train_h1.py` resumes the exported state and rebuilds replay.

PyTorch checkpoints contain model and optimizer state, counters, RNG and curriculum. `SIGUSR1` or `SIGTERM` requests a save and exit at the next completed iteration. Every 500 iterations the learner saves an immutable checkpoint and runs the same native 160-episode evaluator.
