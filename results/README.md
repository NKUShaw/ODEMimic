# Evaluation records

`per_seed.csv` contains completed, fixed-budget evaluations. `aggregate.csv` includes a row only when seeds 42, 43 and 44 are all available at the same budget. Standard deviations use `ddof=1` across seeds.

- Return is the mean episode sum over 160 episodes, not a sum over all episodes.
- H1 display return equals native raw return.
- G1 display return equals 32 × raw return; both are retained.
- Error is in meters. Failures are counts per 160 episodes.
- Nominal transitions equal the regular collection budget; warmup and replay refills are excluded.
- `evaluations/` stores the complete episode reports with machine paths removed.
- `provenance.json` retains numerical arguments, source hashes and hashes of both original and sanitized evaluation files.
- `g1_learning_curves.csv` contains fixed-checkpoint seed-level curves, including continuations.

Reports are a snapshot exported on 2026-10-29. Ongoing seed groups are not promoted to complete aggregate rows.
