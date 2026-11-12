# Why the cube-state estimator does not improve the policy: diagnostics

Companion to [state_estimation.md](state_estimation.md). Every number below comes from runs executed on 2026-11-11/12. No new PPO training was run for these diagnostics. All evaluations use 2048 envs × 600 steps (30 s), held-out seeds 5001 and 5002, and the deterministic policy mean, unless stated otherwise. Raw outputs are in `training_source/project/data/eval/diagnostics/`.

Policies:

| Name | Description |
|---|---|
| A | The original `model_400.pt` |
| A′ | `model_400` fine-tuned for 300 iterations without the estimator |
| C | `model_400` fine-tuned for 300 iterations with the full 19-D estimator features |
| D | `model_400` fine-tuned for 300 iterations with contacts only |

## 1. Estimator ablation: does policy C rely on the estimator?

The features are replaced inside C's actor at evaluation time (`evaluate_policies.py --ablation`). Nothing else changes: same weights, seeds and per-env randomization.

- **zero:** the features are set to 0.
- **shuffle:** each env receives the features of env (i + N/2) mod N, which has its own randomization and trajectory.
- **mean:** the features are replaced by their mean over the test split.

The table shows the paired per-env difference from the normal run (mean ± 95% CI; 4096 envs per preset). `baseline repeat` is a second run with the normal estimator; it gives the run-to-run noise floor, because the GPU simulation is not bit-reproducible.

| Preset | Run | Δ turns | Δ failure rate | Δ survival [s] |
|---|---|---|---|---|
| nominal | zero | −0.005 ± 0.007 | +0.001 ± 0.010 | +0.02 ± 0.18 |
| nominal | shuffle | −0.001 ± 0.007 | +0.000 ± 0.010 | −0.01 ± 0.18 |
| nominal | mean | −0.001 ± 0.007 | +0.007 ± 0.010 | −0.08 ± 0.18 |
| nominal | baseline repeat | −0.001 ± 0.007 | −0.002 ± 0.010 | −0.01 ± 0.18 |
| heavy_slippery | zero | −0.001 ± 0.006 | +0.004 ± 0.005 | −0.04 ± 0.12 |
| heavy_slippery | shuffle | −0.001 ± 0.006 | −0.000 ± 0.005 | +0.03 ± 0.12 |
| heavy_slippery | mean | −0.001 ± 0.006 | −0.000 ± 0.005 | −0.01 ± 0.12 |
| heavy_slippery | baseline repeat | −0.003 ± 0.006 | +0.000 ± 0.005 | +0.02 ± 0.11 |

Baseline levels:
- **nominal:** 0.288 turns, failure rate 0.561, survival 15.8 s.
- **heavy_slippery:** 0.234 turns, failure rate 0.469, survival 17.0 s.

**Conclusion:** policy C does not rely on the estimator. Removing or scrambling its information changes nothing beyond run-to-run noise. The test could have detected an effect of about ±0.007 turns or ±1% failure rate.

## 2. Is the estimate informative in difficult situations?

This uses `diagnose_estimator_events.py` on the held-out test split, which covers unseen seeds 3001 and 3002 and 1.22 M valid rows. Situations are defined from ground truth:

| Subset | Definition |
|---|---|
| slip (translation) | Cube speed relative to the palm > 3 cm/s |
| slip (tumble) | Off-axis angular speed \|ω_xy\| > 1 rad/s |
| drift 10–20 mm | Cube 10–20 mm from its position at the start of the episode |
| drift > 20 mm | Cube more than 20 mm from its starting position |
| near failure | The last 0.5 s before any failure termination |
| near drop | The last 0.5 s before a drop |

R² is computed within each subset, so it is measured against an oracle that knows the subset's mean. Drift and tilt are computed from the estimator alone: the current estimate is compared with the estimator's own estimate at the start of the episode, as a policy would have to do. The linear baseline is a ridge regression on the same 320 inputs.

| Subset | Share of rows | Position RMSE [mm] | Drift error [mm] | Tilt error [°] | Lin-vel R² | ω_xy R² | Yaw-rate R² | Contact F1 |
|---|---|---|---|---|---|---|---|---|
| all | 100% | 4.7 (ridge 10.9) | 2.3 (ridge 8.1) | 3.3 (ridge 17.1) | 0.41 (ridge 0.15) | 0.33 | 0.39 | 0.76 (ridge 0.61) |
| slip (translation) | 10.0% | 5.1 | 2.3 | 3.6 | 0.39 | 0.37 | 0.41 | 0.67 |
| slip (tumble) | 6.4% | 5.1 | 2.2 | 3.6 | 0.40 | 0.37 | 0.36 | 0.61 |
| drift 10–20 mm | 46% | 4.6 | 2.2 | 3.2 | 0.32 | 0.25 | 0.30 | 0.76 |
| drift > 20 mm | 27% | 5.5 | 2.8 | 3.6 | 0.26 | 0.20 | 0.28 | 0.75 |
| near failure | 2.4% | 6.5 | 2.7 | 5.7 | 0.51 | 0.43 | 0.42 | 0.73 |
| near drop | 30 rows | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Position R² is 0.91–0.93 in every subset. Drops are too rare in this data (0.2% of episodes) for a near-drop row breakdown.

Event detection (AUROC over all valid rows):

| Event | Estimator | Ridge | Joint-motion heuristic | True state (upper bound) |
|---|---|---|---|---|
| Drift > 10 mm | **0.95** (estimated drift) | 0.71 | 0.43 | n/a |
| Translational slip | 0.75 (estimated speed) | 0.68 | 0.69 | n/a |
| Tumble | 0.73 (estimated ω_xy) | 0.65 | 0.69 | n/a |
| Failure within 0.5 s | 0.46–0.69 | 0.41–0.43 | 0.69 | 0.47–0.66 |

**Conclusions:**
- **Position, drift and tilt:** the estimator stays accurate during slips, large drift and just before failures. It clearly beats a linear model and proprioceptive heuristics.
- **Velocities:** these are weak everywhere. During slips the lin-vel R² is about 0.4, and the RMSE (5.6 cm/s) is about the same size as the slip speed itself (5.8 cm/s).
- **Failures can't be predicted from cube state:** even the *true* cube state predicts a failure 0.5 s ahead with an AUROC of only 0.47–0.66. Average drift and tilt barely change in the 2 s before a failure; see `lead_time_before_failure.png`. Section 4 explains why.

## 3. Feature influence on the actor

This is measured on 95k states visited by C (`diagnose_feature_influence.py`).

**Features are on a normal scale.** Position and velocity features have a standard deviation of 0.25–0.98, rotation columns 0.13–0.59, and contact probabilities 0.13–0.33. The normalized 320-D observation has a standard deviation of 0.85 on average. Scaling is therefore not the problem.

**Their influence is genuinely small,** and this does not rest on weight magnitudes alone:

| Measure | Value |
|---|---|
| Share of first-layer pre-activation variance coming from the features | 6.7 × 10⁻⁶ |
| Share of gradient × std attribution (executed action) | 0.12%; the joint-position history has 44%, previous commands 56% |
| Executed-action change, features zeroed | mean 0.012 rad per entry; 5.1% of entries change by more than 0.01 |
| Same, features shuffled / replaced by their mean | 0.011 / 0.008 rad |
| Reference: one extra draw of the ±0.01 rad training observation noise | 0.041 rad |
| Reference: swapping C's network for A′'s | 0.028 rad |
| Reference: swapping C's network for A's (`model_400`) | 0.069 rad |

**Why the influence is small.** These are properties of the training setup, not of the estimator:

1. **Actions are saturated.** The action term is `JointPositionDeltaAction` with scale 1, and actions are clipped to ±1.
   - The actor's raw outputs have a median magnitude of 12.8.
   - 95% of action entries lie outside the clip range, and 85% lie more than 3σ beyond it. The policy is essentially a bang-bang controller.
   - Only about 5% of entries can be affected by *any* input.
2. **Exploration does nothing.** The exploration noise has σ = 0.75. In 94% of samples, `clip(mean + σε)` equals `clip(mean)`, so PPO gets almost no learning signal from most action dimensions.
3. **The learning rate was stuck at its minimum.** The adaptive-KL schedule (desired KL 0.01) held the learning rate at its floor of 1e-5, against the configured 3e-4, for all 300 iterations of both A′ and C.
   - Mean changes on outputs of magnitude 10–30 exceed the KL target even at that rate.
   - The whole network moved about 1% (relative weight change 0.7–1.0% per layer, identical in A′ and C).
   - The new estimator columns started at zero and grew to a norm of 0.049. That is 3.6× the change of the original input columns (0.013), so PPO was not ignoring them. The fine-tune as a whole was simply too small.
4. **Fine-tuning changed nothing measurable.** A, A′, C and D are within about 0.01 turns and 1–2% failure rate of each other on every preset. The rise in reward and episode length during training came from the reset-anchor transient settling, not from learning.

## 4. Why the evaluation gives about 0.29 turns in 30 s while the reference shows 1.31

The 1.31 comes from the README and the portable replay. It is a single 30 s rollout of `model_400` (seed 11, play mode, original task), stored in `reference/exact_mjlab_seed11_state_trajectory.npz`. Two things separate it from 0.29.

**(a) About 0.33 of the 1.31 turns are a reset artifact.**
- Frames 0–104 contain 44 frames with jumps of up to 96 mm and 116° per frame. These are repeated environment resets caused by the stale reset-anchor bug, each of which teleports the cube to a new random yaw. Unwrapping yaw across those teleports counts them as rotation.
- Real rotation is 0.98 turns: 0.89 turns from a single stable 24.8 s episode (frames 105–600, about 0.22 rad/s) plus about 0.09 from short episodes.

**(b) The evaluation averages over 4096 randomized episodes and stops counting at the first termination.** Most terminations are themselves an artifact.
- The `cube_pose_deviation` tilt check uses the norm of XYZ-Euler roll/pitch differences. Near |pitch| = 90°, a few degrees of real tilt changes roll/pitch by tens of degrees.
- Physical tilt needed to trigger the 45.8° (0.8 rad) threshold, by the cube's starting |pitch| (7194 test starts):

  | Starting \|pitch\| | Share of starts | Median physical tilt to terminate |
  |---|---|---|
  | 0–30° | 62% | 43.5° |
  | 30–60° | 15% | 27.5° |
  | 60–75° | 13% | 16° |
  | 75–85° | 9% | 7.5° |
  | 85–90° | 2% | 3° |

- Termination outcomes by starting |pitch| (5149 finished test episodes):

  | Starting \|pitch\| | Ended by pose-deviation | Real tilt at the last recorded step |
  |---|---|---|
  | < 30° | 32% | ~45° (genuine) |
  | 30–60° | 98% | 32° |
  | 60–75° | 100% | 16° |
  | 75–90° | 100% | 7° |

  The cube wobbles by about 13° during normal rotation.
- The reference's long episode began at |pitch| = 19.7°, a benign orientation.

Effect on the measured result for `model_400`, nominal preset, held-out seeds:

| Protocol | Turns in 30 s | Failure | Survival | Completed 30 s | Turns if completed | Share ≥ 0.89 turns |
|---|---|---|---|---|---|---|
| Original task (stale anchor) | 0.044 | 92.8% | 2.5 s | n/a | n/a | n/a |
| Fixed anchor, Euler tilt (used for the A/A′/C/D table) | 0.284 | 56.6% | 15.6 s | 43% | 0.57 | 5% |
| Same, observation noise off (play mode) | 0.291 | 55.1% | 16.2 s | 45% | 0.56 | 6% |
| Fixed anchor, **physical tilt** (new opt-in variant) | **0.485** | 37.5% | 22.4 s | 62% | 0.67 | 17% |
| Pose termination removed (drops still end episodes) | 0.649 | 3.6% (drops 3.3%) | 29.2 s | 96% | 0.67 | 24% |

Observation noise is not the cause. Under the corrected (physical-tilt) termination, model_400 achieves 0.49 turns per 30 s on average. A quarter of episodes reach the reference's 0.89 genuine turns when the pose termination is removed. Policy C under the same two protocols gives 0.495 and 0.645 turns, the same as A.

## 5. Reproduce

Run from `training_source/project` with the `leap-estimator` env active and `MUJOCO_GL=egl`.

```bash
C=logs/rsl_rl/leap_right_mbzuai_rotate_estimator_full/2026-11-11_21-41-48_estimator_full_finetune/model_699.pt
A2=logs/rsl_rl/leap_right_mbzuai_rotate_fixedreset/2026-11-11_21-35-11_control_finetune/model_699.pt
```

Experiment 1, ablation:

```bash
for m in none zero shuffle mean; do
  python scripts/estimation/evaluate_policies.py --policy C_abl_$m=$C --ablation $m --seeds 5001 5002 \
    --presets nominal heavy_slippery --output-dir data/eval/diagnostics/ablation/$m
done
python scripts/estimation/analyze_ablation.py --run none=data/eval/diagnostics/ablation/none \
  --run zero=data/eval/diagnostics/ablation/zero --run shuffle=data/eval/diagnostics/ablation/shuffle \
  --run mean=data/eval/diagnostics/ablation/mean --null data/eval/policies/C_estimator \
  --output data/eval/diagnostics/ablation/summary.json
```

The baseline-repeat comparison should use only the nominal and heavy_slippery files of the main C run.

Experiment 2, estimator in difficult situations:

```bash
python scripts/estimation/diagnose_estimator_events.py --estimator data/estimator_models/cube_state_estimator.pt \
  --output-dir data/eval/diagnostics/estimator_events
```

Experiment 3, feature influence:

```bash
python scripts/estimation/diagnose_feature_influence.py --checkpoint $C --reference-a-prime $A2 \
  --reference-model400 ../../checkpoint/model_400.pt --output-dir data/eval/diagnostics/influence_C
```

Termination and noise protocols:

```bash
for v in physical_tilt no_pose_termination; do
  python scripts/estimation/evaluate_policies.py --policy A_$v=../../checkpoint/model_400.pt --policy C_$v=$C \
    --env-variant $v --seeds 5001 5002 --presets nominal --output-dir data/eval/diagnostics/termination/$v
done
for v in fixed original; do
  python scripts/estimation/evaluate_policies.py --policy A_noiseoff_$v=../../checkpoint/model_400.pt \
    --obs-noise off --env-variant $v --seeds 5001 5002 --output-dir data/eval/diagnostics/noise_off/$v
done
```
