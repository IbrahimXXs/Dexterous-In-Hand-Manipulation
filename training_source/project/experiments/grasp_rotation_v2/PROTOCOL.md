# Grasp-Rotation-V2 protocol (pre-registered)

Simulation only. Baselines: hardware-limited A seeds 1–3 (selected checkpoints, `../hwlimit_ac/checkpoints/`) and Retention-V1 R1 (selected checkpoint 2600, `../retention_v1/checkpoints/`). Completed experiments are read-only.

All rollouts use the task `Mjlab-Leap-RetV1-A`. Its dynamics, observations, actions, limits and domain randomization are identical to `Mjlab-Leap-HwLim-A` (checked by `tests/test_retention_v1.py`); it only adds reward terms, which are logged and never affect a rollout. Penalty weights are frozen at the A seed-1 curriculum values at update 3000 (`grasp_rotation_v2/rollout.py: FINAL_CURRICULUM`), as in the Retention-V1 audit.

Metrics are those of `hwlimit_study.evaluation.evaluate`: first episode only, timeout beyond the horizon, uninterrupted turns = genuine yaw while held / 2π, failure = cube_fell, cube_too_fast or cube_pose_deviation, survival = held steps × 0.05 s, yaw rate while held = rotation / survival.

## Seeds

| Use | Seeds | Status |
|---|---|---|
| Phase 1 development: per-grasp ranking | 6101, 6102 | new |
| Phase 1 development: perturbation tolerance | 6103 | new |
| Phase 1 and Phase 4 held-out confirmation | 6201, 6202 | new; not looked at until selection is frozen |
| Historical comparability only | 5001, 5002 | used by the earlier studies; not used for any selection |
| PPO validation (Phase 3 checkpoint selection) | 7001 | as in A and Retention-V1 |

## Phase 1: grasp initialization (no training)

**Grasp assignment.** `rollout.assigned_reset` replaces the uniform cache draw by a fixed per-environment cache index. All other reset and randomization code is the task's own.

**Development rollouts (P1-dev).**
- Seeds 6101 and 6102, nominal preset, 4096 environments, 60 s (1200 steps).
- Environment i starts from grasp i mod 239, so each grasp gets 17–18 environments per seed.
- Policies: A1, A2, A3, R1 (selected), plus the zero-action hold policy (descriptive only).

**Per-grasp metrics** (seeds pooled, per policy p): T30 and T60 (uninterrupted turns in 30 and 60 s), P30 and P60 (share still held), yaw rate while held, failure time and type.

**Grasp score:** S_g = mean over the four policies of T60. It rewards rotation and retention together; a grasp that is held but not rotated scores near 0.

**Eligibility** (all must hold):
1. Mean P60 over the four policies ≥ 0.80.
2. Mean yaw rate while held ≥ the median over all 239 grasps (no slow-but-safe grasps).
3. Each of the four policies has T60 at or above that policy's median over all grasps (consistency across policies).

**Reliability:** report the split-half correlation of S_g between seeds 6101 and 6102. If it is below 0.5, the ranking is too noisy to pick single grasps and only subset-level conclusions are drawn.

**Robust subset C:** greedy over eligible grasps in order of S_g. A grasp is skipped when its joint command is within 0.05 rad RMS (over 16 joints) of an already chosen grasp. Stop at 10 grasps.

**Perturbation tolerance** (seed 6103):
- The first 5 grasps of the subset, 60 s × the four policies. One rollout per (policy, level) with 2050 environments, about 410 per grasp (environment i gets subset grasp i mod 5). Amended 2026-11-20 before any perturbation rollout was run; originally 2048 environments per grasp.
- Cube pose perturbations (uniform, via the event's own `pose_range`), one level per run:
  - L0: none.
  - L1: ±1 mm x/y/z, ±5° yaw.
  - L2: ±2 mm x/y, ±1 mm z, ±10° yaw, ±3° roll/pitch.
  - L3: ±4 mm x/y, ±2 mm z, ±20° yaw, ±5° roll/pitch.
- L2 is the pre-declared "realistic placement" level.
- Joint states are not perturbed: the motors reproduce the commanded pose to ≤ 0.042 rad (17 Nov function test). This is a limitation.

**Nominal grasp B:** the subset grasp with the highest S at L2 (mean over the four policies).

**Held-out confirmation** (seeds 6201 and 6202, 2048 environments each, 60 s; 30-s metrics from the same rollouts), for each of A1, A2, A3 and R1:
- A: full original distribution (uniform cache draw).
- B: nominal grasp.
- C: robust subset (environment i gets subset grasp i mod 10).
- D: nominal grasp with L2 perturbation.

Robustness presets (heavy_slippery, high_delay, obs_noise_x3; seed 6201) are run for conditions A and B.

**Phase 1 decision rule** (held-out, nominal preset):
- A grasp-initialization improvement is reported as significant when B or C beats A on the mean over the four policies on all of:
  - uninterrupted turns in 30 s, +15% or more;
  - lower failure rate, with the 95% bootstrap CI (over environments) excluding 0;
  - all four policies improving individually.
- It is reported as realistic only if D keeps at least half of B's turn gain.
- Performance on the full distribution A is always reported next to it.

## 2. Phase 2: reward candidates (fixed 2026-11-20, after the reward audit and before any V2 training)

**Evidence** (`results/reward_analysis.json`, P1-dev rollouts, seeds 6101+6102, all grasps equally):

- **R1 rotates slower in the same states.** In A's state occupancy (tilt × palm-position cells, 100% of held steps covered), R1's yaw rate while held is 0.285 rad/s against A's 0.323. A in R1's occupancy reaches 0.319.
- **Stalling.** R1 spends 43% of held steps below 0.25 rad/s (4-step rate), A3 21%. Stalled steps (< 0.1 rad/s) carry the highest 2-s failure hazard (1.3–4.4%); 0.25–0.55 rad/s carries the lowest (0.5–0.7%).
- **20-s episode return of the same trajectories** under each reward variant (A mean − R1; positive = the reward prefers A's faster behaviour):
  - Retention-V1: −0.050.
  - A reward: +0.316.
  - Grasp loss only (H3): +0.263.
  - Tilt part + grasp loss, no position part (H1): +0.282.
  - Position part × 0.25: +0.199.
  - Tilt margin 0.6 rad (H2): −0.066.
  - Yaw clip 0.5 (H4): +0.047.

  The position part is the component that turns the preference from A to R1. H2 does not change it.
- **The yaw reward is saturated** (4-step rate ≥ 0.25 rad/s) on 56% (R1) to 78% (A3) of held steps, so it does not reward faster rotation there. The drift/tilt gate (× 0.1 beyond 2 cm or 0.35 rad) is active on 46–52% of held steps for every policy. The gate is analysed but not changed in this pilot.

**Candidates.** Each is Retention-V1 with ONE change; everything else is identical (`tests/test_grasp_rotation_v2.py`).

| Id | Hypothesis | Change | Expected mechanism |
|---|---|---|---|
| P0 | H1 | Retention penalty without its palm-position part (tilt part, margin 0.5 rad, and grasp loss −10 kept) | removes the term that taxes A-like fast rotation; keeps the costs on the actual failure mode (tilt, 99.4% of failures) |
| C5 | H4 | Yaw-reward clip ±0.25 → ±0.5 rad/s (about the 90th percentile of A's 4-step rate) | restores a reward gradient for faster rotation, where the hazard is lowest; retention terms unchanged |

H2 (tilt margin) is not trained: the audit shows no effect on the preference. H3 (grasp loss only) is not trained: H1 keeps H3's grasp-loss cost and differs from it only by the tilt part, and the tilt part targets the observed failure mode.

## 3. Phase 3: training

- **Runs:** one run per candidate, PPO seed 1 (the same as A1 and R1, for a paired comparison). Tasks `Mjlab-Leap-GRV2-P0` and `Mjlab-Leap-GRV2-C5`; PPO, network, 4096 environments, 3001 updates and zero-initialised actor output exactly as A and Retention-V1. Original 239-grasp distribution. From scratch, never resumed (`scripts/sbatch_train.sh`).
- **Validation:** `scripts/curves.py`, identical to the A and Retention-V1 validation (seed 7001, nominal, 1024 environments, 30-s points every 100 updates, 60-s points every 500). The comparison with A1 and R1 is at matched updates.
- **Stage check at update 500.** Stop only for:
  - a bug or reward exploit;
  - a 30-s validation yaw rate below 50% of R1's at update 500 (0.078 rad/s);
  - numerical instability.
- **Futility check at update 1500.** Stop a candidate whose validation turns are below R1's at 1500 (1.087) minus 0.15 AND whose failure rate is above R1's (0.164) plus 0.10. Otherwise continue to 3000.
- **Selection:** max 30-s validation turns (ties → later). Held-out results are never used for selection or tuning.

## 4. Phase 4: held-out comparison

**Policies:** A1, A2, A3 (selected), R1 (selected; final where relevant), and each V2 candidate (selected).

**Setting A (main):** full original grasp distribution, seeds 6201 and 6202, 2048 environments each, 60 s (30-s metrics from the same rollouts). Presets: nominal for both seeds; heavy_slippery, high_delay and obs_noise_x3 on seed 6201.

**Setting B:** the Phase 1 nominal grasp and robust subset, same seeds. Reported separately.

**Thresholds** (Setting A, nominal, both seeds pooled; A and R1 evaluated in the same rollouts). A candidate is "promising" when ALL hold:
1. Rotation: yaw rate while held ≥ the A mean.
2. Turns: 30-s uninterrupted turns ≥ 1.05 × the A mean AND ≥ 1.10 × R1.
3. Retention: failure rate ≤ (A mean + R1) / 2, i.e. it keeps at least half of R1's failure reduction.
4. Survival: 60-s P(held) ≥ the A mean.
5. Robustness: in each robustness preset, 30-s turns ≥ 0.90 × the A mean of that preset.

**Interpretation:**
- Bootstrap 95% CIs over environments.
- One training seed is exploratory. A promising candidate is reported as such, with a recommendation for 3-seed confirmation, never as a reliable PPO improvement.
