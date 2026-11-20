# Grasp-Rotation-V2: grasp optimisation and the retention–rotation balance

Simulation only (MJLab / MuJoCo-Warp, RSL-RL PPO), 20 Hz, under the deploy kit's approved joint limits. The real hand, the deploy kit, joint limits, current limits, gains and safety guards were not touched. Baseline commit `a2c01e2` (Retention-V1). Pre-registered protocol: [PROTOCOL.md](PROTOCOL.md) (sha256 in `logs/protocol.sha256`; the Phase 1 version is in `logs/protocol_history.txt`).

## Summary

**Outcome C: both improvements help, and they are reported separately below.** One training seed per candidate, so every PPO result here is exploratory.

| Held-out, 30 s (seeds 6201+6202, 4096 envs per cell) | turns | failure | yaw rate held (rad/s) | P(held 60 s) |
|---|---|---|---|---|
| A1–A3 mean, full grasp distribution (reference) | 1.200 | 25.2% | 0.295 | 0.732 |
| R1, full distribution | 1.158 | 19.0% | 0.263 | 0.763 |
| **Grasp effect only:** A1–A3 + R1 mean, nominal grasp 49 | 1.599 (+34%) | 0.7% | 0.336 | 0.991 |
| **Reward effect only:** C5, full distribution | 1.869 (+56% vs A, +61% vs R1) | 19.9% | 0.432 | 0.788 |
| P0, full distribution | 1.273 | **11.5%** | 0.282 | **0.862** |
| **Combined:** C5, nominal grasp 49 | **2.466** (+105% vs A on the full distribution) | 0.1% | 0.517 | 0.996 |

- **Grasp initialisation (Phase 1).** Starting every episode from one robust grasp (cache index 49), or from a 10-grasp robust subset, raises the turns of every existing policy by 31–37% and cuts failure from about 24% to about 1%, with no retraining. The gain survives ±2 mm / ±10° placement noise (condition D keeps 94% of it). This is a grasp-initialisation improvement, not a PPO improvement.
- **Reward balance (Phases 2–4).** R1 rotates slower than A because the *position* part of its retention penalty taxes the drift that fast rotation causes, and the ±0.25 rad/s yaw clip gives no reward for going faster. Two from-scratch candidates were trained:
  - **C5** raises the yaw clip to ±0.5 rad/s and keeps R1's retention terms. It passes all five pre-registered held-out thresholds: +0.67 turns vs the A mean [CI +0.64, +0.70], +0.71 vs R1. Its failure rate equals R1's (+0.009 [−0.009, +0.027]) and is 5 points below A's.
  - **P0** drops the position part. It has the lowest failure rate of any policy (11.5% vs R1 19.0%, CI excludes 0). Its yaw rate is below A's, so it fails thresholds 1–2 and is not "promising" under the pre-registered rule.
- **Recommendation:** confirm C5 with three training seeds (not launched). Use grasp 49 as the standard start for repeatable real-hand evaluation once separately approved; that needs a kit start-pose change, which is not made here. Then, per Outcome C, validate on hardware with RealSense + ArUco cube tracking.

## Step 0: setup and preservation

- Git was synced at the start (fast-forward, no local modifications at risk). All new code is in new folders; no tracked file outside them differs from `a2c01e2`.
- `scripts/check_preservation.sh` → `logs/preservation_check.txt`:
  - no file outside this study differs from the baseline;
  - all 730 earlier RSL-RL checkpoints are unchanged (669 from Retention-V1's baseline list plus 61 Retention-V1 run checkpoints; the R1 selected/final hashes equal `retention_v1/checkpoints/MANIFEST.txt`);
  - the grasp cache, estimator B, the 6 kit symlinks and the three datasets are unchanged.
- Tests: full suite 82 passed (`tests/test_grasp_rotation_v2.py` adds 5), log in `logs/pytest_full.log`.

New code:

| Path | Purpose |
|---|---|
| `src/in_hand_rotation_mjlab/grasp_rotation_v2/rollout.py` | evaluation rollouts with pinned grasp assignment and optional placement perturbation (`assigned_reset` wraps the unchanged cache reset; the env's own random draw is used when no assignment is given) |
| `src/in_hand_rotation_mjlab/grasp_rotation_v2/rewards.py` | `retention_penalty_v2`: Retention-V1's penalty with separately scaled position and tilt parts (equal to V1 at defaults, tested) |
| `src/.../config/leap_right_grasp_rotation_v2/` | tasks `Mjlab-Leap-GRV2-P0` and `Mjlab-Leap-GRV2-C5`; each differs from Retention-V1 by exactly one change (tested); PPO config identical to A/R1 apart from the experiment name |
| `scripts/` | `run_items.py` (rollout worker), `analyze_grasps.py`, `analyze_perturbation.py`, `analyze_reward.py`, `curves.py`, `freeze_and_items.py`, `compare_p1.py`, `compare_v2.py`, `check_preservation.sh`, sbatch wrappers |

Metrics (same definitions as hwlimit_ac / Retention-V1): first episode from reset; turns = genuine cube yaw while held / 2π; failure = any non-timeout termination (99%+ are the cube-pose-deviation rule); yaw rate held = yaw / survival time; frozen final-curriculum penalty weights.

Seeds: development 6101/6102 (ranking) and 6103 (perturbation); validation 7001 (training curves); held-out 6201/6202 (all reported held-out results, used once); tests 4401–4403.

## Phase 1: grasp initialisation (no PPO training)

### Grasp analysis (dev seeds, all 239 cache grasps × A1, A2, A3, R1, hold; ~17 environments per grasp per policy)

Files: `results/grasp_table.csv` (one row per grasp: palm-frame cube position and tilt, joint commands, headroom, contacts, fingertip reach, P30/P60, T30/T60, yaw rate and failure time per policy), `results/grasp_ranking.json`, `figures/p1_grasp_reliability.png`, `figures/p1_grasp_features.png`.

- Grasp quality is a property of the grasp, not of the policy. The score S (mean 60-s turns over A1–A3, R1) has split-half reliability r = 0.99, and the policies' per-grasp T60 correlate at 0.86–0.93.
- The distribution is bimodal. S quantiles (0/10/25/50/75/90/100%) are 0.09 / 0.36 / 2.16 / 2.89 / 3.02 / 3.09 / 3.21. 39 "bad" grasps (S < 1) have P60 0.05; the eligible grasps have P60 0.98.
- **What makes a grasp bad:** initial cube tilt relative to the palm (Spearman ρ = −0.70 with S; bad grasps 0.70 rad vs eligible 0.19 rad). These start near the 0.5-rad tilt margin. When the cube settles flat, the pose-deviation rule fires (Retention-V1 diagnosis), so they fail early even though the cube stays in the hand (with the rule off, 96% of R1's bad-grasp episodes are still in hand at 60 s). Other predictors: hold-policy P60 (+0.49), cube height in the palm (+0.47), distance from the kit start pose (−0.40). Ridge regression on the features gives CV R² = 0.74.
- Eligibility (pre-registered): P60 ≥ 0.80 (164 grasps), mean yaw rate ≥ median (120), every policy's T60 ≥ its median (56) → 54 eligible.

### Nominal grasp and robust subset

- **Robust subset C** (greedy top 10 by S with ≥ 0.05 rad RMS command spacing): grasps 81, 49, 192, 123, 79, 106, 196, 47, 233, 189. Dev S 3.15–3.21, P60 0.98–1.00, T30 1.56–1.60 (all grasps: S 2.38, T30 1.19, P60 0.75).
- **Perturbation tolerance** (seed 6103, top 5 × levels L0–L3, ~410 environments per grasp and level; `results/perturbation.json`). All five are robust. At L3 (the largest level) S is 2.85–3.12 and P60 0.89–0.98.
- **Nominal grasp B = cache index 49** (pre-registered rule: highest S at L2, the realistic level of ±2 mm x/y, ±1 mm z, ±10° yaw, ±3° roll/pitch). S is 3.19 / 3.17 / 3.16 / 3.12 at L0–L3, and P60 is 0.992 at L0 and 0.983 at L3.
  - Grasp 49: cube at (42.6, 1.2, −52.4) mm in the palm frame, tilt 0.20 rad, no fingertip contact at t = 0, mean fingertip distance 68 mm.
  - Index PIP is at its kit limit (1.825 rad); every other joint has ≥ 0.08 rad headroom.
  - It is 0.28 rad RMS from the kit start pose, so reproducing it on the hand needs a kit start-pose change (recommendation only; nothing in the kit was changed).

### Held-out evidence (seeds 6201+6202; `results/p1_heldout.{json,md}`, `figures/p1_heldout.png`)

| Condition | A1 | A2 | A3 | R1 | mean turns30 | mean failure30 | mean P(held 60 s) |
|---|---|---|---|---|---|---|---|
| A: full distribution | 1.148 | 1.123 | 1.329 | 1.158 | 1.189 | 23.6% | 0.740 |
| B: nominal grasp 49 | 1.454 | 1.549 | 1.790 | 1.603 | 1.599 | 0.7% | 0.991 |
| C: robust subset | 1.503 | 1.505 | 1.763 | 1.494 | 1.566 | 1.2% | 0.981 |
| D: grasp 49 + L2 placement noise | 1.414 | 1.524 | 1.761 | 1.594 | 1.573 | 1.0% | 0.987 |

- Differences vs A (mean over the four policies, 95% bootstrap CI):
  - B: turns30 +0.409 [+0.400, +0.418], failure −0.230 [−0.237, −0.223];
  - C: +0.377 / −0.224;
  - D: +0.384 / −0.226.
- **Pre-registered decision:** B and C both pass (≥ +15% turns: +34% and +32%; failure CI excludes 0; all four policies improve on both metrics). D keeps 94% of B's turn gain (rule: ≥ 50%).
- It also holds under the robustness presets (seed 6201, mean of the four policies, A → B): heavy/slippery 1.15 → 1.60 turns and 22% → 0.5% failure; high delay 1.07 → 1.50 and 27% → 1.7%; observation noise ×3 1.19 → 1.60 and 24% → 0.7%.
- The yaw rate while held also rises (0.287 → 0.336 rad/s). So the gain is not only fewer early failures: from a good grasp the policies also rotate faster.

## Phase 2: why R1 rotates slower (`results/reward_analysis.json`)

The rollouts of A1–A3, R1 and the hold policy (dev seeds) were scored by recomputing each reward variant offline. The recomputation matches the logged terms: retention mean equal, rotation term r = 0.97.

- **R1 is slower in the same states.** Matched-state analysis on palm tilt × drift × contact cells covers 100% of held steps. A rotates at 0.323 rad/s in its own states and 0.319 in R1's states; R1 rotates at 0.285 in A's states and 0.284 in its own. So the difference is the policy, not the states it visits.
- R1 spends 43% of held steps below 0.25 rad/s (A3: 21%). The yaw reward is saturated at the ±0.25 clip in 56–78% of steps (A3 78%, R1 56%), and the drift/tilt gate is off in 46–52%.
- **Hazard:** the 2-s failure hazard is lowest at 0.25–0.55 rad/s and highest when stalled (< 0.1 rad/s). Rotating faster is not more dangerous in this range.
- **Which term makes R1's reward prefer R1 over A?** Variant audit of the 20-s return, A mean minus R1 (positive = the reward prefers A's behaviour):

| Reward variant | A − R1 |
|---|---|
| Retention-V1 as trained | −0.050 |
| A's reward (no retention terms) | +0.316 |
| H1: no position part (tilt part + grasp loss kept) | +0.282 |
| H1b: position part × 0.25 | +0.199 |
| H2: tilt margin 0.5 → 0.6 rad | −0.066 |
| H3: grasp loss only | +0.263 |
| H4: yaw clip ±0.5 rad/s | +0.047 |

- **Conclusion:** the retention penalty's *position* part flips the reward's preference from A-like to R1-like behaviour (H1 removes almost all of the flip; H2 does nothing). The yaw clip (H4) independently caps the reward for speed: 61–78% of A's steps are already saturated, so there is no gradient towards faster rotation.
- Selected candidates (at most two, pre-registered before training, each a single change from Retention-V1):
  - **P0 (H1):** retention penalty without its position part (tilt part, margin 0.5 rad, and the −10 grasp loss kept).
  - **C5 (H4):** yaw clip ±0.25 → ±0.5 rad/s, retention terms unchanged.
  - H4 is not combined with the retention ablation.
  - H2 was not trained (no effect in the audit). H3 was not trained (P0 differs from H3 only by the tilt part, which targets the observed failure mode).

## Phase 3: PPO from scratch (one seed per candidate)

- Same as A and R1: 320-D observation, actor/critic architecture, 16-D action, 20 Hz, kit limits, physics, domain randomisation, original 239-grasp distribution, PPO hyperparameters, 4096 environments, 3001 updates, episode length, zero-initialised actor output.
- No initialisation from A or R1 checkpoints, and no grasp optimisation inside training.
- Jobs 310013 (P0) and 310014 (C5), seed 1, about 4.8 s/update. Logs: `logs/train_*`; validation curves: `results/curves/*.json`, `results/convergence.json`, `figures/p4_learning_curves.png`.
- **Stage check at 500** (`logs/stage_check_500.md`): both pass (yaw ≥ 50% of R1's, no exploit, stable).
- **Futility check at 1500:** neither stops (P0 1.270 turns and C5 1.718, vs the rule of 0.937 turns and 26.4% failure).

Validation (seed 7001, 1024 environments, 30 s) at matched updates:

| Update | A1 | A3 | R1 | P0 | C5 |
|---|---|---|---|---|---|
| 500 | 0.68 / 39% | — | 0.65 / 30% | 0.93 / 20% | 0.98 / 26% |
| 1000 | 0.95 / 29% | 1.16 / 26% | 0.95 / 18% | 1.10 / 14% | 1.49 / 21% |
| 1500 | 0.97 / 30% | 1.26 / 26% | 1.09 / 16% | 1.27 / 12% | 1.72 / 20% |

(turns / failure)

- Both candidates learn faster than A and R1 from the start. C5 plateaus at about 1.85 turns from update 2000; P0 at about 1.2–1.27, with its failure falling to about 10%.
- **Selected checkpoints** (max validation turns, pre-registered): P0 at update 1500, C5 at update 2400. Copies and sha256 in `checkpoints/MANIFEST.txt`, together with the final (update 3000) checkpoints and params.

## Phase 4: held-out comparison (`results/phase4.{json,md}`, `figures/p4_settingA.png`)

### Setting A (main): full original grasp distribution, seeds 6201+6202

| Policy | turns30 | failure30 | yaw held | turns60 | P(held 60 s) |
|---|---|---|---|---|---|
| A1 | 1.148 | 24.0% | 0.281 | 2.288 | 0.745 |
| A2 | 1.123 | 27.6% | 0.281 | 2.197 | 0.706 |
| A3 | 1.329 | 24.0% | 0.322 | 2.677 | 0.746 |
| A mean | 1.200 | 25.2% | 0.295 | 2.387 | 0.732 |
| R1 (selected 2600) | 1.158 | 19.0% | 0.263 | 2.299 | 0.763 |
| R1 final | 1.060 | 15.8% | 0.239 | 2.046 | 0.782 |
| **P0** (selected 1500) | 1.273 | **11.5%** | 0.282 | 2.626 | **0.862** |
| P0 final | 1.162 | 10.3% | 0.256 | 2.334 | 0.883 |
| **C5** (selected 2400) | **1.869** | 19.9% | **0.432** | **3.846** | 0.788 |
| C5 final | 1.821 | 20.6% | 0.420 | 3.695 | 0.781 |

Differences (95% bootstrap CI over environments; training-seed variance is *not* included):

| | turns30 | failure30 | turns60 | P(held 60 s) |
|---|---|---|---|---|
| C5 − A mean | +0.669 [+0.642, +0.695] | −0.053 [−0.067, −0.038] | +1.459 | +0.056 |
| C5 − R1 | +0.711 [+0.682, +0.740] | +0.009 [−0.009, +0.027] | +1.547 | +0.025 [+0.005, +0.044] |
| P0 − A mean | +0.073 [+0.053, +0.092] | −0.137 [−0.148, −0.124] | +0.238 | +0.129 |
| P0 − R1 | +0.115 [+0.092, +0.137] | −0.074 [−0.089, −0.059] | +0.326 | +0.099 |

Robustness presets (seed 6201), turns30 / failure:

| Preset | A mean | R1 | P0 | C5 |
|---|---|---|---|---|
| heavy/slippery | 1.159 / 23.8% | 1.141 / 18.3% | 1.247 / 14.4% | 1.863 / 19.1% |
| high delay | 1.079 / 28.6% | 1.048 / 20.5% | 1.159 / 16.1% | 1.697 / 23.2% |
| obs noise ×3 | 1.195 / 25.2% | 1.155 / 19.2% | 1.261 / 11.7% | 1.865 / 19.1% |

Pre-registered thresholds (set from dev/validation data before the held-out runs):

| Threshold | P0 | C5 |
|---|---|---|
| 1. yaw rate ≥ A mean | ✗ (0.282 < 0.295) | ✓ |
| 2. turns30 ≥ 1.05 × A mean and ≥ 1.10 × R1 | ✗ (1.273 < 1.274) | ✓ |
| 3. failure30 ≤ (A mean + R1) / 2 = 22.1% | ✓ | ✓ |
| 4. P(held 60 s) ≥ A mean | ✓ | ✓ |
| 5. every preset turns ≥ 0.9 × A mean | ✓ | ✓ |
| **Promising** | **no** | **yes** |

P0 misses threshold 2 by 0.001 turns and threshold 1 clearly. It is a retention improvement (the lowest failure rate of any policy), not a rotation improvement.

### Setting B: optimised grasp (reported separately; seeds 6201+6202)

| Policy | Grasp 49: turns30 / failure / turns60 | Robust subset: turns30 / failure / turns60 |
|---|---|---|
| A mean | 1.597 / 0.9% / 3.202 | 1.590 / 1.3% / 3.174 |
| R1 | 1.603 / 0.1% / 3.177 | 1.494 / 1.0% / 3.021 |
| P0 | 1.679 / 0.1% / 3.368 | 1.609 / 0.2% / 3.292 |
| C5 | **2.466** / 0.1% / **4.963** | **2.357** / 1.0% / **4.832** |

From a good grasp, every policy almost never fails. The policies are then separated by rotation speed alone, and C5 is 54% faster than the A mean.

### Rotation and retention, assessed explicitly

- **Rotation:** C5 improves it substantially and robustly across settings, presets and both horizons, from both the full distribution and the optimised grasp. P0 does not (its yaw rate is A-level, above R1's).
- **Retention:** P0 improves it (failure 11.5% vs R1 19.0% and A 25.2%). C5 keeps R1's level, which is still better than A's.
- **Neither candidate gets both.** C5 rotates 64% faster than R1 at equal failure; P0 fails 40% less than R1 at slightly higher speed.
- The pose-deviation rule causes most of the remaining failures on the full distribution. With the rule switched off (supplementary, not pre-registered; seed 6201), real drops within 60 s are A 3.8–6.6%, R1 0.6%, P0 1.9%, C5 0.8%, and C5 makes 4.62 turns in 60 s (R1: 2.68).
- **Movement cost of C5** (full distribution, seed 6201, while held): mean joint speed 0.295 rad/s vs 0.24–0.25 for A1/A3/R1/P0 (+18%), and mean |action| 0.875 vs 0.73–0.80. Joints at a limit: 4.4 vs 4.8–5.1, so it presses the kit limits less often. The actions stay within the same kit limits, but faster finger motion is a sim-to-real risk to check before any hardware run.

## Phase 5: outcome and recommendation

**Outcome C (both help)**, with individual and combined effects reported separately:

| Effect (held-out 30-s turns) | Gain |
|---|---|
| Grasp initialisation alone (existing policies, grasp 49 vs full distribution) | +0.41 (+34%), failure 24% → 0.7% |
| Reward balance alone (C5 vs A mean, full distribution) | +0.67 (+56%), failure 25% → 20% |
| Combined (C5 on grasp 49 vs A mean on the full distribution) | +1.27 (+105%), failure 25% → 0.1% |

Recommended next milestone:

1. **Three-seed confirmation of C5** (seeds 2 and 3 plus the existing seed 1). Same protocol, held-out seeds 6201/6202 kept for the final comparison. About 4 h per run on a ws-ia GPU, two at a time. **Not launched**: this needs your approval. P0 is a retention result; confirm it in the same batch only if a lower failure rate at equal speed is a goal.
2. **Standardised start grasp for repeatable hardware evaluation:** grasp 49 (with the robust subset as alternates), keeping the full-distribution results as the main benchmark. This needs a separately approved kit change (a start-pose ramp to grasp 49's joint command); nothing in the kit was changed here.
3. **Then, per Outcome C:** RealSense + ArUco cube tracking to validate on the real hand. Any physical run must be separately approved and supervised, and should check C5's higher joint speed first.

Limitations: one training seed per candidate (CIs cover environment variance only); the dev ranking used ~17 environments per grasp and policy (reliability is nevertheless r = 0.99); the grasp result depends on the cube settling as in simulation; the 0.029 m cube only.

## Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Grasp-initialisation analysis | `results/grasp_table.csv`, `grasp_ranking.json`, `figures/p1_grasp_*.png` |
| 2 | Nominal grasp and robust subset | grasp 49; subset C above; `results/perturbation.json` |
| 3 | Effect of the grasp on A and R1 | `results/p1_heldout.{json,md}`, `figures/p1_heldout.png` |
| 4 | Retention-V1 reward-component analysis | `results/reward_analysis.json`, Phase 2 |
| 5 | Reward-balance candidates | P0, C5 (PROTOCOL.md Section 2) |
| 6 | Checkpoints trained from scratch | `checkpoints/` (MANIFEST.txt) |
| 7 | Training and validation curves | `results/curves/`, `results/convergence.json`, `figures/p4_learning_curves.png` |
| 8 | Held-out 30/60-s comparison | `results/phase4.{json,md}` |
| 9 | Settings A and B separately | Phase 4 |
| 10 | Rotation and retention assessment | Phase 4 |
| 11 | Preservation and regression tests | `logs/preservation_check.txt`, `logs/pytest_full.log` |
| 12 | Recommendation | Phase 5 |

## Reproduce

From `training_source/project`, with the `leap-estimator` environment:

```
# rollouts (any number of workers share a list)
ITEMS=experiments/grasp_rotation_v2/items/p1_dev.json sbatch experiments/grasp_rotation_v2/scripts/sbatch_items.sh
python experiments/grasp_rotation_v2/scripts/analyze_grasps.py
#   then p1_pert.json -> analyze_perturbation.py; heldout_p1_all.json -> compare_p1.py
python experiments/grasp_rotation_v2/scripts/analyze_reward.py
# training
CAND=C5 SEED=1 sbatch experiments/grasp_rotation_v2/scripts/sbatch_train.sh
python experiments/grasp_rotation_v2/scripts/curves.py assess
python experiments/grasp_rotation_v2/scripts/freeze_and_items.py   # -> items/phase4.json
python experiments/grasp_rotation_v2/scripts/compare_v2.py
bash experiments/grasp_rotation_v2/scripts/check_preservation.sh
```

Rollout data (`data/*.npz`) are large and regenerable from the item lists; the summaries are in `results/items/`.
