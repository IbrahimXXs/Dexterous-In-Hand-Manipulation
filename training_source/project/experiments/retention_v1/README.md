# Retention-V1: long-horizon grasp retention during sensorless cube rotation

Milestone 1 of the retention and recovery plan. This is a new proprioception-only PPO policy, trained from scratch under the approved joint limits, with a retention-aware reward.

The baseline is the completed hardware-limited A study (`../hwlimit_ac/`), which is not modified. The pre-registered protocol is in [PROTOCOL.md](PROTOCOL.md).

**Status (2026-11-20 01:30): complete. Outcome B (retention up, rotation not up).** Nothing is committed.

| Phase | State |
|---|---|
| 1. Diagnosis | done (Section 2) |
| 2. Reward design and verification | done (Sections 3–4) |
| 3. Training, seed 1, from scratch | done: 3001 updates, Slurm 309503, 20:11–00:20 (Section 5) |
| 4. Evaluation | done: 30-s held-out test, 60-s retention test, deployment rehearsal (Section 6) |
| 5. Decision | Outcome B; recommendation in Section 7 |

---

## 1. Background

On the real hand, the team observes that the cube rotates several times and then tends to leave the reachable workspace or is lost. On 18 Nov, rotation was only estimated, not measured (`../../../../docs/hardware_session_2026-11-18.md`).

## 2. Phase 1: diagnosis of the hardware-limited A policies

**Data:**
- **Script:** `scripts/record_diagnosis.py`, using `retention_v1/traces.py`.
- **Policies:** the selected checkpoints of A seeds 1–3, in `Mjlab-Leap-HwLim-A`, nominal preset, diagnostic seed 9201 (used nowhere else).
- **Rollouts:** 1024 environments per policy × 60 s (3072 episodes). Only the first episode of each environment is recorded.
- **Timeout:** pushed beyond the horizon, as in the study's 30-s protocol, so a reset is never counted as rotation.
- **Failures:** characterised from their last live step.
- **Contact:** the total normal force on each distal link, from the force study's passive sensor, plus the palm and phalanx force.

**Analysis:** `scripts/analyze_diagnosis.py`, producing `results/diagnosis.json` and `figures/diag_*.png`.

| | A1 | A2 | A3 |
|---|---|---|---|
| Uninterrupted turns, 30 s | 1.15 | 1.10 | 1.31 |
| Uninterrupted turns, 60 s | 2.30 | 2.17 | 2.65 |
| P(retain 30 s) | 0.78 | 0.71 | 0.76 |
| P(retain 60 s) | 0.76 | 0.70 | 0.74 |
| Failures by tilt / drift / fall | 24.1% / 0.1% / 0 | 30.0% / 0.1% / 0 | 25.8% / 0.3% / 0 |

### What simulation shows

1. **Failures are tilt failures, and they come early.**
   - 99.4% of failures exceed the 0.8 rad tilt limit; 0.6% exceed the 8 cm drift limit; none are drops.
   - Failure times, 10/25/50/75/90% quantiles: 2.4 / 4.4 / 6.4 / 11 / 21 s.
   - Hazard per 10-s block: 19.3%, 5.5%, 1.8%, 1.0%, 0.8%, 0.4%.
   - Retention between 30 and 60 s is almost complete: 0.78 → 0.76 for A1.
2. **There is no progressive migration in survivors.**
   - Median drift from the start is 15.6, 16.8 and 17.1 mm at 10, 30 and 60 s. Median tilt is 0.24, 0.24 and 0.25 rad.
   - The cube's palm-frame position oscillates about ±15 mm once per revolution around a fixed centre (`figures/diag_example_trajectories.png`).
   - Hypothesis A (gradual migration out of reach) is **not supported** in simulation.
3. **Where the cube sits in the palm predicts failure, one-sidedly** (`figures/diag_workspace.png`).
   - **Survivor centre:** (41.7, 2.2, −54.3) mm from the palm centre, s.d. (9.9, 9.9, 3.2) mm.
   - **Hazard of failing within 2 s, relative to the base rate:**

     | Region | Relative hazard |
     |---|---|
     | x beyond +2–3 s.d. | 14.5–16.7× |
     | y beyond +2 s.d. | 5× |
     | y beyond +3 s.d. | 18.5× |
     | z (toward the palm) beyond −2 s.d. | 6.7× |
     | z beyond −3 s.d. | 18.8× |
     | the opposite directions | 0.5–2.4× |

   - **Leading indicator:** 97% of tilt failures are preceded by a position excursion beyond Mahalanobis distance 3, with a median lead of 6 s.
   - **Persistent, not progressive:** failing episodes sit at about 3.4 for the whole 5 s before failing, against 1.5 for survivors (`figures/diag_failure_aligned.png`).
4. **Tilt grows over the last 2–3 s.**
   - Median tilt, as a share of the 0.8 rad threshold: 46% at 3 s before failure, 67% at 1 s, 83% at 0.25 s.
   - About a third of failures are abrupt: tilt is below half the threshold 1 s before failing.
   - Hazard rises with tilt above 0.5 rad: 1.9× at 0.5–0.6, 6.7× at 0.6–0.7, 25× at 0.7–0.8.
5. **Contact support is lost late, as a consequence rather than a cause** (hypothesis B: a late symptom).
   - Distal contacts stay at the survivor level (1.62–1.64) until about 1 s before failure, then fall to 1.39 at 0.25 s.
   - The total normal force rises before failure (4.7 N at 1 s before, against a survivor mean of 3.7 N), consistent with the hand pressing on a tipping cube.
6. **Joint limits are not the cause** (hypothesis C: **not supported**).
   - The share of joints commanded at a limit is the same before failure (0.25) as in survivors (0.26).
   - Action saturation is lower before failure (0.31) than in survivors (0.38).
7. **The starting grasp matters a lot** (hypothesis D: **strongly supported**).
   - Survival differs strongly between the 235 grasps used, with about 13 trials each: χ² = 2033, df 234, p ≈ 1e-284.
   - The same grasps are good or bad for all three policies: per-grasp 30-s survival correlates 0.81–0.83 between seeds.
   - 21% of grasps survive less than half the time; 55% survive at least 90% of the time.
   - Failing episodes start farther from the useful region: initial palm Mahalanobis 3.7 vs 2.3, and on average 8 mm further in +x and 10 mm further in +y.
   - The number of initial contacts does not differ (1.09 vs 1.09).

**Diagnosis: E, a combination.**
- **Mechanism:** a poor starting grasp (D) leaves the cube in an unfavourable palm region (+x, +y, or sunk toward the palm). Rotation from there tips it past the tilt limit within the first seconds (B is only the final stage).
- **Not supported:** gradual migration (A) and joint limits (C).
- **What helps:** a retention signal tied to palm position and tilt gives earlier feedback than the termination, and it applies from step 0 to the bad starting grasps. That is the behaviour the policy has to learn: move the cube back to the useful region, or avoid rotating it out of it.

### Simulation vs. the real hand (what can and cannot be said)

- **What the logs contain:** joints, commands, currents and estimator B's estimates. There is no camera or ground truth.
- **Through the same estimator B:**
  - **Position:** the real cube's estimated palm position sits at x ≈ −0.5 to +3 cm and changes by 0.9–3.8 cm over a 30-s run. The same estimate in simulation sits at x ≈ +4 cm and changes by a median of 1.5 cm, matching the true sim change of 1.5 cm.
  - **Contact:** estimated contact patterns differ too. On the hand, the thumb is 0.18–0.26 and the index 0.13–0.30; in simulation, the thumb is 0.47 and the index 0.06.
- **Two explanations remain open:**
  1. The real cube rests elsewhere. The kit resets to the default pose with an operator-placed cube, not to the sim grasp cache.
  2. B is out of distribution on the hand. It is already about 1.5× degraded on these gaits in simulation.
- **What this means:** simulation shows early tilt failures from poor grasps and no long-term migration. The real-hand report of the cube "moving across the palm, then falling" is consistent with the identified risky directions, but **the same mechanism on the hand is not demonstrated**. That needs video or tracking.

## 3. Phase 2: Retention-V1 reward

**Code:** `src/in_hand_rotation_mjlab/retention_v1/rewards.py` and `tasks/hand_cube/config/leap_right_retention_v1/`.

**Unchanged from hardware-limited A** (enforced by `tests/test_retention_v1.py`):
- the actor (320-D; terms `joint_pos`, `prev_commanded_joint_pos`; history 10);
- the action (tanh, ±1/24 rad per step) and the approved limits with the kit margin;
- the reset cache, 20-s episodes, physics, domain randomisation and terminations;
- every existing reward term and the curriculum;
- PPO.

**New terms** (simulator state is used for the reward only; nothing reaches the actor):

| Term | Weight | Raw value | Rationale |
|---|---|---|---|
| `retention_penalty` | −0.05 | Sum over the risky directions (+x, +y, −z) of clamp(excess beyond μ + 1.5 s.d., in s.d., 0, 3)², plus clamp((tilt − 0.5 rad)/0.1, 0, 3)² | Zero inside the useful region, so normal rotation and fingertip repositioning are unaffected. Quadratic where the hazard rises. One-sided: the safe directions are never penalised. Tilt is measured exactly as the termination measures it (same anchor). |
| `grasp_loss` | −10 | 1 on the step `cube_pose_deviation` fires, when `cube_fell` did not | Pose failures (99% of failures) had no explicit cost. This uses the same magnitude as the existing `object_fallen` (−10): −0.5 once after dt scaling. It is never paid on a timeout, and never twice with `object_fallen`. |

**Scaling:** rewards are scaled by the control dt (0.05 s), as for every term. One unit of the retention penalty (at 2.5 s.d. or 0.6 rad) costs 0.0025 per step, about ⅙ of the maximum rotation reward (1.25 × 0.25 × 0.05 = 0.0156). At the cap (9 units: 4.5 s.d., or tilt ≥ 0.8 rad) it costs about 1.4× the maximum rotation reward.

**Coefficient choice** (`results/reward_audit.json`, `scripts/reward_audit.py`):
- **Audit setup:** training-time semantics (20-s episodes, automatic resets) with the curriculum frozen at A1's end-of-training weights. It covered the 3 A policies and a hold-still policy, each with 512 environments × 40 s.
- **Rejected weight:** −0.15 (`results/reward_audit_weight0.15.json`) made the retention penalty 64–73% of the rotation reward on the A policies, so it was too dominant.
- **Chosen weight:** −0.05.

| Mean per-step contribution | A1 | A2 | A3 | Hold still (zero action) |
|---|---|---|---|---|
| rotation (`rotate_finite_diff`) | +0.00709 | +0.00697 | +0.00767 | −0.00022 |
| existing penalties | −0.00197 | −0.00207 | −0.00237 | −0.00091 |
| `retention_penalty` (non-zero on 31–34% of steps; 44% for hold) | −0.00150 | −0.00164 | −0.00188 | −0.00191 |
| `grasp_loss` | −0.00031 | −0.00037 | −0.00034 | −0.00001 |
| **total** | **+0.00331** | **+0.00289** | **+0.00309** | **−0.00306** |

The audit shows four things:
- the retention penalty is about 21–25% of the rotation reward;
- rotating policies stay clearly positive;
- holding still is clearly worse than rotating, so it cannot be exploited;
- `grasp_loss` fired exactly once per pose failure (834 of 834) and never on a timeout (0 of 1520 in a 3-s-episode stress test) or a fall.

## 4. Verification and tests

`tests/test_retention_v1.py` (6 tests, all pass):
- **Maths:**
  - the penalty is zero inside the region and on the safe sides, however far;
  - it is quadratic and capped on the risky sides;
  - the tilt margin is applied, and the components add.
- **Configuration:**
  - RetV1-A differs from HwLim-A by exactly the two terms;
  - all other rewards, weights, terminations, events, curriculum, episode length, dt, action class, cache and actor terms are identical;
  - the PPO configuration is identical except the experiment name.
- **Environment (GPU):**
  - the actor is 320-D with the same terms;
  - commands stay inside the approved limits under saturating actions;
  - `grasp_loss` fires exactly on pose failures, at −10, never on timeouts;
  - the retention term uses the termination's own tilt reference and equals the formula on settled steps. Rewards read kinematics one 5-ms substep before the observation, like every existing term.

**Preservation:** `scripts/check_preservation.sh` checks the baseline at commit d4f0cf4. The check before launch passed:
- 1385 tracked files, 669 checkpoints (including all hwlimit runs), estimator B, the 6 kit policy symlinks and 3 datasets are all unchanged;
- no tracked file was modified.

## 5. Phase 3: training

- **Run:** one from-scratch run, seed 1.
- **Launch:** `scripts/sbatch_train.sh`, Slurm 309503 on `ws-l3-011`, started 2026-11-19 20:11. It runs alone on its GPU at 4.9 s per update.
- **Validation:** the 30-s points (every 100 updates) and the 60-s points (every 500 updates) are evaluated as checkpoints appear, by `scripts/curves.py` on the interactive GPU, with the job as a fallback.

**Stage 3A check at 500 updates** (2026-11-19 20:58; validation seed 7001, 30 s; A1 is the same seed under the A reward):

| Update | R1 turns | R1 failure | R1 survival | R1 yaw rate (rad/s) | A1 turns | A1 failure | A1 survival | A1 yaw rate |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.12 | 0.12 | 27.7 s | 0.033 | 0.20 | 0.81 | 12.2 s | 0.154 |
| 200 | 0.28 | 0.20 | 26.6 s | 0.074 | 0.26 | 0.85 | 12.1 s | 0.184 |
| 300 | 0.41 | 0.28 | 25.6 s | 0.103 | 0.32 | 0.92 | 12.7 s | 0.178 |
| 400 | 0.55 | 0.33 | 24.8 s | 0.141 | 0.48 | 0.63 | 19.3 s | 0.164 |
| 500 | 0.65 | 0.30 | 25.5 s | 0.156 | 0.68 | 0.39 | 22.7 s | 0.184 |

- **No collapse:** rotation is learned and rising, and the yaw rate is 85% of A1's (the stop rule is 50%).
- **Different learning path:** R1 retains the cube from the start and learns rotation afterwards, while A1 rotates earlier but fails more.
- **Reward balance:** over updates 450–550, the retention penalty is 24% of the rotation reward (−0.0143 vs +0.0600), as in the audit.
- **Stability:** PPO is stable, with a learning rate of 3.9e-4 against 4.1e-4 for A1. Action noise std is 1.26 against 0.93 for A1, to be monitored.
- **60-s validation at update 500:** 1.10 turns.

Decision: continue the same run to 3000 updates.

## 6. Results

All numbers come from `results/comparison.{json,md}` (`scripts/compare.py`), `results/convergence.json`, `results/deploy_*.json` and `data/test60_*.npz`. A is the hardware-limited A study's selected checkpoints, evaluated with the same code, seeds and environments.

### 6.1 Training and selection

- **Learning curve** (`figures/cmp_learning_curves.png`): R1's validation turns track A1 and A2 from update 700 on and plateau at about 1.1 from update 1500. A3 stays higher (about 1.27). R1's validation failure rate is below all three A seeds at every point (0.15–0.21 after update 600, against 0.23–0.30 for A).
- **Selected checkpoint** (pre-registered rule: max 30-s validation turns, seed 7001): **update 2600** (1.171 turns). Final checkpoint (update 3000): 1.067 turns, 14.6% failure.
- **Convergence:** validation turns are flat over the last 1000 updates (slope −0.017 per 1000 updates, 95% CI [−0.086, +0.053]), and the failure rate changes by −0.02. The assessment says "not converged (still improving)" only because the training reward rose 3.4% over the last 500 updates (criterion C3, threshold 2%).
- **Action noise:** the training noise std rose to 2.4, against 1.6 for A1. Evaluation is deterministic, and the held-out results show no instability.
- **Frozen copies:** `checkpoints/R1_{selected_model_2600,final_model_3000}.pt` plus params; sha256 values are in `checkpoints/MANIFEST.txt`.

### 6.2 30-s held-out test (selected checkpoint, seeds 5001+5002, 2048 environments per policy)

| Preset | Turns: A mean (seeds 1/2/3) | Turns: R1 (Δ, 95% CI) | Failure: A mean | Failure: R1 (relative) | Survival: A → R1 |
|---|---|---|---|---|---|
| nominal | 1.210 (1.163/1.141/1.326) | 1.164 (−0.046 [−0.155, +0.031]) | 24.1% | **17.2%** (−29%) | 24.7 → **27.6 s** |
| heavy_slippery | 1.179 | 1.151 (−0.028 [−0.129, +0.058]) | 22.3% | **17.2%** (−23%) | 25.1 → 27.3 s |
| light_grippy | 1.093 | 1.110 (+0.017 [−0.094, +0.091]) | 32.6% | **23.0%** (−29%) | 23.1 → 26.8 s |
| high_delay | 1.092 | 1.052 (−0.039 [−0.110, +0.019]) | 27.2% | **19.9%** (−27%) | 23.7 → 26.3 s |
| weak_motors | 1.207 | 1.163 (−0.045 [−0.160, +0.036]) | 24.3% | **17.6%** (−28%) | 24.6 → 27.5 s |
| obs_noise_x3 | 1.206 | 1.165 (−0.041 [−0.146, +0.034]) | 24.0% | **16.8%** (−30%) | 24.7 → 27.5 s |

- **Failure and survival:** improved in every preset. The confidence intervals exclude zero, and R1 beats each of the three A seeds individually.
- **Turns:** no significant difference in any preset. R1 equals A1/A2 and is below A3.
- **Rotation speed while holding:** 11% lower than the A mean (0.262 vs 0.295 rad/s, nominal). The turns stay equal only because R1 holds the cube longer.
- **Turns with resets** (counting turns after a failure and reset): 14% lower. A recovers some turns from fresh grasps after failing.
- **Final checkpoint (3000):** failure falls further (14.1% nominal, −44%), but turns drop to 1.07 (−8%, CI excludes 0).
- **Other metrics:** joint-limit use is slightly higher (commands at limit 31% vs 26%). The kit clip is 0, as for A. Drift and tilt means are unchanged.

### 6.3 60-s retention test (seed 5001, 1024 environments, first episode only)

P(cube still held at t), selected checkpoint (`figures/cmp_survival60.png`):

| Preset | Policy | 10 s | 20 s | 30 s | 60 s | Turns in 60 s | Median failure time |
|---|---|---|---|---|---|---|---|
| nominal | A1 / A2 / A3 | .83 / .81 / .80 | .79 / .77 / .77 | .77 / .74 / .76 | .76 / .73 / .75 | 2.32 / 2.24 / 2.64 | 6.5 / 7.2 / 5.8 s |
| nominal | **R1** | **.95** | **.89** | **.83** | **.78** | 2.32 | **19.8 s** |
| high_delay | A1 / A2 / A3 | .80 / .79 / .76 | .76 / .74 / .72 | .76 / .71 / .71 | .75 / .69 / .70 | 2.13 / 2.06 / 2.38 | 5.6 / 6.7 / 4.6 s |
| high_delay | **R1** | **.90** | **.83** | **.78** | .74 | 2.12 | **15.3 s** |
| heavy_slippery | A1 / A2 / A3 | .85 / .84 / .82 | .79 / .80 / .77 | .76 / .79 / .76 | .72 / .76 / .74 | 2.08 / 2.30 / 2.52 | 8.3 / 6.9 / 6.0 s |
| heavy_slippery | **R1** | **.94** | **.88** | **.84** | **.80** | 2.31 | **16.0 s** |

- **Mostly postponed, partly prevented.** R1 removes most early failures: the median failure time is 2.5–3× later. But R1 keeps failing at a low rate after 20 s, while A has almost stopped failing by then. The gain at 60 s is therefore smaller than at 10–20 s: +3.5 points nominal, +5 heavy_slippery, and +0 to +4 high_delay (R1 ends level with A1).
- **Final checkpoint at 60 s:** P(hold) is 0.80, 0.77 and 0.81 for nominal, high_delay and heavy_slippery, above every A seed. Turns are 2.09, 1.98 and 2.26.
- **Workspace:** R1 spends less time outside the 3-sd palm region (4.5% of held steps vs 5.9–9.3%), and fewer episodes ever leave it (68% vs 73–89%). Palm drift at the end is unchanged (median 18 mm).
- **Contacts:** mean distal contacts while held are 1.80, the same as A2/A3 (A1 holds with fewer).
- **Revolutions before failure** (failing episodes): 0.61 vs 0.29–0.37. R1's failures happen later, after more rotation.
- **Initial grasp:** per-grasp survival correlates with A's at r = 0.81, so the same grasps are hard. On the 52 grasps where A mostly fails (A survival 11%), R1 survives 36%. On good grasps, 96% vs 92%.

### 6.4 Pre-registered targets (selected checkpoint, nominal; PROTOCOL.md Section 5)

| Target | Threshold | R1 | Result |
|---|---|---|---|
| T1 30-s turns | ≥ 1.39 (A mean + 15%) | 1.164 | **not met** |
| T2 pose-failure rate | < 19.3% | 17.2% | met |
| T3 survival | > 24.7 s | 27.6 s | met |
| T4 60-s turns and P(hold 60 s) above A mean | 2.401 / 0.745 | 2.322 / 0.783 | **not met** (turns) |
| T5 yaw rate | ≥ 90% of A (0.266) | 0.262 (89%) | **not met** (narrowly) |

**Outcome by the pre-registered rule: B.** Retention improves clearly and robustly; rotation does not improve and the rotation speed is slightly lower.

### 6.5 Deployment readiness (simulation only; no hardware)

- **Kit-format folder:** `deploy/policies/R_seed1/` (checkpoint and params, sha256 in `deploy/MANIFEST.txt`), outside the kit. The kit is unmodified.
- **Offline check** (`scripts/build_deploy.py`, `results/deploy_verification.json`): the kit's NumPy runtime matches the RSL-RL actor to 9.9e-6 on tanh actions (512 probe observations). The interface (tanh, ±1/24 rad, default pose, no estimator) matches training.
- **Closed-loop rehearsal** (`scripts/rehearse_deploy.py`, `results/deploy_rehearsal.json`; 128 environments, 30 s, nominal):

| Kit runtime | R1 turns | Δ vs training actor (1.195) | A1, same test |
|---|---|---|---|
| Reset to measured pose (as on the hand) | 1.157 | −0.04 ± 0.06 | −0.09 ± 0.06 |
| Reset to grasp target (as in training) | 1.241 | +0.05 ± 0.03 | −0.01 ± 0.05 |
| Target reset, training limits patched in memory | 1.197 | +0.00 ± 0.04 | +0.01 ± 0.05 |

- **Kit agreement and steps:** the kit rebuilds the training observation to within 1.2e-7, and the largest command step is 0.0417 rad (abort guard 0.05).
- **Hardware clip:** the bridge clip acts on 99% of steps, by at most one action step, on joints at their limits. A1/A3/C show the same (84–99%). It is the known PIP/MCP limit gap between the kit runtime and training (`../hwlimit_ac/README.md` Section 8.5), and it disappears with the training limits. For R1 it costs nothing measurable. The proposed narrowing-only fix there remains a pending decision; it was not applied.

## 7. Conclusion and recommendation

**Outcome B.** Within the approved limits, the retention reward produces a policy that holds the cube clearly more reliably than the A baseline. This holds in all six physics conditions and against each A seed. The pose-failure rate falls by 23–30% (by 34–44% at the final checkpoint) and early failures largely disappear. The main goal, more uninterrupted turns, is not achieved: R1 rotates 11% more slowly while holding, so its 30-s turns equal the A mean (1.16 vs 1.21, n.s.) instead of exceeding it by 15%.

What this means:
- **Rotation vs. retention.** Retention was not what limited A's turns in 30 s. A's failures are mostly early and grasp-driven (Section 2). R1 avoids many of them but rotates more cautiously, and the two effects cancel.
- **Hardware transfer is unproven.** On the real hand the failure mechanism has not been shown to be the simulated one (Section 2, "Simulation vs. the real hand"). Whether R1's sim retention gain shows up on hardware is open.

Recommendations (none launched):
1. **Do not run seeds 2–3 to chase Outcome A.** One seed already sits in the A range for turns, and the reward change cannot plausibly add 15–20%. More seeds would confirm B, not change it.
2. **If reliability matters for the hardware demos,** two more seeds (about 4.1 h each, in parallel) would make the retention gain seed-robust before R is used as a hardware candidate. I recommend this only if a hardware A-vs-R comparison is planned. This choice is yours.
3. **For more turns,** the evidence points elsewhere than retention shaping: the initial grasp (r = 0.81 between policies) and rotation speed while holding.
4. **Pending kit decision (unchanged):** the narrowing-only PIP-limit fix from the hwlimit study. R1 is not affected by the gap in simulation.

Limitations: one seed; sim-only; the 60-s test uses one seed (5001) and three presets; the selected checkpoint is chosen on validation turns, so the final checkpoint (more retention, fewer turns) is reported as secondary.
