# C5-HW-V2: results report (single-seed exploratory experiment)

**Date:** 2026-11-26. **Status:** complete in simulation. **Verdict: FAIL (single-seed exploratory).**

**Scope:** one training run (PPO seed 1). It is a **single-seed exploratory experiment**:
- Nothing here shows reproducibility across training seeds, or generalisation across training randomness.
- Every confidence interval is a paired bootstrap over *evaluation environments* (10,000 resamples). It describes evaluation variation for this one trained model, not variation between independent training runs.

The protocol (`PROTOCOL.md`) and the code were frozen and hashed before training (`logs/frozen.sha256`). Steps A and B were planned in `FIT_PLAN.md`, hashed before they ran (`logs/fit_plan.sha256`).

## 1. Headline

| | Measured (simulation) |
|---|---|
| **Pre-registered verdict** | **FAIL (single-seed exploratory)**. All three criteria fail (§5). |
| 1 Kit-start survival (setting K, 30 s) | 0.955, against 0.954 for the C5 3-seed mean (difference +0.001 [−0.003, +0.004]) and 0.939 for C5-EFFORT (+0.016 [+0.011, +0.021]). Failure 0.045 against a bar of ≤ 0.023. **No gain over C5.** |
| 2 Jam proxy (setting A, 60 s) | 4.39% of environments, against 0.51% for C5-EFFORT (difference +3.88 points [+3.22, +4.54]); C5 3-seed mean 14.6%. |
| 3 Rotation (setting A, 30 s) | 0.956 turns against 1.784 for the C5 mean: **ratio 0.536** [0.523, 0.549], bar 0.85. |
| What happened | Trained only from simulated kit starts, the policy is specialised to them. From kit starts it rotates 1.959 turns per 30 s (0.924 × C5, 1.068 × C5-EFFORT) with C5's failure rate. From every other start it is clearly worse than both C5 and C5-EFFORT: from random grasps its 30-s failure is 0.347 (C5 0.180), from grasp 49 0.034 (C5 0.002). |
| Post-hoc finding (descriptive) | 2.66% of the simulated kit starts are lost by all eight baseline policies, and by C5-HW-V2 too. Criterion 1 asked for ≤ 2.30% failure, below that shared floor (§6.3). Criteria 2 and 3 fail independently, by wide margins. |
| Index jams at 350 mA (grasp 49 screen) | None of 1 s or more (0 of 256 environments; index MCP/PIP saturation 0.16%, C5 seed 1 7.5%). Sustained saturation moved to the thumb: 28.9% of environments hold some joint saturated for ≥ 1 s, most often th_axl (§7). |
| Deploy candidate | Built outside the kit and verified offline: kit = training within 2.7e-5; closed-loop rehearsal within its CI. Labelled **NOT RECOMMENDED** because of the verdict (§8). |

**Assumed, not measured:** that the simulated kit start represents the real one (it is milder, §4); that the jam proxy predicts hardware guard trips; that the 350 mA-equivalent effort cap stands in for the hardware current limit. The actuator response is known **not** to match (§3). **A simulation result is not evidence of hardware safety, and this policy has not been run on the hand.**

## 2. What was run

| Item | Value |
|---|---|
| Goal | One policy trained under the real hand's conditions, aimed at its three failure modes: index jams at 350 mA, a simulated actuator ~3.5× slower than the hand, and placement-caused early drops. |
| Step A (response fit) | **Failed** its pre-registered rule (§3), so the response change was **not** applied. |
| Step B (kit-start resets) | 4,035 simulated kit starts built (§4). |
| Task | `Mjlab-Leap-C5HW2` = `Mjlab-Leap-C5HW-EFFORT` (C5 + 350 mA-equivalent effort range 0.167–0.204 Nm + utilisation penalty w = 0.2201) with exactly one change: every training episode starts from the step-B kit-start cache (`apply_kit_start_resets`). `tests/test_c5_hw_v2.py` checks with a full config-tree diff that only the two reset-cache leaves differ, that the PPO config differs only in its experiment name, and that `RESPONSE = None`. |
| Training | PPO seed 1, 4096 environments, 3001 updates (0..3000), from scratch, zero-initialised actor output, C5's PPO configuration. Slurm job 330366 on ws-l3-021, 18:05–22:01, exit 0. Run directory `logs/rsl_rl/c5_hw_v2/2026-11-26_18-06-43_c5hw2_seed1`. |
| Stage check (update 500) | **PASS** (`results/stage_check_500.json`). Validation yaw rate 0.2525 rad/s against a stop threshold of 0.132; 1.179 turns. Utilisation penalty −0.015 per episode against rotation +0.219. No NaN (the NaN-termination counter is 0 at every update; all 70 logged scalars finite). |
| Selection | Max 30-s validation turns (seed 7001; in the run's own task, i.e. from kit starts under the effort-matched actuator; ties → later): **update 3000** (1.972), which is also the final update. The convergence check (GRV2 criteria C1–C4) reports **"approximately converged"**: all four criteria hold (`results/convergence.json`). Validation rose from 1.179 turns at update 500 to 1.882 at 2000 and 1.972 at 3000; its 30-s failure stayed at 0.036–0.044 from update 2000 on (`results/curves/C5HW2_1.json`). Checkpoint `checkpoints/C5HW2_1_selected_model_3000.pt`, sha256 `ee30cac6…e734ac92` (`checkpoints/MANIFEST.txt`). |
| Numerical events | None. The NaN-termination counter is 0 at every one of the 3001 updates and all 70 logged scalars are finite (`results/training_check_final.json`). No held-out evaluation episode ended on a NaN: 0 of 45,056 environments. All 9,511 early terminations were cube-pose deviations (`results/v2_per_eval_seed.json`). The policy's final action noise (std 4.19) is above the C5 / C5-EFFORT range (3.30–3.87); this is descriptive only. |
| Evaluation | 22 new rollouts of 60 s for C5HW2_1, 2048 environments each: settings A (full 239-grasp distribution), B (grasp 49), C (robust 10-grasp subset) and K (kit-start cache) on seeds 6201 and 6202, plus the presets heavy_slippery, high_delay and obs_noise_x3 on A / 6201, each under the **default** (550 mA) and the **effort-matched** (350 mA-eq.) actuator. Slurm jobs 331805 and 331806 on ws-l2-017 and ws-l2-019. The baselines' K rollouts are new (32 items, job 330367 on ws-l4-004); their A/B/C rollouts are C5-HW-EFFORT-V1's (same code, tasks and seeds). Pairing is checked by cache-id equality (`scripts/posthoc_kit_failures.py`). Held-out seeds were used once. |

## 3. Step A: response fit (failed)

Plan `FIT_PLAN.md` §A; results `results/response_fit.md` and `.json`.

- **Method.** Frozen C5_1 (grasp 49, 256 environments, seed 8401, 30 s) on SPEC §4's 24-point grid: action delay {0–25, 0–50, 25–50, 25–75 ms} × motor damping scale {0.5, 0.75, 1.0} × friction-loss scale {0.5, 1.0}. Each point was scored with the audit's own lag estimator (`audit_hardware_logs.lag_position`) against the 20 Nov C5 hardware logs: median lag 79 ms, tracking summary 0.0287 rad.
- **Tracking band (pre-registered: 0.70–1.30× the hardware).** No point is inside it. The simulated tracking error is 2.81× (best) to 3.82× the hardware's.
- **Lag.** The best-lag point (delay 0–25 ms, damping ×0.5, friction ×0.5) has a median lag of 218 ms against 79 ms; median per-joint lag error 134 ms; 0 of 16 joints within 30 ms. The current training setting (25–75 ms, ×1, ×1) gives 282 ms and 3.82×.
- **Sensitivity.** The same grid at the 350 mA-equivalent effort limit also fails (best 2.83×, 219 ms).
- **Free-air cross-check.** The function-test stage-6 sinusoid without a cube: hardware median lag 95 ms (210 samples); simulation 364 ms at the current setting and 275 ms at the best-lag point. The mismatch is therefore in the simulated actuator, not in contact, and no point of this randomisation grid removes it.
- **Consequence (pre-registered stop rule).** No setting came within 150 ms of the hardware, so the response change was **not** trained on. `RESPONSE = None` in `leap_right_c5_hw_v2/env_cfgs.py`. The "hardware-matched" evaluation actuator of SPEC §6 is therefore not available either. A structural change to the actuator model is out of scope.

## 4. Step B: kit-start resets

Plan `FIT_PLAN.md` §B; cache `cache/kit_start_cache.npz` (sha256 in `logs/frozen.sha256`) and record `cache/kit_start_cache.json`; survival `results/kit_start_survival.md`.

- **Procedure** (`scripts/make_kit_start_cache.py`, task `Mjlab-Leap-C5HW-EFFORT`, seed 8501, 4096 environments): the hand at the kit's home pose (clipped to the approved limits); the cube at grasp 49's position + 10 mm, flat, at grasp 49's yaw (−170.19°), with ±4 mm x/y and ±15° yaw noise; 1 s settle; a 5-s linear ramp to the policy's default pose at the 0.167–0.204 Nm effort range. The stored command is the measured post-ramp pose, as the kit resets it.
- **Kept:** 4,035 of 4,096 (98.5%); the rejects are position filters (cube off the palm in x/y 1.5%, cube low 0.1%).
- **Recorded deviation (before any policy evaluation).** The planned "≥ 2 fingertip contacts" filter was not applied: the reference grasp-49 state itself shows 0–1 fingertip-sensor contacts while held, and the filter rejected 100% of starts. Every kept start has hand–cube contact (median 5 contact pairs, none with zero).
- **Against the hardware.** The simulated post-ramp joint pose differs from the hardware's (median first sample of 35 C5-family run logs) by a median of 0.068 rad and at most 0.154 rad (if_pip). In simulation the fingers stop less flexed: MCP by about 0.09 rad and PIP by 0.09–0.15 rad.
- **Bar to beat** (validation seed 7001, 2048 environments, 30 s, effort-matched actuator):

  | Policy | Survival from kit starts [95% CI] | Survival from grasp 49 | Kit − grasp 49 [95% CI] |
  |---|---|---|---|
  | C5_1 | 0.955 [0.946, 0.963] | 1.000 | −0.045 [−0.054, −0.036] |
  | C5_2 | 0.958 [0.949, 0.966] | 0.999 | −0.041 [−0.050, −0.032] |
  | C5_3 | 0.953 [0.943, 0.961] | 0.999 | −0.046 [−0.055, −0.037] |
  | C5-EFFORT (EFFORT_1) | 0.930 [0.919, 0.941] | 0.987 | −0.057 [−0.069, −0.045] |

  Stop rule B ("nearly every start fails") was not triggered.
- **Limitation, stated before training.** In simulation the kit start costs C5 only 4–5 percentage points of 30-s survival. On the real hand, C5 seed 3 dropped the cube at 5–7 s in 6 of 12 valid trials (24 Nov). The simulated kit start does not reproduce that placement failure, so a simulated gain here is weak evidence about the hand.

## 5. Pre-registered decision

Effort-matched actuator, seeds 6201 + 6202 pooled, nominal preset (`scripts/compare_v2.py decision` → `results/v2_comparison.{json,md}`; the script was frozen and hashed before training).

| Criterion (PROTOCOL.md) | Threshold | Value | Met |
|---|---|---|---|
| 1 Kit-start 30-s failure vs C5 mean and C5-EFFORT | ≤ 0.5 × each baseline's failure, and both survival-difference CIs > 0 | Failure 0.045 vs C5 mean 0.046 and C5-EFFORT 0.061 (bar 0.023). Survival difference vs C5 +0.001 [−0.003, +0.004]; vs C5-EFFORT +0.016 [+0.011, +0.021] | **no** |
| 2 Jam proxy, setting A, 60 s | ≤ C5-EFFORT (point estimate) | 0.0439 vs 0.0051; difference +0.0388 [+0.0322, +0.0454] | **no** |
| 3 Turns per 30 s, setting A | ≥ 0.85 × C5 mean (point estimate) | 0.956 vs 1.784; ratio 0.536 [0.523, 0.549]; every bootstrap resample is below 0.85 | **no** |

**Verdict: FAIL (single-seed exploratory).** Criteria 1, 2 and 3 all failed. Per evaluation seed, the setting-A rotation ratio is 0.539 (6201) and 0.533 (6202), and the kit-start failure is 0.046 and 0.045 (C5 mean 0.045 and 0.047), so the result does not depend on pooling (`results/v2_per_eval_seed.json`).

## 6. Descriptive comparison

Everything in this section is descriptive; none of it is a criterion. Full tables: `results/v2_comparison.md`; per-seed values and CIs: `results/v2_per_eval_seed.json`.

### 6.1 Every start setting (effort-matched actuator, seeds 6201 + 6202 pooled, nominal)

| Setting | C5-HW-V2 turns30 / fail30 / jam60 | C5 3-seed mean | C5-EFFORT | Turns ratio vs C5 mean [95% CI] |
|---|---|---|---|---|
| K, kit starts (its training distribution) | 1.959 / 0.045 / 0.0005 | 2.119 / 0.046 / 0.0478 | 1.833 / 0.061 / 0.0024 | 0.924 [0.921, 0.928] |
| A, full 239-grasp distribution | 0.956 / 0.347 / 0.0439 | 1.784 / 0.180 / 0.146 | 1.625 / 0.178 / 0.0051 | 0.536 (§5) |
| B, grasp 49 | 1.677 / 0.034 / 0.0012 | 2.349 / 0.002 / 0.0245 | 2.154 / 0.009 / 0.0000 | 0.714 [0.709, 0.720] |
| C, robust 10-grasp subset | 1.459 / 0.108 / 0.0100 | 2.241 / 0.012 / 0.0627 | 2.060 / 0.017 / 0.0012 | 0.651 [0.642, 0.660] |

- **On its own start distribution (K)** C5-HW-V2 is faster than C5-EFFORT (ratio 1.068 [1.062, 1.075]) with fewer failures (difference −0.016 [−0.021, −0.011]), and almost free of index jams (0.05% vs C5 4.8%). Against C5 it is slower (0.924×) with the same failure (difference −0.001 [−0.004, +0.003]).
- **From every other start it is worse than both baselines**, in rotation and in failure, and the 30-s failure from random grasps (0.347) is about twice C5's. Even from grasp 49, which C5 and C5-EFFORT almost never drop (0.002 and 0.009), it fails 3.4% of the time.
- **Validation and held-out agree:** the training-time validation from kit starts (seed 7001) gave 1.972 turns at update 3000; the held-out K value is 1.959.

### 6.2 Robustness presets and the default actuator

- **Presets** (setting A, seed 6201, effort-matched): C5-HW-V2 turns / failure are 1.015 / 0.346 (heavy_slippery), 0.875 / 0.350 (high_delay) and 0.946 / 0.342 (obs_noise_x3). In each preset every other policy (A1–A3, R1, C5_1–C5_3, C5-EFFORT) reaches at least 1.047 turns with at most 0.290 failure.
- **Default actuator (550 mA):** the same pattern. Setting A 0.916 turns / 0.373 failure (C5-EFFORT 1.617 / 0.176); K 1.959 / 0.046 (C5-EFFORT 1.833 / 0.066); B 1.753 / 0.025; C 1.465 / 0.112. No policy saturates at 550 mA, so the jam proxy is 0 for all of them.

### 6.3 Post-hoc: the kit-start failures are mostly shared (not pre-registered)

`scripts/posthoc_kit_failures.py` → `results/posthoc_kit_failures.json`, added after the decision. Setting K, effort-matched, both seeds pooled (4,096 environments; every policy gets the same start in each environment).

- **A shared floor.** 109 environments (2.66%) are lost within 30 s by all eight baseline policies (A1–A3, R1, C5_1–C5_3, C5-EFFORT); 3.71% by at least six of them. C5-HW-V2 loses every one of the 109.
- **Its failures are mostly the common ones.** Of C5-HW-V2's 186 failures, 81.7% are on starts that at least six of the eight baselines also lose, and 75.3% on starts that all three C5 seeds lose. On the 3,284 starts that no baseline loses, it fails 0.55% of the time.
- **Consequence for criterion 1.** The bar (failure ≤ 0.5 × min(C5 mean, C5-EFFORT) = 2.30%) is below the 2.66% of starts that every baseline loses. Meeting it would have required saving starts that no tested policy saves. This was not checked before the criterion was frozen. It is a weakness of the criterion's design, found after the decision. It does not change the verdict, and criteria 2 and 3 fail on their own by wide margins.

## 7. 350 mA jam screen (descriptive)

`scripts/seed_screen.py` → `results/seed_screen.json`: c5_hw_v1's seed-screen procedure, unchanged. Frozen checkpoints, grasp 49, seed 8301, 256 environments, 30 s, effort limit capped at 0.167–0.204 Nm. The C5 rows are read from `c5_hw_v1/results/c5_seed_screen.json` (same procedure and seed).

| Policy | Turns30 | Survived | Index MCP/PIP saturated (share of samples) | Envs with index jam ≥ 1 s | Envs with any joint saturated ≥ 1 s (joint most often) |
|---|---|---|---|---|---|
| C5 seed 1 | 2.446 | 1.000 | 7.5% | 2.7% | 3.5% (if_mcp) |
| C5 seed 2 | 2.363 | 1.000 | 1.7% | 0 | 42.2% (mf_mcp) |
| C5 seed 3 | 2.226 | 1.000 | 3.6% | 0.4% | 5.1% (rf_mcp) |
| C5-EFFORT | 2.155 | 0.992 | 0.32% | 0 | 1.2% (rf_mcp) |
| **C5-HW-V2** | 1.634 | 0.965 | **0.16%** | **0** | **28.9% (th_axl)** |

- **The index jam is gone**, as with C5-EFFORT. Index MCP/PIP saturation is the lowest of all five policies.
- **The thumb saturates instead.** th_axl is saturated in 5.5% of samples (C5 seed 1 3.0%, seed 3 6.7%), and 9.0% of environments hold a saturation (any joint) for ≥ 2 s. The real guard stops of C5 were index jams; whether sustained thumb saturation trips the guard on the hand is unknown. For reference, C5 seed 3 had no safety stop in 13 attempts on 24 Nov (`docs/hardware_session_2026-11-24.md`) with 6.7% th_axl saturation in this screen.
- Without the cap, the same screen gives 1.740 turns and 0.980 survival.

## 8. Deployment candidate (built, labelled NOT RECOMMENDED)

As pre-registered, the candidate was built whatever the verdict and labelled with it (`deploy/MANIFEST.txt`: "NOT RECOMMENDED (verdict: FAIL (single-seed exploratory))"). All of this ran offline and in simulation. Nothing was copied into `LEAP_Hand_RL_Deploy_Kit/policies/`, no kit file changed (the kit's `git status` is clean), and no hardware was used.

| Item | Value |
|---|---|
| Package | `deploy/policies/C5HW2_seed1/` (`model_3000.pt` and `params/`), kit format. |
| Checkpoint sha256 | `ee30cac61798e97cf8a06aba47e2b93b54d02345a3cf2c1d38bf0e00e734ac92`, identical to `checkpoints/C5HW2_1_selected_model_3000.pt` |
| params/env.yaml, agent.yaml sha256 | `ff91f5f16f479d04ec760191d73a51d7c672f6a0d85b3d29fe927d69e84d478b`, `ed3c0efdfe5635b16141fc2279d95601d65ce5baf1c4b0cff3924e16d93e4dc2` |
| NumPy export | `deploy/exports/C5HW2_seed1_actor.npz` (kit `export_npz`), sha256 `bdebf95197ad79b997f7a064a31f580b0ba392527d91b12fba0f0cdb83c4168f` |
| Actor compatibility | Kit `LeapPolicy` loads it: 320 → [512, 512, 256] → 16 (elu), normaliser, history 10, delta ±1/24 rad, tanh, 20 Hz, no cube estimator. The default pose equals training's. `describe()` differs from C5_seed1 only in checkpoint, config path and iteration. |
| Numerical equivalence | Kit runtime vs RSL-RL training actor: max \|Δ raw\| 2.7e-5, max \|Δ tanh\| 1.5e-5, within the 3e-5 bar (`results/deploy_verification.json`). The export's arrays and normaliser are bit-exact to the checkpoint; export forward vs kit runtime max \|Δ\| = 0 over 4,096 observations one at a time, 8.4e-5 batched (float32 summation order; C5-EFFORT 4.6e-5) (`results/export_verification.json`). |
| Kit offline inspection | `kit/inspect_policy.py`, 600 open-loop steps: finite, within limits, "OK"; action saturation 73% (C5-EFFORT 71%) (`logs/kit_inspect_candidate.log`). |
| Known kit/training gap (unchanged) | The kit's MCP/PIP limits differ from training by up to 0.0417 rad on if/mf/rf MCP and PIP, as for every earlier candidate (the kit PIP-limit fix still awaits approval). |

**Closed-loop rehearsal** (`scripts/rehearse_deploy.py` → `results/deploy_rehearsal.json`). Simulation only, in the candidate's own task `Mjlab-Leap-C5HW2` (kit starts, effort-matched), with the kit runtime and bridge logic in the loop. Seed 5001, 128 environments, 30 s, job 331986 on ws-l1-005.

| Mode | Turns (kit) | Training-actor reference | Δ ± 95% CI | Failure | Bridge clip, share of steps | Max command step |
|---|---|---|---|---|---|---|
| Kit reset "measured" (as on the hand) | 1.894 | 1.884 | +0.010 ± 0.049 | 0.0703 | 0.995 | 0.0417 rad |
| Kit reset "target" (as in training) | 1.915 | 1.884 | +0.030 ± 0.048 | 0.0547 | 0.995 | 0.0417 rad |
| Diagnostic: target + training limits in the kit runtime (in memory only) | 1.929 | 1.884 | +0.045 ± 0.051 | 0.0625 | 0.000 | 0.0417 rad |

- All three modes are within their CI of the training actor. The kit's observation error after 10 steps is 0 in every mode.
- The bridge clips in about 99.5% of steps. This is the known MCP/PIP limit gap, and it falls to 0 with training limits; C5-EFFORT showed 0.982 / 0.984 / 0.000.
- Command steps never exceed 1/24 rad, so the kit's 0.05 rad step guard cannot trip.
- One run at 128 environments; GPU simulation is not bit-reproducible, so the small positive differences are within run-to-run variation.


## 9. Limitations: what is assumed, not measured

1. **One training seed.** Training-seed variance is unknown. Among the C5 seeds, 30-s turns span 1.68–1.85 on the full distribution.
2. **The actuator response is not matched** (§3). The simulated actuator is about 3.5× slower in tracking than the hand, and no randomisation setting in the pre-registered grid fixes it. Every result here is under that mismatch.
3. **The simulated kit start is milder than the real one** (§4). It tests the start procedure (home pose, ramp, measured-pose command), not the operator's placement error. Its post-ramp pose differs from the hand's by up to 0.154 rad.
4. **The jam proxy is not validated against hardware** (C5-HW-EFFORT-V1 report §4.2, finding 4).
5. **The 350 mA → 0.1855 Nm mapping** uses Kt = 0.53 Nm/A, which was not measured on this hand.
6. **The loss off the kit-start distribution is attributed to the one change, in one run.** The protocol changed only the start distribution, so in this run the drop on settings A, B and C follows from training only from kit starts. Whether mixing kit starts with the grasp cache would avoid it was not tested, and with one seed the size of the drop is not known to reproduce.
7. **Criterion 1 was set below a shared failure floor** (§6.3, post-hoc).
8. **Thumb saturation is not validated against hardware** (§7). In the jam screen, 28.9% of environments hold some joint saturated for ≥ 1 s, mostly th_axl: more than C5 seeds 1 and 3 and C5-EFFORT (1.2–5.1%), less than C5 seed 2 (42.2%, mostly mf_mcp).

## 10. Deviations and additions (none changes a threshold or a decision rule)

- Step B's fingertip-contact filter was not applied (§4); recorded in `cache/kit_start_cache.json` before any policy was evaluated on the cache.
- The response change and the "hardware-matched" actuator were dropped by the pre-registered step-A rule, before training.
- The SPEC's seed-level statistics are impossible with one seed; the paired environment bootstrap was declared before training (`PROTOCOL.md`).
- Added after the decision, descriptive only: `scripts/posthoc_kit_failures.py` (§6.3), `scripts/per_eval_seed.py` (per-seed values, CIs for settings B/C/K and the termination reasons), `results/training_check_final.json` (whole-run numerical check) and the kit offline inspection. None changes a threshold or the verdict.
- Evaluation ran on several Slurm GPU hosts (ws-l1-002, ws-l1-005, ws-l2-017, ws-l2-019, ws-l4-004) with identical code; pairing was verified by cache-id equality.

## 11. Tests and preservation (final runs, 2026-11-26)

| Check | Result |
|---|---|
| `tests/test_c5_hw_v2.py`: the config-tree diff shows only the two reset-cache leaves, the PPO config differs only in its experiment name, `RESPONSE = None` | **5 passed** (`logs/pytest_test_c5_hw_v2_final.log`) |
| Full training suite `tests/` | **93 passed** (`logs/pytest_training_suite_final.log`). An earlier run during training gave 92 passed, 1 failed (`logs/pytest_training_suite.log`): `test_estimation_env.py::test_actor_observation_layout`, line 56, the obs-delay assertion that also failed intermittently on 23 and 24 Nov. The test file is unchanged since 726a4be; it was not modified, skipped or fixed. On the merged `task7-consistency` branch the suite again gave 92 passed, 1 failed (this test); the same test also fails on unmodified `main`, and every existing task's configuration is identical with and without the C5-HW-V2 package (`logs/test_estimation_env_intermittent.md`). |
| Repository root `tests/` | **76 passed** (`logs/pytest_root_final.log`) |
| Frozen files (`logs/frozen.sha256`: protocol, plan, configs, cache, step-A/B results, decision scripts) | all match (`sha256sum -c`) |
| `scripts/check_preservation.sh` | **All OK** (`logs/check_preservation_final.log`). No tracked file outside this milestone differs from 8bdcbee, and no stray untracked files. 1,035 pre-existing RSL-RL checkpoints are unchanged; the hwlimit_ac (12), retention_v1 (2), grasp_rotation_v2 (8) and c5_hw_v1 (4) manifests match; 217 kit logs, audit outputs and kit policies are unchanged; `hand.env` and the grasp cache are unchanged. |
| Deploy kit | Nothing copied into `LEAP_Hand_RL_Deploy_Kit/policies/`; the kit's `git status` is clean. No hardware was used. |

## 12. Files

- Plan and protocol: `FIT_PLAN.md`, `PROTOCOL.md`; hashes in `logs/fit_plan.sha256` and `logs/frozen.sha256`.
- Configuration: `src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_right_c5_hw_v2/` (`env_cfgs.py`, `__init__.py`).
- Tests: `tests/test_c5_hw_v2.py`.
- Scripts (`scripts/`): `response_fit.py`, `make_kit_start_cache.py`, `eval_items.py`, `compare_v2.py`, `curves.py`, `sbatch_train.sh`, `sbatch_job.sh`, `freeze_checkpoints.py`, `seed_screen.py`, `build_deploy.py`, `export_candidate.py`, `rehearse_deploy.py`, `check_preservation.sh`; added after the decision (descriptive): `per_eval_seed.py`, `posthoc_kit_failures.py`.
- Cache: `cache/kit_start_cache.npz` and `.json`.
- Results: `results/`: `response_fit.*`, `kit_start_survival.*`, `stage_check_500.json`, `training_check_final.json`, `curves/`, `convergence.json`, `items/`, `v2_comparison.{json,md}`, `v2_per_eval_seed.json`, `posthoc_kit_failures.json`, `seed_screen.json`, `deploy_verification.json`, `export_verification.json`, `deploy_rehearsal.json`.
- Checkpoints and deploy candidate: `checkpoints/` and `deploy/`. The `.pt` and `.npz` binaries are git-ignored (no large binaries in git); their sha256 are in `checkpoints/MANIFEST.txt` and `deploy/MANIFEST.txt`.
- Raw per-environment arrays: `data/` (git-ignored): 54 held-out and 16 step-B rollouts, response-fit and jam-screen recordings.
