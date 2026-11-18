# A vs C under the approved LEAP Hand joint limits

**Question.** Under hardware-compatible joint limits and a longer PPO budget, does adding estimated cube state and fingertip-contact probabilities (C) to proprioception improve continuous in-hand cube rotation, compared with proprioception-only PPO (A)?

The pre-registered protocol is in [PROTOCOL.md](PROTOCOL.md). It was written before training, and its sha256 is in `logs/protocol.sha256`.

**Status (2026-11-18): complete.** Results are in Section 8.

| Phase | State |
|---|---|
| 1. Hardware constraints | done (Section 2) |
| 2. Estimator B validation | done: B passed the pre-training check and was reused unchanged (Section 3). It degraded on the newly trained gaits (Section 8.4). |
| 3. Study setup | done (Sections 4–5) |
| 4. Training | done: 6 runs × 3001 updates (Slurm 304098 and 304099, 17 Nov 19:15 → 18 Nov 08:11) |
| 5. Evaluation | done: validation curves (180 points), convergence assessment, held-out test (144 rollouts) |
| 6. Deployment readiness | done, simulation only: kit-format folders verified offline and rehearsed in closed loop. One limit gap was found and needs a decision (Section 8.5). |

**Answer.** Under the approved limits, after 3000 updates, C does **not** outperform A.
- Held-out turns at the selected checkpoints, nominal conditions: A 1.21, C 1.17 (C − A = −0.04, 95% CI [−0.15, +0.06]).
- C − A lies between −0.06 and 0.00 under every physics variation, and every CI includes 0.
- 4 of 6 runs reached the pre-registered plateau. The other 2 missed it narrowly, so neither condition is formally converged.

This study is separate from the completed A/B/C/D study (`../scratch_state_estimation/`) and the force-estimation study (`../force_estimation/`). Neither, nor any other existing file, was modified (Section 7).

---

## 1. Background

The 17 Nov hardware session found a dominant sim-to-real mismatch (`../../../../docs/hardware_session_2026-11-17.md`):
- The deployment kit clamps the sideways (rot) joints to ±0.52 rad.
- The policies were trained with ±1.05 rad.

In the closed-loop sim rehearsal this cost C 0.59 turns and A 0.14 turns, so C fell below A.

## 2. Phase 1: hardware constraints

### 2.1 Approved limits and their source

- **Source:** `LEAP_Hand_RL_Deploy_Kit/kit/deploy_direct.py`, `_HW_LIMIT_LO` and `_HW_LIMIT_HI`, documented as "physical right-hand limits from `leap_hand_controller`".
- **Margin:** the kit's default `--limit-margin-rad 0.01`.
- **Enforcement:** the kit clips every motor command to these limits.

They are the only limits enforced on the hand. The professor's course kit applies them, and they were active in every run of the 17 Nov session (no guard trips, current ≤ 390 mA).

The session report states that widening them needs the TA's approval. This study does not widen anything: it trains inside them.

| Joint (sim order) | Kit limit (rad) | Used in training (kit ± 0.01 margin, ∩ model soft limits) | Original training range (model soft limits) |
|---|---|---|---|
| if/mf/rf_mcp | −0.262 … 1.920 | −0.250 … 1.910 | −0.250 … 2.166 |
| if/mf/rf_rot | −0.524 … 0.524 | −0.514 … 0.514 | −0.995 … 0.995 |
| if/mf/rf_pip | −0.262 … 1.920 | −0.252 … 1.825 | −0.446 … 1.825 |
| if/mf/rf_dip | −0.262 … 1.920 | −0.252 … 1.910 | −0.306 … 1.982 |
| th_cmc, th_axl | 0 … 1.571 | 0.010 … 1.561 | −0.288 … 2.033 |
| th_mcp | −0.349 … 1.745 | −0.339 … 1.735 | −0.397 … 2.370 |
| th_ipl | −0.349 … 1.745 | −0.339 … 1.735 | −1.260 … 1.799 |

`tests/test_hwlimit_study.py` re-parses the kit file and checks these values, the joint permutation, the margin, and the 0.05 rad command-step guard.

**Assessment:** the approved limits are established from the available documentation: the kit source plus the session report.

### 2.2 How the limits enter the simulation (`tasks/hand_cube/config/leap_right_hwlimit_study/env_cfgs.py`)

1. **Command limits.** The integrated joint command, which is what the kit sends to the motors, is clamped to the limits in the "used in training" column at every step. This is in place of the model's soft limits.
   - The clamped command is what the simulated motors track.
   - It is also what the `prev_commanded_joint_pos` observation reports, exactly as the kit feeds back the clipped command.
   - The policy therefore learns under the constraint; actions are not merely clamped afterwards.
   - MuJoCo's mechanical joint ranges are unchanged, because a real joint can still be pushed past a command limit by contact.
2. **Reset states.** The original grasp cache is mostly infeasible under these limits:
   - only 10 of its 449 grasp commands, and 8 of its joint states, lie inside them;
   - rot joints exceed them by up to 0.49 rad, and the thumb cmc/axl joints go up to 2.1 rad.

   A new cache (`cache/hwlimit_grasp_cache.npz`, 239 grasps, built by `scripts/build_grasp_cache.py`) was made in three steps:
   - **Projection:** each original grasp's joint state and command are projected into the limits.
   - **Settling:** the projected grasp is settled for 1 s under training domain randomization, in 4096 environments (about 9 randomized instances per grasp).
   - **Acceptance:** a grasp is kept if ≥ 50% of its instances retain the cube for a further 10 s under the original validator's criteria and settle cleanly: ≥ 1 fingertip contact, penetration ≤ 2 mm, joint state ≤ 0.02 rad outside the limits.

   The unchanged `scripts/validate_grasp_cache.py` (1024 environments, 10 s hold) compares the two caches:

   | Measure | Original cache, original env | New cache, HwLim env |
   |---|---|---|
   | Grasps | 449 | 239 |
   | 10-s retention | 95.4% | 98.1% |
   | Initial ≥ 2 fingertip contacts | 58% | 48% |
   | Initial max penetration | 4.9 mm | 1.75 mm |
   | Cube-position spread (s.d., mm) | 13.5 / 17.4 / 5.7 | 13.2 / 15.6 / 4.4 |

   The grasp commands moved by 0.28 rad on average (largest joint per grasp), and by up to 0.48 rad.

### 2.3 Deployment semantics verified

| Item | Training (both conditions) | Kit | Status |
|---|---|---|---|
| Joint order | `if_mcp, if_rot, if_pip, if_dip, …, th_cmc, th_axl, th_mcp, th_ipl` | hardware order = sim order with rot/mcp swapped per finger (`_HW_TO_SERVER`) | matches (tested) |
| Observation | 320-D: 10 × 16 joint positions relative to default, then 10 × 16 absolute previous commands; empirical normalizer inside the actor checkpoint | same, from `q_hw` and the clipped `prev_cmd_hw` | matches (kit = training to ≤ 1e-5, hardware session) |
| Action | tanh(u) → delta in ±1/24 rad per step, integrated, then clamped | `squash: tanh`, then clipped to the kit limits | matches; the kit clip is now a no-op (Section 3, `kit_clip_frac` = 0) |
| Command step | ≤ 0.0417 rad per step | powered run aborts above 0.05 rad | within the guard (tested) |
| Control rate | 20 Hz (dt 0.005 × decimation 10) | 20 Hz loop (measured 50.2 ms) | matches |
| C features | estimator B on the 320-D input, 19 features | `kit/cube_estimator.py`, NumPy port of B, self-checked on load | matches (hardware session) |
| Episode reset | previous command = grasp-cache target | previous command = measured pose | known, small effect (≤ 0.03 turns in the session's rehearsal); unchanged |

### 2.4 Open questions for the TA or supervisor

None of these block training.

1. **Confirm the limits.** Confirm that the kit limits are the intended operating limits for this hand. The study adopts them as approved because they are enforced and were used on 17 Nov. Training inside them never commands anything the kit does not already allow.
2. **Actuator lag.** The hand lags about 100 ms (session report, item 7), while nominal training randomizes the delay over 25–75 ms. Domain randomization was kept identical to the completed study, as specified. Lag robustness is measured by the `high_delay` test preset (70–100 ms).
3. **Resting pose.** With torque off, the PIPs rest at about −25°, outside the −15° limit. The kit's function test already handles this. It does not affect training.

## 3. Phase 2: estimator B under the new limits

`scripts/validate_estimator.py`, results in `results/estimator_validation.json`.

**Method:**
- The six selected A and C policies of the completed study were each rolled out for 30 s in 512 environments, on simulation seed 8101 (used nowhere else).
- Each was rolled out once in the original environment and once in the HwLim environment.
- B was scored closed-loop on the visited states. Ground truth was used only as labels.

**Decision rule, fixed before running:** B is reused if each pooled error is at most 1.25× its value in the original environment, and the mean contact F1 drops by at most 0.05.

| Pooled over 6 policies | Original env | HwLim env | Ratio |
|---|---|---|---|
| Position RMSE | 8.11 mm | 9.58 mm | 1.18 |
| Rotation error (symmetry-aware, mean) | 12.9° | 14.9° | 1.15 |
| Linear velocity RMSE | 0.0201 m/s | 0.0186 m/s | 0.93 |
| Angular velocity RMSE | 0.610 rad/s | 0.555 rad/s | 0.91 |
| RMSE of the 19 actor features | 0.499 | 0.494 | 0.99 |
| Contact F1 (mean over 4 fingertips) | 0.663 | 0.637 | −0.026 |
| Input shift: mean \|z\| under B's normalizer | 0.82 | 0.66 | |
| Input shift: fraction of \|z\| > 4 | 0.55% | 0.10% | |

**Result: B passes.** It is reused unchanged: `estimator_B.pt`, sha256 `6e333d60…`, checked by the launcher and the tests.

**Caveat:** the shift is policy-dependent. For A seeds 2 and 3, rotation error rose from about 7° to 12° and position error from about 5.4 to 8 mm. A, seed 1 and the C policies changed little. The new limits keep the hand in a narrower, more central part of B's input distribution (smaller |z|). The trained HwLim policies will be audited after training (PROTOCOL Section 6).

**The same rollouts reproduce the hardware observation:**
- The original policies under the approved limits command at least one joint at a limit in 95–99% of steps; the real hand showed 87–98%.
- Their returns fall:
  - A: 0.80 → 0.58 turns;
  - C: 0.94 → 0.57 turns.
- In the original environment the kit would have clipped 14–20% of their joint commands. In the HwLim environment it clips 0%.

## 4. Training setup

| | A | C |
|---|---|---|
| Task | `Mjlab-Leap-HwLim-A` | `Mjlab-Leap-HwLim-C` |
| Actor input | 320-D proprioception | 320-D proprioception + 19 features from frozen estimator B |
| Everything else | identical: HwLim environment, rewards, terminations, curriculum, domain randomization, observation noise and delay, PPO (adaptive KL, learning rate 1e-3, 512-512-256 ELU), 4096 environments, zero-initialized actor output | same |
| Budget | 3001 updates (393 M environment steps), checkpoints every 50 | same |
| Seeds | 1, 2, 3 | 1, 2, 3 |

**Changes from the completed study, applied to both conditions:**
- approved command limits;
- the new reset cache;
- a budget of 3001 updates instead of 1001;
- separate log directories (`logs/rsl_rl/hwlim_{A,C}/`).

**Compute:**
- **GPU limits:** the `gpu` partition allows 1 GPU per user, and my interactive job held it. The `ws-ia` partition allows 2 running jobs per user (RTX 5000 Ada, the same GPU model).
- **Layout:** three runs share each GPU, which changes wall-clock time only. A benchmark on the same GPU model measured:
  - 15.1 s per update per run with three runs sharing one GPU;
  - 5.7 s per update for a single run alone;
  - about 12.6 h for 3000 updates with three runs per GPU.
- **Memory:** about 14 GB GPU memory and 2.5 GB host memory per job.
- **Wall-clock limit:** 24 h per job.
- **Jobs:**
  - 304098 on `ws-l5-007`: A1, C1, A2;
  - 304099 on `ws-l5-009`: C2, A3, C3.
- After training, each job evaluates its runs' validation curves on the same GPU.

## 5. Convergence and selection

These rules are defined in PROTOCOL.md Sections 2–4.

**Validation:**
- Every 100 updates, on seed 7001, nominal preset, 1024 environments.
- Primary measure: uninterrupted turns `V`.

**A run is approximately converged only if, over its final 1000 updates, all of these hold:**
- its OLS trend projects less than 5% further gain;
- the last 500 updates improve on the previous 500 by at most max(0.03 turns, 3%);
- training reward changes by at most 2%;
- validation failure rate changes by at most 0.03.

**Selection:** the highest validation `V`, ties going to the later checkpoint. The final checkpoint is kept and tested separately.

## 6. Files

**New, all in new directories:**

| Path (under `training_source/project/`) | Purpose |
|---|---|
| `src/in_hand_rotation_mjlab/hwlimit_study/limits.py` | approved limits (kit copy, sim order, intersection) |
| `src/in_hand_rotation_mjlab/hwlimit_study/evaluation.py` | evaluation protocol, plus feasibility metrics and estimator scoring |
| `src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_right_hwlimit_study/` | tasks `Mjlab-Leap-HwLim-{A,C}`: environment and PPO configs |
| `experiments/hwlimit_ac/PROTOCOL.md` | pre-registered protocol |
| `experiments/hwlimit_ac/cache/` | new grasp cache, its build summary (`.json`) and sha256 |
| `experiments/hwlimit_ac/scripts/` | `build_grasp_cache.py`, `validate_estimator.py`, `train_run.sh`, `sbatch_runs.sh`, `curves.py`, `check_preservation.sh` |
| `experiments/hwlimit_ac/results/` | `estimator_validation.json`, validation curves, convergence, and test results (pending) |
| `experiments/hwlimit_ac/logs/` | build, validation, launch, training and curve logs |
| `experiments/hwlimit_ac/preservation/` | baseline fingerprints (Section 7) |
| `tests/test_hwlimit_study.py` | 8 tests: kit-limit equality and mapping, configs identical except the actor, B and cache sha256, cache inside the limits, original task unchanged, commands inside the limits and equal to the observation under saturating actions, reset commands inside the limits |

Training outputs go to `logs/rsl_rl/hwlim_A/` and `logs/rsl_rl/hwlim_C/`, which are git-ignored.

**Tests:** the 8 new tests pass. The existing suites also pass: 58 tests across `test_scratch_study`, `test_estimation`, `test_scratch_deploy`, `test_force_estimation` and `test_scratch_env`.

## 7. Preservation

Before any new output was written, the following were recorded in `preservation/` at the baseline commit e6fd994:
- sha256 of all 942 tracked files (deploy kit and hardware logs included);
- sha256 of all 303 existing RSL-RL checkpoints;
- estimator B's sha256;
- listings of the three datasets.

`scripts/check_preservation.sh` verifies them. The check passed at launch: everything unchanged, no tracked file modified.

## 8. Results

The full tables, with per-seed values and every metric, are in `results/tables.md`. The machine-readable version is `results/test_summary.json`.

### 8.1 Learning curves and convergence

Source: `figures/learning_curves.png` and `results/convergence.json`. Validation is on seed 7001, nominal preset, 1024 environments, every 100 updates.

| Run | Pre-registered status | Failing criterion | Selected | V at selected | V at 3000 | Mean V over 100–3000 | First V ≥ 0.5 |
|---|---|---|---|---|---|---|---|
| A1 | approximately converged | none | 2600 | 1.16 | 1.13 | 0.91 | 500 |
| A2 | not converged (still improving) | C1: gain 0.060 vs 0.054/1000; C2: 0.034 vs 0.032; C3: +3.1% vs 2% | 2900 | 1.13 | 1.08 | 0.92 | 400 |
| A3 | approximately converged | none | 2600 | 1.33 | 1.27 | 1.09 | 400 |
| C1 | not converged (still improving) | C2: 0.033 vs 0.032 | 2600 | 1.14 | 1.12 | 0.91 | 500 |
| C2 | approximately converged | none | 2500 | 1.23 | 1.22 | 0.98 | 400 |
| C3 | approximately converged | none | 2000 | 1.14 | 1.09 | 0.96 | 400 |

**Reading:**
- Neither condition is formally converged, since each has one seed that narrowly fails a criterion.
- Every run's projected gain over its last 1000 updates is at most 0.06 turns, and checkpoint-to-checkpoint noise is about 0.03 (s.d.). All runs are therefore at or very near a plateau, and 3000 updates was enough budget for this comparison.
- **Learning speed is the same for A and C:**
  - mean V over updates 100–3000: A 0.98, C 0.95;
  - first reached 0.5 turns at updates 400–500 for both.

  This differs from the completed study, where C learned clearly faster than A under the wider limits.

### 8.2 Held-out test

Source: `figures/test_comparison.png`. Test seeds 5001 and 5002, 2 × 1024 environments per policy, 30 s, all under the approved limits.

**Selected checkpoints, uninterrupted turns** (primary metric; per-seed values are for seeds 1 / 2 / 3):

| Preset | A, per seed | A mean | C, per seed | C mean | C − A | Bootstrap 95% CI | Welch 95% CI | Permutation p (C > A) |
|---|---|---|---|---|---|---|---|---|
| **nominal** | 1.16 / 1.14 / 1.33 | **1.21** | 1.14 / 1.23 / 1.15 | **1.17** | **−0.04** | [−0.15, +0.06] | [−0.25, +0.17] | 0.75 |
| heavy_slippery | 1.09 / 1.16 / 1.29 | 1.18 | 1.16 / 1.26 / 1.12 | 1.18 | +0.00 | [−0.12, +0.11] | [−0.21, +0.21] | 0.45 |
| light_grippy | 1.04 / 1.03 / 1.21 | 1.09 | 1.06 / 1.04 / 1.00 | 1.03 | −0.06 | [−0.17, +0.02] | [−0.28, +0.16] | 0.80 |
| high_delay (70–100 ms) | 1.06 / 1.05 / 1.17 | 1.09 | 1.05 / 1.11 / 1.04 | 1.07 | −0.03 | [−0.10, +0.05] | [−0.17, +0.12] | 0.75 |
| weak_motors | 1.16 / 1.14 / 1.33 | 1.21 | 1.14 / 1.22 / 1.14 | 1.17 | −0.04 | [−0.16, +0.06] | [−0.26, +0.18] | 0.75 |
| obs_noise_x3 | 1.16 / 1.14 / 1.32 | 1.21 | 1.11 / 1.20 / 1.11 | 1.14 | −0.06 | [−0.17, +0.03] | [−0.27, +0.15] | 0.80 |

**Final checkpoints (update 3000):** the same picture. Nominal: A 1.17, C 1.15, C − A −0.02 [−0.12, +0.08]. Across presets C − A ranges from −0.04 to +0.02, and every CI includes 0.

**Secondary metrics** (selected checkpoints, nominal; C − A with bootstrap CI where computed):

| Metric | A | C | Note |
|---|---|---|---|
| turns with resets | 1.46 | 1.42 | −0.05 [−0.20, +0.09] |
| mean yaw rate (rad/s) | 0.295 | 0.283 | −0.012 [−0.040, +0.012] |
| survival (s of 30) | 24.7 | 24.6 | −0.1 [−0.9, +0.8] |
| failure rate | 0.24 | 0.26 | +0.01 [−0.02, +0.05]; all failures are pose failures (tilt/drift), 0 drops, 0 "too fast" |
| fraction of 1-s windows rotating | 0.86 | 0.86 | |
| drift (mean, mm) / tilt (mean, deg) | 16.0 / 15.9 | 15.8 / 15.9 | |
| joint-steps commanded at an approved limit | 26% | 27% | |
| steps with any joint at a limit | 98.8% | 99.1% | Both policies use the limits fully, like the real hand on 17 Nov (87–98%). |
| commands the kit would clip | 0 | 0 | |
| largest per-step command change | 0.0417 rad | 0.0417 rad | The kit aborts above 0.05. |
| mean action change per step | 0.167 | 0.225 | C's actions are about 35% more variable. |

**Conclusion for the study question:** with the approved limits and an approximately converged budget, the estimated cube state and contact features gave no measurable benefit.
- The point estimate slightly favours A (about −0.04 turns).
- The uncertainty spans roughly −0.15 to +0.06 turns.
- A real C advantage larger than about 0.06 turns (5%) is unlikely at this sample size. A small disadvantage cannot be excluded.
- With n = 3 seeds per condition, nothing here is statistically significant in either direction.

**Context:** in the completed study (wider limits, 1000 updates), nominal test turns were A 0.785 and C 0.927, with C learning faster. The two studies differ in limits, reset states and budget, so the numbers are not directly comparable. What changed qualitatively is that A no longer lags C.

### 8.3 Variability across seeds

Training-seed spread is larger than the condition difference:
- A3 is the best run overall (1.33 turns nominal).
- C2 is the best C run (1.23).
- With A3 left out, A's mean would be 1.15 against C's 1.17.

The comparison therefore rests on seed-level variance, which is why the bootstrap and Welch CIs, not environment-level CIs, are the relevant uncertainty.

### 8.4 Estimator B audit on the new policies (PROTOCOL Section 6)

Closed-loop audit on the nominal test rollouts of the trained policies. This is reported only; B was not changed.

| States visited by | Position RMSE | Rotation error | Linear velocity RMSE | Angular velocity RMSE | Feature RMSE | Contact F1 | Mean \|z\| |
|---|---|---|---|---|---|---|---|
| original policies, original env (Phase-2 reference) | 8.1 mm | 12.9° | 0.0201 | 0.610 | 0.499 | 0.663 | 0.82 |
| original policies, HwLim env (Phase 2) | 9.6 mm | 14.9° | 0.0186 | 0.555 | 0.494 | 0.637 | 0.66 |
| **new HwLim-A policies** | 11.8 mm | 19.2° | 0.0207 | 0.664 | 0.598 | 0.474 | 0.95 |
| **new HwLim-C policies** | 11.0 mm | 19.8° | 0.0203 | 0.641 | 0.588 | 0.531 | 0.98 |

**Finding:** on the gaits learned under the limits, B is substantially degraded.
- Position error is 1.36–1.45× the reference.
- Rotation error is 1.48–1.53×.
- Contact F1 drops by 0.13–0.19.
- Feature RMSE is 1.18×.

By the Phase-2 rule (≤ 1.25×, F1 drop ≤ 0.05) it would now fail. The pre-training check passed because it used the original policies' gaits; the new policies explore states B was not trained on.

So C was effectively trained and tested with a degraded estimator. This may explain part of the missing advantage. It does not show that a better estimator would help.

**Recommendation (not carried out; needs your approval):**
- Retrain an estimator B′ on rollouts of the HwLim A and C policies, with the same architecture, labels and protocol as B.
- Then train a C′ arm with B′ for comparison: 3 seeds, the same budget, and fresh A seeds are not needed.

### 8.5 Deployment readiness (Phase 6, simulation only)

**Deployment folders.** Kit-format folders for the six selected checkpoints are in `deploy/policies/{A,C}_seed{1,2,3}/`, with sha256 values in `deploy/MANIFEST.txt`.
- Each holds the checkpoint and `params/env.yaml` + `agent.yaml`; C folders also hold `cube_estimator.npz`.
- The `.npz` is byte-identical to the kit's existing C file (`bea4ff9c…`).
- The folders live outside the kit, which is unmodified.
- Usage: `./leap.sh shadow <folder>` and then `./leap.sh run <folder>`.

**Offline verification** (`scripts/build_deploy.py`, `results/deploy_verification.json`):
- The kit's NumPy runtime matches the RSL-RL training actors on 512 probe observations:
  - tanh actions to ≤ 3.1e-5;
  - C's 19 features to ≤ 1.1e-5.
- The parsed interface is correct: tanh squash, ±1/24 rad deltas, `clip_actions: null`, the training default pose, and the estimator present for C only.

**Closed-loop rehearsal** (`scripts/rehearse_deploy_hwlim.py`, `results/deploy_rehearsal.json`):
- One kit runtime runs per simulated environment, with the bridge's limit clip and command feedback, in the HwLim environment.
- 128 environments, 30 s, seed 5001.
- The kit rebuilds the training observation to within 6e-8.
- The largest per-step command change is 0.0417 rad.

| Policy | Training actor | Kit, reset to measured pose (as on the hand) | Kit, reset to grasp target (as in training) |
|---|---|---|---|
| A1 | 1.19 | −0.09 ± 0.06 | −0.01 ± 0.05 |
| A2 | 1.17 | −0.09 ± 0.07 | −0.01 ± 0.05 |
| A3 | 1.37 | −0.06 ± 0.08 | −0.02 ± 0.06 |
| C1 | 1.15 | −0.04 ± 0.05 | −0.05 ± 0.05 |
| C2 | 1.29 | **−0.19 ± 0.08** | **−0.18 ± 0.07** |
| C3 | 1.17 | −0.07 ± 0.06 | +0.02 ± 0.04 |

**Open item: PIP upper limit.**
- The kit runtime clamps commands to the model's hard joint ranges, then the bridge clips to the kit limits.
- For the finger PIP joints this allows commands up to 1.885 rad. Training clamped at the model's soft limit, 1.825 rad.
- All other joints match training: the kit limit binds, except an MCP lower-bound difference of 0.002 rad.
- Diagnostic: rerunning with the kit runtime's limits replaced in memory by the training limits removes C2's loss (+0.03 ± 0.07) and leaves A1 unchanged (+0.01 ± 0.05). The gap therefore explains C2's deficit.
- **Proposed fix** (not applied; it changes deployment software): let the kit read an optional per-policy limits file, `params/joint_limits.json`. The runtime would clamp to the intersection of those limits and its current ones. This can only narrow limits, never widen them, and leaves every hardware guard unchanged.
- Until then, the policies can be run as they are; C2 is the one that will be most affected.

Reset to the measured pose (the kit's behaviour) costs 0.04–0.09 turns compared with training-style resets, consistent with the 17 Nov session.

No powered hardware run was made.

### 8.6 Limitations

- n = 3 training seeds per condition. Seed variance dominates, and differences smaller than about 0.1 turns cannot be resolved.
- Simulation only. The rehearsal covers the kit's software path, not real contact, friction, cable effects or the measured ~100 ms lag; lag is covered only by the high_delay preset.
- Domain randomisation was kept identical to the completed study, as specified. The nominal actuator delay (25–75 ms) is shorter than the lag measured on the hand.
- The reset cache was rebuilt by projecting the original grasps. Its states are physically settled and validated, but 239 grasps is fewer than the original 449, and the cache has fewer two-finger initial contacts.
- Estimator B is degraded on the new gaits (Section 8.4). The C arm therefore tests "B as it is", not a well-matched estimator.
- Two of the six runs miss the plateau criteria by small margins.

### 8.7 Deviations from the protocol and the request

1. **Where validation points were computed.** Validation points for updates 100–1200 (17 Nov evening) and 1300–2700 (18 Nov morning) were computed during training on the interactive GPU. The rest were computed by the training jobs after training. The protocol, seed, environments and code are the same throughout. The curve script coordinates processes through per-point claim files; `logs/launch.log` records each host.
2. **Where test rollouts ran.** The held-out test ran on three workers: two `ws-ia` jobs (306152, 306153) and the interactive GPU. Each JSON records its host.
3. **Grasp-cache acceptance.** Acceptance followed the original cache's validated property (retention) rather than the stricter "≥ 2 contacts" criterion, which the original cache itself does not meet (Section 2.2).
4. **Known intermittent test.** One older test, `test_estimation_env.py::test_actor_observation_layout`, failed once in a full-suite run and passed in the other 4 runs (3 file-level runs, 1 full rerun). It checks observation timing in the original estimation environment and does not use the new code. Final suite: 71 passed (`logs/pytest_final.log`).
