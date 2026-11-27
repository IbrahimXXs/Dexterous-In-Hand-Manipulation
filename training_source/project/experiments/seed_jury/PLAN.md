# Seed jury: pre-registered plan

**Status:** written 2026-11-27 before any disagreement value was computed. It is hashed (sha256 + timestamp in `logs/`) and committed before any result exists.

**What is pre-registered:**
- Step 1 analyses already-collected hardware logs (20/24/25 Nov). The data are old, but the analysis below was fixed before computing anything. Small n.
- Step 2 is a new simulation evaluation (no training).
- Step 3 is preparation only. The hand is never powered or moved by this study.

**Hypothesis.** Real-hand drops happen in states that training never showed. In such states the three C5 seeds (same recipe, different training seeds) disagree. A jury of the three seeds could outvote a rogue move.

## 0. Policies

| Name | Kit folder (read only) | Checkpoint | sha256 |
|---|---|---|---|
| C5s1 (kit name `C5`) | `LEAP_Hand_RL_Deploy_Kit/policies/C5` | `model_2400.pt` | `2a932dfe37a5540d…` (= trial CSVs) |
| C5s2 (kit name `C5s2`) | `LEAP_Hand_RL_Deploy_Kit/policies/C5s2` | `model_1500.pt` | `49579bdf1f598d8c…` (never run on the hand) |
| C5s3 (kit name `C5s3`) | `LEAP_Hand_RL_Deploy_Kit/policies/C5s3` | `model_3000.pt` | `c221af89199cdeab…` (= trial CSVs) |

- The same files are `experiments/grasp_rotation_v2/checkpoints/C5_{1,2,3}_selected_model_*.pt`, with identical sha256. The simulation uses those.
- All three are 320 → 512-512-256 → 16 (ELU), and each has its own observation normaliser.
- Observation: `joint_pos` and `prev_commanded_joint_pos`, history 10.
- Action: delta ±1/24 rad, tanh squash, `clip_actions: null`, 20 Hz.

## 1. Step 1: replay the real hand through the three seeds (CPU only)

### 1.1 Replay (the kit's own code)

**Code used.**
- `LEAP_Hand_RL_Deploy_Kit/kit/policy_runtime.py`, `LeapPolicy`, one instance per seed, each with its own normaliser.
- `kit/deploy_direct.py` for the joint-order maps `_hw_to_server` / `_server_to_hw` and the hardware limits `_HW_LIMIT_LO/HI`, with the default `--limit-margin-rad` 0.01. `dynamixel_sdk` is stubbed for the import only; no bus code runs.

**Runs.** Every C5s1 and C5s3 run in `LEAP_Hand_RL_Deploy_Kit/logs/trials.csv` (20 Nov; policy `C5` = C5s1), `trials_20261124.csv` and `trials_20261125.csv`, matched by the `log` column to `physical_*.npz`.
- Intervened runs are excluded: 24 Nov trial 8.
- None of the C5 runs had a stall recovery (`stall_recovery_count` = 0 throughout), so the policy is reset only once per run.

**Observation at step k, as the kit built it.**
- Fed to every seed's `LeapPolicy.step(q, None, applied)`:
  - `q = _hw_to_server(q_hw[k])`;
  - `applied = _hw_to_server(command_hw[k−1])` for k ≥ 1.
- The three seeds therefore see the identical observation (only the executed commands enter it).

**Reset (k = 0).**
- `LeapPolicy.reset(q_reset)` is called, with `applied` at step 0 = `q_reset`.
- `q_reset` (the measured post-ramp pose) is not logged. It is recovered per joint as `_hw_to_server(command_hw[0]) − δ(raw_action_server[0])`, where δ(a) = −1/24 + (tanh(a) + 1)/2 · (2/24) rad.
- Joints whose step-0 command sits within 1e-6 rad of a server or hardware limit use `_hw_to_server(q_hw[0])` instead.

**Command of seed s at step k.** `c_s,k = clip(_server_to_hw(target_s,k), lower, upper)`, where `target_s,k` is `LeapPolicy.step`'s output (already clipped to the server limits) and lower/upper are the kit's hardware limits ∓ the 0.01 rad margin. This is exactly the kit's executed-command path.

**GATE.**
- For the seed that actually ran, max over all steps of all runs of |replayed raw action − `raw_action_server`| must be ≤ **1e-4**.
- The max |c_e,k − `command_hw[k]`| is also reported.
- If the gate fails, the study **stops**. No disagreement analysis is run, and the user is told.

### 1.2 Disagreement

- **Notation:** e = executing seed; m_k = per-joint median over the three seeds of c_s,k.
- **D_k** = ‖c_e,k − m_k‖₂ over the 16 joints (rad).
- **Per finger,** D_f,k = ‖·‖₂ over the 4 joints of digit f (index, middle, ring, thumb; the hardware and server orders group the same 4 joints per digit).
- **Smoothing:** D̄_k = trailing mean of D over steps k−9 … k (0.5 s). At the start of a run, it averages the steps available. The same applies to D̄_f,k.
- **Time:** t_k = `elapsed_s[k] − elapsed_s[0]` (seconds from the policy start).
- **Window score:** the mean of D̄ over the steps with t in [a, b).

### 1.3 Run sets (from the trial CSVs, fixed now)

| Set | Rule | Runs |
|---|---|---|
| **P** (primary) | dropped, operator drop time ≥ 10 s | C5s1: 20 Nov t12 (45 s), 24 Nov t7 (30 s). C5s3: 24 Nov t1 (15 s), t11 (45 s), t13 (20 s), t15 (50 s), t17 (50 s); 25 Nov t5 (30 s). **n = 8** |
| **Early** | dropped, drop time < 10 s | C5s1: 24 Nov t2 (8 s), t3 (8 s), t6 (3 s). C5s3: 24 Nov t5, t9, t12 (7 s), t14, t16, t18 (5 s); 25 Nov t2 (7 s), t3, t4 (5 s). n = 12 |
| **Held** | not dropped (held to the end of the run) | C5s1: 20 Nov t2, t5, t15, t18; 24 Nov t10. C5s3: 24 Nov t4; 25 Nov t1. n = 7 |

### 1.4 Windows

Post-drop data never enter a window: every window ends at least 1 s before the drop, and the smoothing is trailing.
- **Pre-drop:** [t_drop − 3, t_drop − 1].
- **Control:** consecutive 2-s windows [3, 5), [5, 7), … The last one must end ≤ t_drop − 5 s (dropped runs) or ≤ the last logged step (held runs).
- **Sensitivity:** [t_drop − 2, t_drop − 0.5].
- **Early drops:** pre-drop window [max(0.5, t_drop − 3), t_drop − 1]; skipped if it is shorter than 0.5 s.

### 1.5 Tests and descriptives (pre-registered)

**H1 (primary).**
- **Per run in P:** r = (pre-drop window score) / (median over that run's control windows of their scores).
- **Supported if** the median of r over the 8 runs is > 1 **and** its 95% bootstrap CI excludes 1. The CI comes from 10,000 resamples of runs with replacement, percentile interval, rng seed 20261128.
- **Also reported:** each r, the number of runs with r > 1, and per seed.

**H2.**
- **AUC** (Mann–Whitney) of the window score for separating P's pre-drop windows (positives, one per run) from the control windows of the same 8 runs (negatives).
- **CI:** 95%, cluster bootstrap over runs (10,000; each resample redraws runs with all their windows).
- **Supported if** the CI's lower bound is > 0.5.
- **Secondary:** the negatives also include the Held runs' control windows.

**Sensitivity.** H1 and H2 are repeated with the sensitivity window.

**Per finger (descriptive).** H1's r computed from D̄_f, per finger: median and bootstrap CI. No test.

**Early drops (descriptive).** Each early drop's pre-drop window score against two baselines from the **same seed's Held runs**:
- **(a) time-matched:** the same absolute window [max(0.5, t_drop − 3), t_drop − 1] in each Held run. This is the main comparison, because the first seconds after the policy start differ from later control periods. Reported: the score's percentile among those values, and the ratio to their median.
- **(b) pooled control windows:** the Held runs' control windows. Reported: the ratio to their median.

**Largest single-finger outlier move (descriptive).** In each pre-drop window (P and Early), the step and finger with the largest unsmoothed D_f,k.
- **Comparison:** the 99th percentile of unsmoothed D_f,k over all control-window steps of the same seed's P and Held runs, per finger.
- **Reported:** finger, time, value, p99, and whether the value exceeds p99.
- **The same for the executed command step** ‖c_e,k − c_e,k−1‖₂ per finger (the "sudden move").
- The operator's notes for each run are listed next to these values.

**25 Nov camera check.** For 25 Nov trial 5 (P) and trial 4 (Early), the camera drop time replaces the operator's.
- **Camera drop time:** the earliest t such that the camera's cumulative quarter turns (`LEAP_Hand_RL_Deploy_Kit/logs/trials_20261125_trial{4,5}_vision.csv`) stay within ±0.25 quarter turns of their final value from t to the end of the series, i.e. the cube's rotation has stopped for good.
- **Reported:** the run's r, and the H1 median with the camera-based r substituted.

### 1.6 Output

**Files:**
- `results/real_replay.json` and `results/real_replay.md`, holding the gate, H1, H2, sensitivity, per finger, early drops, outlier moves and the camera check;
- `figures/real_replay.png` and `figures/real_replay_dark.png`, in the README style (`scripts/make_readme_figures.py`).

**Figure content.**
- D̄ over time for 4 runs fixed now: 24 Nov t15 (C5s3, drop 50 s), 25 Nov t5 (C5s3, drop 30 s), 20 Nov t12 (C5s1, drop 45 s) and 24 Nov t4 (C5s3, held 60 s). Drops and the pre-drop and control windows are marked.
- A panel of pre-drop vs control window scores per P run, normalised by each run's control median.

## 2. Step 2: the jury in simulation (evaluation only; the user approves the GPU job first)

### 2.1 Variants (each one policy: 320-D observation in, 16-D action out)

- **Executed command in the observation.** The environment builds the observation from its own executed command, so the "previous command" is the jury's executed command, as on the hand.
- **Seed outputs.** Each seed's actor is loaded with `scratch_study.evaluation.load_study_policy`, so its own normaliser is used. Let a_s be its raw output, and δ(a) as in 1.1 (the environment applies tanh and the delta mapping).

| Variant | Output u (raw action given to the environment) |
|---|---|
| **J-median** | per-joint median of a_1, a_2, a_3. Since δ is monotone and the three share the previous command, the executed command is the per-joint median command |
| **J-mean** | u = atanh(clip(mean_s tanh(a_s), −1 + 1e-6, 1 − 1e-6)), so the executed delta is the mean of the three deltas |
| **J-veto** | u = a_3 (C5s3), except joint j takes the median value where \|δ(a_3,j) − δ(a_med,j)\| > τ_j |
| C5s1, C5s2, C5s3 | that seed alone (references) |

**τ for J-veto.**
- τ_j = 95th percentile, per joint, of |δ(a_3,j) − δ(a_med,j)| over all alive environment-steps of the first episode.
- These come from C5s3-alone rollouts at setting K / effort actuator, nominal preset, **calibration seed 9411**, 256 environments × 1,200 steps.
- Written to `results/tau.json` **before** any evaluation item runs. J-veto items read that file.

### 2.2 Conditions

Everything reuses the existing harness unchanged: `c5_hw_v1/scripts/eval_items.py::rollout` and `summarise`, under `c5_hw_v2/scripts/eval_items.py::setup`. Only the policy loader is wrapped (2.1), and the environment step is wrapped to record joint positions and the command target, as `gait_efficiency/scripts/sim_gait.py` does.

| Condition | Setting / actuator / preset |
|---|---|
| **K** | kit-start cache, effort (350 mA-equivalent) actuator, nominal |
| **A** | default starts, default actuator, nominal |
| **heavy_slippery**, **high_delay**, **obs_noise_x3** | setting A, default actuator, that preset (as the stress runs of `c5_hw_v1`) |

- **Seeds:** new evaluation seeds **9401 and 9402**, 256 environments × 1,200 steps (60 s) each.
- **Size:** 6 policies × 5 conditions × 2 seeds = 60 rollouts, plus the calibration rollout.

### 2.3 Metrics (per policy and condition, both seeds pooled: 512 environments)

- **Survival at 30 s and at 60 s:** share of environments whose first episode lasts the horizon (`summarise` "completed", 30 s and 60 s tags).
- **Turns per 30 s:** mean signed turns over the first 600 steps (`summarise` "30s" `turns`).
- **Jam proxy:** `summarise` "60s" `jam_proxy` (index MCP/PIP jam ≥ 1 s, the audit detector).
- **Rhythm and quarter turns per cycle:**
  - computed with `gait_efficiency/scripts/gait.py` `count_cycles`, unchanged, on the recorded joint positions;
  - the held window runs to end_step − 1, with cycles from 3 s;
  - quarter turns = 4 |Σ yaw| / 2π over the whole held time;
  - included: environments held ≥ 20 s;
  - summarised as ratio of sums and median rhythm, as in `gait_efficiency`.
- **Mean |command change| per step:** mean over alive environment-steps (first episode, from the second step) and joints of |target_k − target_k−1| (rad).
- **D in simulation:** D_k = ‖c_exec − median_s c_s‖₂ with c = previous command + δ(·), and D̄ its 10-step trailing mean.
  - Reported per policy and condition: the mean, median and 95th percentile over alive steps with t ≥ 3 s.
  - **Real vs simulation (descriptive):** C5s3 alone at setting K, over steps with t in [3 s, end of the first episode − 5 s], against the real replay's C5s3 control-window steps. Reported: medians, 95th percentiles and the real/simulation ratio. A ratio > 1 means the real states are more "surprising" to the jury.

### 2.4 Pre-registered success, against C5s3 alone

**Candidates:** J-median, J-mean and J-veto only. The singles are references.

**Non-inferiority at K.**
- survival at 60 s ≥ C5s3's − 0.01, **and**
- turns per 30 s ≥ 0.90 × C5s3's.
- Both are point estimates over the 512 environments.

**Primary**, either route:
- **(i) Stress survival.** S = mean over the three stress presets of survival at 60 s, each preset pooled over both seeds. Δ = S(variant) − S(C5s3) must be **≥ +0.03**, and its 95% bootstrap CI must exclude 0. The CI comes from 10,000 resamples of environments, independently within each (policy, preset, seed) cell, rng seed 20261129.
- **(ii) Jam.** At K, the jam proxy is ≤ 0.75 × C5s3's (a reduction of at least 25%), and the 95% bootstrap CI (same scheme) of jam(variant) − jam(C5s3) lies below 0. If C5s3's jam proxy at K is 0, route (ii) cannot pass.

**Selection.** Among candidates that pass non-inferiority and the primary, select the largest Δ; break ties by the larger jam reduction.
- If none passes, **no variant is selected and Step 3 is not done.**
- Every variant and condition is reported either way.

### 2.5 Output

`results/sim.json`, `results/sim.md`, `results/tau.json`, `results/items/*.json` and one figure (light and dark): survival at 60 s and turns per 30 s per policy and condition, with the success thresholds marked.

## 3. Step 3: prepare the hand test, do not run it (only if 2.4 selects a variant)

1. **Jury folder support.** `kit/policy_runtime.py` (this branch only) learns to load a jury folder: `jury.json` lists the three seed folders, the rule and τ.
   - A test checks that every existing kit policy gives bit-identical actions before and after the change.
   - No guard, limit or `hand.env` change.
   - The jury folder lives in this experiment's folder, never in `LEAP_Hand_RL_Deploy_Kit/policies/`.
2. **Latency.** Three forward passes per step on CPU, over the 1,200 steps of a replayed run. Reported: median and 99th percentile. Acceptance: p99 ≤ 10 ms (the control period is 50 ms).
3. **Offline check.** The jury runs without error on every replayed real observation from Step 1, and its output is reported per run.
4. **`docs/hardware_protocol_jury_DRAFT.md`** (DRAFT, needs supervisor approval):
   - a torque-off shadow run first;
   - then jury vs C5s3, ABBA × 5 (10 each), camera on, same placement, guards unchanged;
   - primary metric: time to drop; secondary: camera rev/min while held, and safety stops;
   - predictions hashed before the session.

## 4. Reporting rules

- **Labels:** every result is marked pre-registered or exploratory, with n stated.
- **Nulls and failures** are written plainly.
- **Numbers that reach `docs/report/REPORT.md`** get `NUMBERS.md` rows.
- **Deviations** from this plan are listed in the results with their reason. The plan itself is not edited after hashing.
