# C5-HW-EFFORT-V1: results report (single-seed exploratory experiment)

**Date:** 2026-11-24. **Status:** complete in simulation; not committed.

**Scope:** one training run (PPO seed 1). It is a **single-seed exploratory experiment**:
- Nothing here shows reproducibility across training seeds, or generalisation across training randomness.
- Every confidence interval is a paired bootstrap over *evaluation environments* (10,000 resamples). It describes evaluation variation for this one trained model, not variation between independent training runs.

The protocol (`PROTOCOL_EFFORT_V1.md`) and the code were frozen and hashed before training (`logs/frozen.sha256`). The thresholds are those of SPEC §7, unchanged.

## 1. Headline

| | Measured (simulation) |
|---|---|
| **Pre-registered verdict** | **SINGLE-SEED PASS (exploratory; not confirmed)**. All three success criteria are met and no rejection criterion fires. |
| Jam proxy (decision setting) | 0.51% of environments vs 14.6% for the C5 3-seed mean: a **96.5% reduction** (CI 95.0–97.9%). Difference −0.141 [−0.149, −0.133]. |
| Rotation (turns per 30 s) | 1.625 vs C5 1.784, a **ratio of 0.911** (CI 0.903–0.919). The bar is 0.90, so the margin is **small**: per evaluation seed, 6201 gives 0.923 and **6202 gives 0.899**. Against A (1.193) the ratio is 1.36. |
| Failure (30 s) | 0.178 vs C5 0.180; difference −0.003 [−0.011, +0.006]. |
| Cost | About 9% slower rotation than C5, in every setting, under both actuators and in every preset (§3). |

**Assumed, not measured:**
- that the simulated jam proxy predicts the hardware guard trips;
- that the 350 mA-equivalent effort cap (0.53 Nm/A × 0.35 A) is the right simulated stand-in for the hardware current limit;
- that the actuator response, which is not fitted, does not change the conclusion.

Section 6 lists why each of these is uncertain. **A simulation pass is not evidence of hardware safety.**

## 2. What was run

| Item | Value |
|---|---|
| Task | `Mjlab-Leap-C5HW-EFFORT`: GRV2 C5 with exactly two changes, verified by the config-tree diff in `tests/test_c5_hw_v1.py`. |
| Change 1 | Effort-limit randomisation 0.167–0.204 Nm (350 mA × 0.53 Nm/A × U(0.9, 1.1)) instead of 0.262–0.321 Nm (550 mA). |
| Change 2 | Penalty −w Σ_j [max(0, u_j − 0.8)/0.2]², with u_j = \|τ_j\|/L_j and **w = 0.2201**, frozen before training. It was calibrated on frozen C5_1 capped rollouts to 10% of the rotation reward (`results/effort_weight_calibration.json`). |
| Training | PPO seed 1, 4096 environments, 3001 updates, from scratch. Slurm job 319440 on ws-l5-017, 00:27–04:27. Run directory `logs/rsl_rl/c5_hw_v1_EFFORT/2026-11-24_00-27-03_c5hw_effort_v1_seed1`. |
| Stage check (update 500) | **PASS** (`results/stage_check_500.json`). Validation yaw rate 0.253 rad/s against a stop threshold of 0.132; turns 1.062. The penalty was −0.015 per episode against rotation +0.123. No NaN. |
| Selection | Max 30-s validation turns (seed 7001): **update 2900** (1.637); the final update 3000 scores 1.609. The GRV2 convergence check reports "not converged (still improving)": the curve was flat at about 1.59–1.64 from update 2100 on, with a last uptick at 2900. The run was not extended; 3001 updates is the protocol. |
| Numerical events | One isolated simulator-NaN episode reset, at update 2685. The env's built-in `nan` termination caught it (3.1% of that update's terminations). Training was unaffected: validation was normal at 2700–2900. None of the three C5 runs logged one. **No** evaluation episode of the selected checkpoint ended on a NaN (0 of 36,864). |
| Evaluation | 144 rollouts of 60 s each (`items/`), for A1–A3, R1, C5_1–C5_3 and EFFORT_1. The settings are A (full 239-grasp distribution), B (grasp 49) and C (robust 10-grasp subset), each on seeds 6201 and 6202 with 2048 environments. The presets heavy_slippery, high_delay and obs_noise_x3 run on setting A with seed 6201. Each rollout runs under the **default** actuator (550 mA) and under the **effort-matched** one (350 mA-eq.). Pairing (the same reset draws for every policy) is asserted in `compare_effort.py`. |

The training-time validation runs in each run's own task, so EFFORT_1 is validated under the cap and C5 without it. The two validation curves are **not** directly comparable. The held-out evaluation puts every policy under both actuators.

## 3. Comparison with A and C5

The full tables, including every metric, setting and preset, are in `results/effort_v1_comparison.md` and `.json`. Per-evaluation-seed values and the rotation-ratio CI are in `results/effort_v1_per_eval_seed.json`. Figures: `figures/effort_v1_summary.png` and `figures/effort_v1_saturation_per_joint.png`.

### 3.1 Decision setting: full distribution, nominal, effort-matched actuator, seeds 6201 and 6202 pooled

| Metric | EFFORT_1 | C5 mean (±SD over 3 training seeds) | A mean | R1 |
|---|---|---|---|---|
| Turns / 30 s | **1.625** | 1.784 (C5_1 1.850, C5_2 1.819, C5_3 1.682) | 1.193 | 1.148 |
| Turns / 60 s | 3.300 | 3.646 | 2.378 | 2.299 |
| Failure, 30 s | **0.178** | 0.180 (0.199 / 0.149 / 0.193) | 0.249 | 0.175 |
| Failure, 60 s | 0.206 | 0.193 | 0.265 | 0.224 |
| Yaw rate while held (rad/s) | 0.368 | 0.406 | 0.290 | 0.257 |
| Turns, if survived 30 s | 1.903 | 2.097 | 1.494 | 1.300 |
| **Jam proxy, 60 s** (envs with ≥ 1 s index jam) | **0.005** | 0.146 (0.179 / 0.100 / 0.158) | 0.139 | 0.134 |
| Jam proxy, 30 s | 0.005 | 0.139 | 0.135 | 0.128 |
| Index blocked fraction | 0.0027 | 0.0375 | 0.0196 | 0.0124 |
| Saturation u ≥ 0.98, all joints | 0.0012 | 0.0129 | 0.0105 | 0.0103 |
| Saturation: if_mcp / if_pip / th_axl | 0.0025 / 0.0007 / 0.0037 | 0.0362 / 0.0025 / 0.0267 | 0.0189 / 0.0017 / 0.0057 | 0.0076 / 0.0058 / 0.0077 |
| Utilisation mean / p95 / p99 | 0.231 / 0.616 / 0.784 | 0.254 / 0.734 / 0.997 | 0.238 / 0.681 / 0.986 | 0.257 / 0.711 / 0.985 |
| Tracking error \|cmd − q\| (rad) | 0.104 | 0.109 | 0.096 | 0.098 |
| Joint speed (rad/s) | 0.326 | 0.316 | 0.255 | 0.245 |
| Command-limit occupancy | 0.225 | 0.260 | 0.260 | 0.321 |

### 3.2 Pre-registered criteria (SPEC §7, unchanged)

| Criterion | Threshold | Value | Met |
|---|---|---|---|
| 1a Jam-proxy reduction vs C5 | ≥ 50% | 96.5% (CI 95.0–97.9%) | yes |
| 1b 95% CI of the difference excludes 0 | — | −0.141 [−0.149, −0.133] | yes |
| 2a Turns30 vs C5 | ≥ 0.90 × C5 | 0.911 (CI 0.903–0.919; 0.35% of bootstrap resamples fall below 0.90) | yes, narrowly |
| 2b Turns30 vs A | ≥ 1.20 × A | 1.362 (CI 1.350–1.375) | yes |
| 3 Failure30 | ≤ C5 + 0.03 | 0.178 vs 0.210 | yes |
| Reject: turns < 0.80 × C5 / jam reduction < 25% / failure > C5 + 0.08 | — | — | none fires |

### 3.3 Per evaluation seed (effort-matched, full distribution, nominal)

| Eval seed | EFFORT_1 turns30 / fail30 / jam60 | C5 mean | A mean | Turns ratio vs C5 | Turns ratio vs A | Jam reduction |
|---|---|---|---|---|---|---|
| 6201 | 1.642 / 0.165 / 0.006 | 1.778 / 0.183 / 0.146 | 1.195 / 0.246 / 0.142 | 0.923 | 1.374 | 96.0% |
| 6202 | 1.609 / 0.191 / 0.004 | 1.789 / 0.177 / 0.146 | 1.191 / 0.252 / 0.137 | **0.899** | 1.350 | 97.0% |

On evaluation seed 6202 alone, the rotation criterion would be missed by 0.001. The pre-registered decision pools both seeds.

### 3.4 Against each C5 training seed (descriptive, not a criterion)

| | C5_1 (the policy run on the hand) | C5_2 | C5_3 |
|---|---|---|---|
| Jam reduction | 97.1% | 94.9% | 96.8% |
| Turns30 ratio | **0.878** | **0.894** | 0.966 |
| Failure30 difference | −0.021 | +0.029 | −0.015 |

Against the two faster C5 seeds, the rotation ratio is below 0.90.

### 3.5 Secondary settings, robustness presets and the default actuator (descriptive)

| Setting | EFFORT_1 turns30 | C5 mean | Ratio | Fail30 EFFORT_1 / C5 | Jam60 EFFORT_1 / C5 |
|---|---|---|---|---|---|
| B grasp 49, effort | 2.154 | 2.349 | 0.917 | 0.009 / 0.002 | 0.000 / 0.024 |
| C robust subset, effort | 2.060 | 2.241 | 0.919 | 0.017 / 0.012 | 0.001 / 0.063 |
| A heavy_slippery, effort | 1.503 | 1.728 | **0.870** | 0.170 / 0.179 | 0.005 / 0.105 |
| A high_delay, effort | 1.497 | 1.672 | **0.895** | 0.202 / 0.202 | 0.005 / 0.144 |
| A obs_noise_x3, effort | 1.636 | 1.774 | 0.922 | 0.171 / 0.181 | 0.007 / 0.143 |
| A nominal, **default actuator** | 1.617 | 1.794 | 0.901 | 0.176 / 0.183 | 0 / 0 |
| B / C nominal, default actuator | 2.164 / 2.061 | 2.361 / 2.268 | 0.916 / 0.909 | 0.006 / 0.003; 0.015 / 0.011 | 0 / 0 |

- **Robustness.** In every preset, EFFORT_1 is at or above 0.90 × the A mean of that preset (1.29–1.37 ×).
- **Default actuator (550 mA).** EFFORT_1 is not hurt when the cap is lifted: 1.617 vs 1.625 turns.
- **No policy saturates at 550 mA** (u ≥ 0.98 in ≤ 0.01% of samples). The jam proxy is therefore 0 for every policy, and the default actuator cannot discriminate between them on effort.

## 4. Index-jam analysis

**Definition (`PROTOCOL_EFFORT_V1.md`).**
- A sample is *blocked* when index MCP or PIP is at u ≥ 0.98 and |cmd − q| ≥ 0.2 rad.
- A *jam episode* comes from the audit's own detector (`audit_hardware_logs.jam_episodes`).
- An environment *jams* if it has an episode of 1 s or more.

### 4.1 Simulation results (full distribution, effort-matched, 4096 environments per policy)

| Policy | Jam rate if held 60 s | Jam rate if failed | P(fail \| jam) | P(fail \| no jam) | Onset < 3 s | 3–10 s | ≥ 10 s |
|---|---|---|---|---|---|---|---|
| **EFFORT_1** (21 jam envs) | 0.004 | 0.009 | 0.38 | 0.21 | 0.52 | 0.29 | 0.19 |
| C5_1 | 0.164 | 0.235 | 0.28 | 0.20 | 0.47 | 0.28 | 0.24 |
| C5_2 | 0.086 | 0.176 | 0.28 | 0.14 | 0.87 | 0.10 | 0.02 |
| C5_3 | 0.129 | 0.270 | 0.36 | 0.18 | 0.67 | 0.27 | 0.07 |
| A1 / A2 / A3 | 0.142 / 0.089 / 0.118 | 0.19 / 0.17 / 0.26 | 0.31 / 0.43 / 0.44 | 0.24 / 0.27 / 0.23 | 0.48–0.77 | 0.22–0.30 | 0.01–0.23 |
| R1 | 0.110 | 0.217 | 0.36 | 0.20 | 0.71 | 0.13 | 0.16 |

### 4.2 Findings

1. **The reduction is real effort avoidance, not a threshold artefact.** EFFORT_1 does not park just under the u = 0.98 line. Its share of joint-samples drops in every high-utilisation band:

   | Band | EFFORT_1 | C5 seeds | A3 |
   |---|---|---|---|
   | u ≥ 0.8 | 0.83% | 3.0–4.1% | 2.8% |
   | u 0.90–0.98 | 0.14% | 0.73–0.99% | 0.71% |
   | if_mcp, u 0.90–0.98 | 0.35% | 1.2–2.3% | — |
   | if_mcp, u ≥ 0.98 | 0.25% | 1.8–6.3% | — |

   Per-joint p99 utilisation is ≤ 0.88 on every joint; for C5_1 it is 1.00 on if_mcp, mf_mcp, rf_mcp, th_axl and th_mcp.
2. **The cost is speed, not retention.** Failure is unchanged. The yaw rate while held is 9% lower, and so are turns if survived. Joint speed is not lower (0.326 vs 0.316 rad/s). Command-limit occupancy falls from 0.260 to 0.225. The policy moves as fast, but in postures that need less force near the limits.
3. **Torque demand in physical units.** These figures are measured in simulation and use the configured 0.53 Nm/A mapping. They are joint-pooled 99th percentiles under the *uncapped* actuator:

   | Policy | p99 demand |
   |---|---|
   | C5 mean | ≈ 350 mA-eq. (u 0.633 of 0.2915 Nm) |
   | A mean | ≈ 340 mA-eq. |
   | EFFORT_1 | ≈ 275 mA-eq. (u 0.498) |

   EFFORT_1's peak demand leaves about 20% headroom below the 350 mA limit. C5 and A sit at it.
4. **The jam proxy does not separate A from C5 in simulation** (A mean 13.9%, C5 mean 14.6%, R1 13.4%). On the hand on 20 Nov, only C5 showed sustained index jams; the A3/R1/P0 stops were start transients (SPEC E1, E9). The simulated proxy therefore captures *saturation-plus-error episodes*, but its link to hardware guard trips is **not validated**. It may over-count for A and R1, or under-count for C5, whose hardware jams persist in a way the simulation does not reproduce (SPEC E5).
5. **When jams occur.** In every policy, about half or more of jams start within 3 s, which matches the hardware's early index loading (SPEC E2). Jammed environments fail more often (P(fail | jam) 0.28–0.44 vs 0.14–0.27 without). For EFFORT_1 the conditional rates rest on 21 environments and are imprecise.

## 5. Deployment candidate (built only because the gate passed)

All of this was produced offline. No hardware was used, and no kit file changed (the kit's `git status` is clean).

| Item | Value |
|---|---|
| Package | `deploy/policies/C5HW_EFFORT_V1_seed1/` (`model_2900.pt` and `params/`), in kit format. The original C5 folders were not touched. |
| Checkpoint sha256 | `d1d46f0558eb958d78bfb3e84e9a0575659d7550df1feb3fe7aeb4c3c41a4469`, identical to `checkpoints/EFFORT_1_selected_model_2900.pt` |
| params/env.yaml sha256 | `14cd28d04d323eeda3f74fda2b6935e55c1ac2de666b6f5084644997a1627b41` |
| params/agent.yaml sha256 | `3f1fe0d598efdd7635625f17fc06fef1291a51c5ee937fd66dd9fb1a55d247f3` |
| NumPy export | `deploy/exports/C5HW_EFFORT_V1_seed1_actor.npz` (kit `export_npz`), sha256 `d3deae0882ce8228413fb6b05c3f7c8477321e58721325cd3715133b170f2c29`. Regeneration gives the same hash. |
| Actor compatibility | Kit `LeapPolicy` loads it: 320 → [512, 512, 256] → 16 (elu), normaliser, history 10, delta ±1/24 rad, tanh, 20 Hz, no cube estimator. Default pose equals training. The `describe()` output differs from C5_seed1 only in checkpoint path, config path and iteration. |
| Numerical equivalence | Kit runtime vs RSL-RL training actor: max \|Δ raw\| 2.3e-5, max \|Δ tanh\| 8.5e-6 (`results/deploy_verification.json`). The export's arrays are bit-exact to the checkpoint. Export forward vs kit runtime: max \|Δ\| = 0 over 4096 observations (`results/export_verification.json`). |
| Kit offline inspection | `kit/inspect_policy.py`, 600 open-loop steps: finite, within limits, "OK" (`logs/kit_inspect_candidate.log`). Action saturation 71% (C5_seed1 79%, A3 64%, same check). |
| Known kit/training gap (unchanged) | The kit's MCP/PIP limits differ from training by up to 0.0417 rad on if/mf/rf MCP and PIP. C5_seed1 and P0_seed1 have the same gap (the kit PIP-limit fix still awaits approval). |
| Closed-loop rehearsal (simulation, kit runtime in the loop) | All 3 modes are within their CI of the training actor (§5.1). |

### 5.1 Rehearsal

These are simulation-only rollouts in the candidate's own capped task (`Mjlab-Leap-C5HW-EFFORT`), with the kit runtime and bridge logic in the loop. Seed 5001, 128 environments, 30 s (`scripts/rehearse_deploy.py` → `results/deploy_rehearsal.json`).

| Mode | Turns (kit) | Training-actor reference | Δ ± 95% CI | Failure | Bridge clip, share of steps | Max command step |
|---|---|---|---|---|---|---|
| Kit reset "measured" (as on the hand) | 1.603 | 1.633 | −0.030 ± 0.086 | 0.188 | 0.982 | 0.0417 rad |
| Kit reset "target" (as in training) | 1.628 | 1.633 | −0.005 ± 0.070 | 0.148 | 0.984 | 0.0417 rad |
| Diagnostic: target + training limits in the kit runtime (in memory only) | 1.631 | 1.633 | −0.002 ± 0.058 | 0.125 | 0.000 | 0.0417 rad |

- **Observation error.** The kit's observation error after 10 steps is ≤ 6e-8 in every mode.
- **Bridge clips.** The clip share of about 98% is the known kit-vs-training MCP/PIP limit gap; it falls to 0 with training limits. The original C5_seed1 rehearsal shows the same (0.980 / 0.978 / 0.000; `grasp_rotation_v2/results/deploy_rehearsal.json`).
- **Command steps.** Steps never exceed 1/24 rad, so the kit's 0.05 rad step guard cannot trip.
- **Run-to-run variation (measured, reported in full).** A first rehearsal run was stopped by my 30-min `timeout` (exit 124) before writing results. Its two completed modes logged reference 1.657, measured 1.545 (−0.112 ± 0.088) and target 1.647 (−0.010 ± 0.048) (`logs/rehearse_deploy_timeout1800_killed.log`). The reference itself moved from 1.657 to 1.633 between the runs, because GPU simulation is not bit-reproducible. At 128 environments, the measured-reset gap is therefore uncertain. Its two estimates are −0.11 and −0.03 turns, against −0.03 for C5_seed1. A small loss from the hand-like reset cannot be ruled out.

## 6. Limitations: what is assumed, not measured

1. **One training seed.** Training-seed variance is unknown. Among the C5 seeds, turns30 spans 1.68–1.85 and the jam proxy 0.10–0.18. A second EFFORT seed could land on either side of the 0.90 rotation bar, which this run cleared by 0.011.
2. **The actuator response is not matched.** Simulated lag is about 280 ms; the hardware's is 80 ms, with about 3.5× tighter tracking (SPEC E6). Under the hardware's faster response, the force profile (and with it the current) may differ from the simulated one. The response fit of SPEC §4 was not done in this milestone.
3. **The current mapping is assumed.** 350 mA → 0.1855 Nm uses Kt = 0.53 Nm/A from `leap_right_constants.py`. The gain match was checked (SPEC E4); the torque constant itself was not measured on this hand.
4. **The jam proxy is not validated against hardware** (§4.2, finding 4). Its 96% reduction is a simulation result.
5. **The persistence of the hardware jam is not reproduced** in simulation (SPEC E5), and neither is the guard's spike mechanism (453–747 mA spikes above a 350 mA limit).
6. **Rotation is narrowly above its threshold.** It is below 0.90 on evaluation seed 6202, against C5_1 and C5_2, and in the heavy_slippery and high_delay presets. All of these are descriptive; the pre-registered pooled criterion is met.
7. **Validation curves are not comparable across tasks** (§2), and the run's own convergence check reports "still improving".

## 7. Deviations and additions (none changes a threshold or a decision rule)

- The SPEC's "hardware-matched" actuator (cap plus fitted response) was not available; the effort-matched actuator was declared as the decision actuator **before training** (PROTOCOL).
- The SPEC's seed-level Welch test is impossible with one seed; the paired environment bootstrap was declared before training.
- Three training seeds were replaced by one, on the user's instruction of 2026-11-24.
- Added after the result: `scripts/export_candidate.py` (export plus equivalence check) and the per-evaluation-seed and rotation-ratio CI analysis (`results/effort_v1_per_eval_seed.json`). These are extra checks and do not affect the verdict.
- Evaluation ran on several GPU hosts (gpu-01 and ws-ia nodes) with identical code; pairing was verified by cache-id equality.

## 8. Tests and preservation (final runs, 2026-11-24)

| Check | Result |
|---|---|
| `tests/test_c5_hw_v1.py`: the config-tree diff shows exactly the 2 changes, plus capped limits and a penalty equal to the offline recomputation on GPU, and C5 unchanged | **6 passed** (`logs/pytest_test_c5_hw_v1_final.log`) |
| Full training suite `tests/` | **87 passed, 1 failed** (`logs/pytest_training_suite_final.log`). The failure is `test_estimation_env.py::test_actor_observation_layout`, line 56 (the obs-delay assertion). It is the same assertion that failed on 23 Nov (`pytest_training_suite.log`: 81 passed / 1 failed). The same test **passed** in this session's earlier run (88/88, `logs/pytest_training_suite_effort_v1.log`). The test file is unchanged since 726a4be and no tracked file was modified, so the failure is **intermittent and pre-existing**. It was not modified, not skipped and not fixed. |
| Repository root `tests/` (kit audit, live monitor, package) | **37 passed** (`logs/pytest_root_final.log`) |
| `scripts/check_preservation.sh` | **All OK**. No tracked file outside this milestone differs from f8e47b8. No stray untracked files. 974 pre-existing RSL-RL checkpoints are unchanged; the hwlimit_ac (12), retention_v1 (2) and grasp_rotation_v2 (8) manifests match; 156 kit logs and audit outputs are unchanged; the grasp cache is unchanged. |

## 9. Files

- Configuration: `src/in_hand_rotation_mjlab/c5_hw_v1/`, `src/.../tasks/hand_cube/config/leap_right_c5_hw_v1/`.
- Tests: `tests/test_c5_hw_v1.py`.
- Scripts (`scripts/`): `calibrate_effort_weight.py`, `sbatch_train.sh`, `curves.py`, `freeze_checkpoints.py`, `eval_items.py`, `sbatch_eval.sh`, `compare_effort.py`, `build_deploy.py`, `export_candidate.py`, `rehearse_deploy.py`, `check_preservation.sh`.
- Results: `results/` (calibration, stage check, curves, convergence, items, comparison, per-seed, deploy and export verification, rehearsal), `figures/`, `checkpoints/` (+ `MANIFEST.txt`), `deploy/` (+ `MANIFEST.txt`).
- Raw per-environment arrays: `data/` (gitignored, 144 × npz).
