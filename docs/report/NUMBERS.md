# Number sources for REPORT.md

One row per number used in [REPORT.md](REPORT.md). The report section where it appears is given in brackets at the start of the meaning column; a number used again later with the same meaning is listed once (first use, later sections noted). All paths are relative to the repository root.

**Not listed:** section, figure and reference numbers; bibliographic details in the reference list (years, volumes, pages, DOIs, arXiv IDs); policy and condition labels (A1–A3, C2, C5, C5s3, R1, P0, `model_400`, RealSense "D435"); the fixed constant 2π in the definition of turns; and dates, which are taken from the file names of the session reports cited below (`docs/hardware_session_2026-11-{17,18,20,24,25}.md`) and from the research log in `README.md`.

| Number as written | Meaning | Source file | Where in that file |
|---|---|---|---|
| 58 mm | [Abstract, §2.1] cube edge length (MBZUAI cube) | docs/course_package.md | Intro paragraph ("keeps a 58 mm cube spinning") |
| seven (simulation studies) | [Abstract] number of simulation studies | README.md | First paragraph ("ran seven simulation studies") |
| five (real-hand sessions) | [Abstract, §1] supervised hardware sessions (17, 18, 20, 24, 25 Nov) | README.md | First paragraph ("five supervised sessions") |
| Four (bugs) | [Abstract, §1, §3] bugs fixed in the course training environment | README.md; training_source/project/experiments/scratch_state_estimation/README.md | README Research log, 12–13 Nov row ("Fixed 4 bugs"); study §1.1–1.4 |
| −0.041 turns [−0.150, +0.055] | [Abstract, §5, §10] C − A, held-out nominal turns under the kit joint limits, bootstrap 95% CI | training_source/project/experiments/hwlimit_ac/results/tables.md | "Held-out test, selected checkpoints", row `nominal | turns` |
| +0.593 … [+0.354, +0.833] | [Abstract, §6.4] C5 − A turns per 30 s, seed-level Welch 95% CI | training_source/project/experiments/grasp_rotation_v2/results/c5_seeds.md | "Seed-level inference", row turns30 |
| three (training seeds) | [Abstract, §1, §6.4] C5 seeds | training_source/project/experiments/grasp_rotation_v2/README.md | "C5 three-seed confirmation" |
| 3.48 | [Abstract, §7.2] C5 full revolutions per 60-s trial, 20 Nov | docs/hardware_policy_actuator_audit_2026-11-20.md | §3 table, C5 row |
| 0.50 | [Abstract, §7.2] `model_400` (and A3) full revolutions per trial, 20 Nov | docs/hardware_policy_actuator_audit_2026-11-20.md | §3 table, M400 and A3 rows |
| n = 5 and 4 | [Abstract] trials counted for C5 and `model_400` | docs/hardware_policy_actuator_audit_2026-11-20.md | §3 table, "Trials counted" column |
| 350 mA | [Abstract, §2.1, §8, §10, §11] current limit set on the hand's motors | docs/hardware_policy_actuator_audit_2026-11-20.md | §1 "Guards" row; §10 |
| 550 mA | [Abstract, §8.1, §8.2, §10] current limit of the training actuator | docs/hardware_policy_actuator_audit_2026-11-20.md | §10 first bullet; also c5_hw_v1/SPEC.md E4 |
| cache index 49 / grasp 49 | [§1, §6.2, §7, §11] nominal start grasp | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 1, "Nominal grasp B = cache index 49" |
| 16 Dynamixel motors | [§2.1, §5] motor count | docs/hardware_session_2026-11-17.md; docs/hardware_policy_actuator_audit_2026-11-20.md | Session Summary ("drives all 16 motors"); audit §1 `qd_hw` ("Dynamixel present velocity") |
| IDs 0–15 | [§2.1] motor IDs | docs/course_package.md | "Physical LEAP Hand", safety rules |
| 450 mA | [§2.1, §8.1] abort threshold on any motor current | docs/hardware_policy_actuator_audit_2026-11-20.md | §1 "Guards" row |
| 0.65 rad after step 10 | [§2.1] tracking-error abort | docs/hardware_policy_actuator_audit_2026-11-20.md | §1 "Guards" row |
| 8 rad/s | [§2.1] velocity abort | docs/hardware_policy_actuator_audit_2026-11-20.md | §1 "Guards" row |
| 60 °C | [§2.1] temperature limit | docs/hardware_policy_actuator_audit_2026-11-20.md | §1 "Guards" row |
| 0.05 rad | [§2.1] command-step abort | docs/hardware_policy_actuator_audit_2026-11-20.md | §1 "Guards" row |
| six faces, 15.55 mm, `DICT_4X4_50` | [§2.1] cube markers | docs/live_vision_digital_twin.md | §1 table, "Cube geometry" row |
| 320 | [§2.1, §4] actor observation dimension | training_source/project/experiments/scratch_state_estimation/README.md | §2.1 Conditions table, row A |
| 10 frames, 16 joint positions, 16 previous commands | [§2.1] observation history layout | training_source/project/experiments/scratch_state_estimation/README.md | §2.1 Conditions table, row A |
| ±0.01 rad, 0–1 step delay | [§2.1] simulated joint-position noise and delay | training_source/project/experiments/scratch_state_estimation/README.md | §2.1 Conditions table, row A |
| 16-D | [§2.1] action dimension | training_source/project/experiments/scratch_state_estimation/README.md | §1.4 Mechanism |
| ±1/24 rad (0.0417 rad) | [§2.1] maximum per-step joint-target change | training_source/project/experiments/scratch_state_estimation/README.md; training_source/project/experiments/hwlimit_ac/README.md | §1.4; hwlimit §2.3 table, "Command step" row |
| 20 Hz, 0.005 s, decimation 10 | [§2.1] control rate and physics step | training_source/project/experiments/hwlimit_ac/README.md | §2.3 table, "Control rate" row |
| 50.2 ms | [§2.1] measured hand loop period, 17 Nov | docs/hardware_session_2026-11-17.md | §2 Function test, "20 Hz tracking" |
| 3.8.1 (`mujoco`, `mujoco-warp`) | [§2.2] pinned simulator versions | docs/course_package.md | Training, "Dependency fix" |
| ±0.25 rad/s, ×0.1, 2 cm, 0.35 rad, 1.25 | [§2.2] yaw-rate reward clip, drift-gate factor and thresholds, weight | training_source/project/experiments/scratch_state_estimation/README.md | §2.2 Reward table, first row |
| −0.3 | [§2.2] cube linear-velocity L1 weight | training_source/project/experiments/scratch_state_estimation/README.md | §2.2 Reward table |
| 0.4 rad, −0.1 | [§2.2] joint-pose deviation threshold and weight | training_source/project/experiments/scratch_state_estimation/README.md | §2.2 Reward table |
| −0.1 (torque²) | [§2.2] torque² weight | training_source/project/experiments/scratch_state_estimation/README.md | §2.2 Reward table |
| −0.05 (work) | [§2.2] work weight | training_source/project/experiments/scratch_state_estimation/README.md | §2.2 Reward table |
| −10 (cube fallen) | [§2.2] cube-fallen weight | training_source/project/experiments/scratch_state_estimation/README.md | §2.2 Reward table |
| 20 s, 0.2 m, 1 m/s, 0.8 rad, 0.08 m | [§2.2] terminations: time-out, drop height, speed, tilt, drift | training_source/project/experiments/scratch_state_estimation/README.md | §2.2 "Terminations" |
| ×0.7–1.4, 0.6–1.4, ±2 mm, 5–15 physics steps, ×0.9–1.1 | [§2.2] DR ranges: cube mass, friction, cube CoM, actuator delay, effort limit | training_source/project/experiments/scratch_state_estimation/README.md | §2.2 Domain randomization table |
| 4096 | [§2.2] parallel environments | training_source/project/experiments/scratch_state_estimation/README.md | §2.1 "PPO" bullet |
| 32, 5 epochs × 4 minibatches, γ 0.99, λ 0.95, clip 0.2, 0.003, 1e-3, KL 0.01 | [§2.2] PPO hyperparameters | training_source/project/experiments/scratch_state_estimation/README.md | §2.1 "PPO" bullet |
| 512-512-256 | [§2.2, §4.1] actor/critic (and estimator) hidden layers | training_source/project/experiments/scratch_state_estimation/README.md | §2.1 "Critic" bullet; §2.4 Model |
| 83-D | [§2.2] privileged critic input | training_source/project/experiments/scratch_state_estimation/README.md | §2.1 "Critic" bullet |
| 1001 updates (131.2M environment steps) | [§2.2, §4.2] budget of the first study | training_source/project/experiments/scratch_state_estimation/README.md | §2.1 "Budget" |
| 3001 updates (393 M environment steps) | [§2.2, §5] budget from the hwlimit study on | training_source/project/experiments/hwlimit_ac/README.md | §4 Training setup table, "Budget" row |
| RTX 5000 Ada | [§2.2] GPU model | training_source/project/experiments/scratch_state_estimation/README.md | §5 Compute, "Hardware" |
| 30-s window | [§2.3] evaluation horizon | training_source/project/experiments/scratch_state_estimation/README.md | §2.5 "600 control steps (30 s)" |
| 60-s turns, P(held 60 s) | [§2.3] secondary metrics | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 4 Setting A table columns |
| 7001, 1024 environments | [§2.3] validation seed and size | training_source/project/experiments/scratch_state_estimation/README.md | §2.5 "Checkpoint selection" |
| 5001/5002 | [§2.3, §4.2, §5] held-out test seeds, first three studies | training_source/project/experiments/scratch_state_estimation/README.md; training_source/project/experiments/hwlimit_ac/README.md | §2.5 "Test"; hwlimit §8.2 |
| 6201/6202 | [§2.3, §6.3, §8.2, §8.3] held-out test seeds, GRV2, C5-EFFORT and C5-HW-V2 | training_source/project/experiments/grasp_rotation_v2/README.md | Step 0, "Seeds" |
| ×1.3–1.8, 0.4–0.7 | [§2.3] heavy_slippery mass and friction | training_source/project/experiments/scratch_state_estimation/README.md | §2.5 presets table |
| n = 3 / three per condition | [§2.3, §4.2, §10] training seeds per condition | training_source/project/experiments/scratch_state_estimation/README.md | §2.5 "Statistics" |
| 17 Nov (pre-registration start) | [§2.3, §2.2] first pre-registered study | README.md | "How the work was run", Pre-registration bullet |
| ≤ 3.1e-5 | [§2.3] kit runtime vs training actor, hardware-limited policies | training_source/project/experiments/hwlimit_ac/README.md | §8.5 "Offline verification" |
| 5 s (placement), 5 s (ramp), 60 s | [§2.3] real-hand trial timing | docs/hardware_trials_A3_C5_R1_P0.md | Design, "One trial = …" |
| 10 cm | [§3] hand placed too low in portable scene | docs/course_package.md | Status, "Fixed 2026-11-11" |
| −2.4 turns | [§3] earlier wrong result caused by that scene | docs/course_package.md | Status, "Fixed 2026-11-11" |
| seed 11, 1.31 turns in 30 s | [§3] `model_400` reference replay | docs/course_package.md | Status table, `replay.sh` row |
| \|pitch\| = 90° | [§3] Euler singularity | training_source/project/experiments/scratch_state_estimation/README.md | §1.2 Bug |
| ≥ 30°, about half, 98–100%, 7–32° | [§3] tilted starts wrongly terminated | training_source/project/experiments/scratch_state_estimation/README.md | §1.2 "Measured consequence" |
| 120° | [§3] reset teleport in the yaw-increment test | training_source/project/experiments/scratch_state_estimation/README.md | §1.3 Tests |
| ±1 | [§3] action clip of the original wrapper | training_source/project/experiments/scratch_state_estimation/README.md | §1.4 Mechanism |
| 95%, 12.8, 94% | [§3] `model_400` saturation, median \|μ\|, ineffective exploration | training_source/project/experiments/scratch_state_estimation/README.md | §1.4 Mechanism, last bullet |
| 17 CPU, 5 GPU tests | [§3] tests of the fixes | training_source/project/experiments/scratch_state_estimation/README.md | §1.7 table |
| Four (conditions) | [§4.1] A, B, C, D | training_source/project/experiments/scratch_state_estimation/README.md | §2.1 Conditions table |
| 19 features | [§4.1] estimator features for C and D | training_source/project/experiments/scratch_state_estimation/README.md | §2.1 Conditions and Features tables |
| 24 cube rotations | [§4.1] symmetry group in the rotation loss | training_source/project/experiments/scratch_state_estimation/README.md | §2.4 Rotation |
| 5,529,600 rows; A seeds 2 and 3 | [§4.1] estimator training split | training_source/project/experiments/scratch_state_estimation/README.md | §2.3 split table, train row |
| 1,536,000 rows; A seed 1 | [§4.1] estimator test split (held-out policy) | training_source/project/experiments/scratch_state_estimation/README.md | §2.3 split table, test row; §3.1 |
| 10.2 mm, 16.8° | [§4.1] B position RMSE, orientation mean | training_source/project/experiments/scratch_state_estimation/README.md | §3.1 table, MLP column |
| 15.2 mm, 39.2° | [§4.1] ridge baseline | training_source/project/experiments/scratch_state_estimation/README.md | §3.1 table, Ridge column |
| 0.79 (micro-F1) | [§4.1] contact micro-F1 | training_source/project/experiments/scratch_state_estimation/README.md | §3.1 table |
| R² 0.41, 0.35 | [§4.1] velocity R² (linear, yaw rate) | training_source/project/experiments/scratch_state_estimation/README.md | §3.1 "Velocity is not reliable" |
| 5.6 mm, 7°; 9–16 mm, 12–20° | [§4.1] closed-loop errors, training vs unseen policies | training_source/project/experiments/scratch_state_estimation/README.md | §3.4 "Training versus unseen policies" |
| six physics presets | [§4.2] test presets | training_source/project/experiments/scratch_state_estimation/README.md | §2.5 presets table |
| 0.756, 0.906, 1.018 | [§4.2] pooled turns A, C, D | training_source/project/experiments/scratch_state_estimation/README.md | §3.3 turns table, "Pooled" row |
| +0.150 [−0.006, +0.339] | [§4.2, §10] C − A pooled turns | training_source/project/experiments/scratch_state_estimation/README.md | §3.3 "Condition differences" table |
| +0.263 [+0.148, +0.376] | [§4.2] D − A pooled turns | training_source/project/experiments/scratch_state_estimation/README.md | §3.3 "Condition differences" table |
| +0.113 [−0.075, +0.276] | [§4.2] D − C pooled turns | training_source/project/experiments/scratch_state_estimation/README.md | §3.3 "Condition differences" table |
| about 57% | [§4.2] share of oracle gain recovered by C | training_source/project/experiments/scratch_state_estimation/README.md | §3.3 Observations, "Share of the oracle gain" |
| 0.686, 0.473, p = 0.05 | [§4.2] mean validation turns C vs A, permutation p | training_source/project/experiments/scratch_state_estimation/README.md | §3.2 learning-speed table and bullet |
| 19–71% | [§4.2] C turns kept under feature ablation | training_source/project/experiments/scratch_state_estimation/README.md | §3.5 "Observed" |
| 0.187 N, R² 0.78 | [§4.3] estimator MAE, R² on unseen policies | training_source/project/experiments/force_estimation/README.md | §8.3 table, estimator row |
| 0.301 N | [§4.3] ridge MAE | training_source/project/experiments/force_estimation/README.md | §8.3 table |
| 0.94–0.98 | [§4.3] contact AUROC | training_source/project/experiments/force_estimation/README.md | Headline |
| 61% | [§4.3] MAE increase, seen → unseen policies (0.116 → 0.187 N) | training_source/project/experiments/force_estimation/README.md | §8.4 "The main generalization gap…" |
| five robustness presets, 0–12% | [§4.3] extra MAE under presets | training_source/project/experiments/force_estimation/README.md | §8.4 |
| about 2 N | [§4.3] underestimation threshold | training_source/project/experiments/force_estimation/README.md | §8.3 "By force magnitude" |
| all 5 runs, r +0.23 to +0.44 | [§4.3] thumb force vs current, 18 Nov | docs/hardware_session_2026-11-18.md | §2 Reading, "Thumb" |
| r −0.18 to +0.17 | [§4.3] index force vs current, 18 Nov | docs/hardware_session_2026-11-18.md | §2 Reading, "Index" |
| A seed 2, C seed 2 | [§5] policies run on 17 Nov | docs/hardware_session_2026-11-17.md | Status line |
| ≤ 390 mA | [§5] max current, 17 Nov | docs/hardware_session_2026-11-17.md | Summary, "Hardware runs completed" |
| ±0.52 rad; ±1.05 rad | [§5] kit vs training rot-joint range | docs/hardware_session_2026-11-17.md | Summary, "Main finding" |
| 87–98% | [§5] steps with a joint at the controller limit | docs/hardware_session_2026-11-17.md | Summary, last bullet |
| 0.14, 0.59; 0.53, 0.62; 1.01, 0.72 | [§5] rehearsal cost of limits and resulting turns (A, C) | docs/hardware_session_2026-11-17.md | Summary, "Main finding (simulation)"; §4 table |
| three seeds each, 3001 updates | [§5] A vs C under the kit limits | training_source/project/experiments/hwlimit_ac/README.md | §4 table |
| 10 of 449 | [§5] original grasp commands inside the kit limits | training_source/project/experiments/hwlimit_ac/README.md | §2.2 item 2 |
| 239 grasps | [§5, §6.2, Figure 1] new grasp cache | training_source/project/experiments/hwlimit_ac/README.md | §2.2 cache table |
| 98.1%, 95.4% | [§5] 10-s retention, new vs original cache | training_source/project/experiments/hwlimit_ac/README.md | §2.2 cache table |
| 1.163 / 1.141 / 1.326; 1.135 / 1.225 / 1.147 | [§5] per-seed nominal turns, A and C | training_source/project/experiments/hwlimit_ac/results/tables.md | Held-out test, row `nominal | turns` |
| 1.210, 1.169 | [§5] A and C mean nominal turns | training_source/project/experiments/hwlimit_ac/results/tables.md | Held-out test, row `nominal | turns` |
| Welch [−0.253, +0.170] | [§5] C − A Welch 95% CI | training_source/project/experiments/hwlimit_ac/results/tables.md | Held-out test, row `nominal | turns` |
| 0.241, 0.256, +0.014 [−0.020, +0.049] | [§5] nominal failure rate A, C, difference | training_source/project/experiments/hwlimit_ac/results/tables.md | Held-out test, row `nominal | failure` |
| −0.06 and 0.00 | [§5] range of C − A across presets | training_source/project/experiments/hwlimit_ac/README.md | "Answer" bullet 2 |
| 100–3000, A 0.98, C 0.95 | [§5] mean validation turns over training | training_source/project/experiments/hwlimit_ac/README.md | §8.1 "Learning speed is the same" |
| Four of the six runs | [§5] runs meeting the plateau criteria | training_source/project/experiments/hwlimit_ac/README.md | "Answer" bullet 3 |
| 11.0–11.8 mm; 8.1 mm | [§5] B position RMSE on new C / A gaits vs reference | training_source/project/experiments/hwlimit_ac/README.md | §8.4 table |
| 1.885 rad, 1.825 rad | [§5, §10] kit vs training PIP upper limit | training_source/project/experiments/hwlimit_ac/README.md | §8.5 "Open item: PIP upper limit" |
| −0.19 ± 0.08 | [§5] C seed 2 rehearsal loss from the PIP gap | training_source/project/experiments/hwlimit_ac/README.md | §8.5 rehearsal table, C2 row |
| seed 1 each; two 30-s trials | [§5] 18 Nov pilot | docs/hardware_session_2026-11-18.md | Status; §1 table |
| −0.25 to +0.04 | [§5] estimated rotation on the hand, 18 Nov | docs/hardware_session_2026-11-18.md | §2 "Estimated cube rotation is near zero" |
| 19–22%, 60 ms | [§5] late steps on 18 Nov | docs/hardware_session_2026-11-18.md | §3 table |
| 16 ms; 1 ms | [§5, §7.1] USB latency timer default and fixed value | docs/hardware_session_2026-11-18.md | §3 "Likely cause"; table header |
| 4.9% | [§5] mean late steps, 20 Nov | docs/hardware_session_2026-11-20.md | §1 "Timing" |
| 99.4% | [§6.1] share of A failures that are tilt failures | training_source/project/experiments/retention_v1/README.md | §2 "What simulation shows", item 1 |
| 6.4 s | [§6.1] median failure time (50% quantile) | training_source/project/experiments/retention_v1/README.md | §2 item 1, failure-time quantiles |
| 0.81–0.83 | [§6.1] per-grasp survival correlation between A seeds | training_source/project/experiments/retention_v1/README.md | §2 item 7 |
| −0.05; 0.5 rad; −10 | [§6.1] retention-penalty weight, tilt margin, grasp-loss weight | training_source/project/experiments/retention_v1/README.md | §3 "New terms" table |
| 24.1%, 17.2%, −29% | [§6.1] nominal failure A vs R1 | training_source/project/experiments/retention_v1/README.md | §6.2 table, nominal row |
| 1.164 vs 1.210; −0.046 [−0.155, +0.031] | [§6.1] nominal turns R1 vs A | training_source/project/experiments/retention_v1/README.md | §6.2 table, nominal row |
| 11% | [§6.1] R1's lower rotation speed while holding | training_source/project/experiments/retention_v1/README.md | §6.2 "Rotation speed while holding" |
| ≥ 1.39 | [§6.1] pre-registered turns target T1 | training_source/project/experiments/retention_v1/README.md | §6.4 table, T1 |
| 0.99 | [§6.2] split-half reliability of the grasp score | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 1, grasp analysis, first bullet |
| ρ = −0.70 | [§6.2] Spearman of initial tilt with grasp score | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 1, "What makes a grasp bad" |
| four existing policies; 1.189 to 1.599 (+34%) | [§6.2] mean turns, full distribution vs grasp 49 | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 1 held-out table and "Pre-registered decision" |
| 23.6% to 0.7% | [§6.2] mean failure, full distribution vs grasp 49 | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 1 held-out table |
| ±2 mm, ±10°, 94% | [§6.2] placement noise and retained gain | training_source/project/experiments/grasp_rotation_v2/README.md | Summary, "Grasp initialisation" bullet |
| 0.28 rad RMS | [§6.2] grasp 49 distance from kit start pose | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 1, "Nominal grasp" sub-bullet |
| ±0.25 rad/s; 56–78% | [§6.3] yaw clip and share of saturated steps | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 2, second bullet |
| −0.050; +0.282 | [§6.3] A − R1 return: as trained / H1 (no position part) | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 2 variant table |
| ±0.5 rad/s | [§6.3] C5 yaw clip | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 2, selected candidates |
| 1.200, 25.2%, 0.295, 0.732 | [§6.3] A mean (3 seeds), Setting A | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 4 Setting A table, "A mean" |
| 1.158, 19.0%, 0.263, 0.763 | [§6.3, §7.2] R1, Setting A | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 4 Setting A table, R1 row |
| 1.273, 11.5%, 0.282, 0.862 | [§6.3, §7.2] P0, Setting A | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 4 Setting A table, P0 row |
| 1.869, 19.9%, 0.432, 0.788 | [§6.3] C5 seed 1, Setting A | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 4 Setting A table, C5 row |
| five (thresholds) | [§6.3, §6.4] pre-registered Phase 4 thresholds | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 4 thresholds table |
| 0.001 | [§6.3] margin by which P0 missed threshold 2 | training_source/project/experiments/grasp_rotation_v2/README.md | Phase 4, below thresholds table |
| +18% | [§6.3] C5's higher mean joint speed | training_source/project/experiments/grasp_rotation_v2/README.md | "Movement cost of C5" |
| seeds 2 and 3 | [§6.4] additional C5 seeds | training_source/project/experiments/grasp_rotation_v2/README.md | "C5 three-seed confirmation" |
| 1.793 | [§6.4] C5 3-seed mean turns30 | training_source/project/experiments/grasp_rotation_v2/results/c5_seeds.md | Setting A table, `C5_3seed_mean` |
| 0.0025 | [§6.4] p, C5 − A turns30 | training_source/project/experiments/grasp_rotation_v2/results/c5_seeds.md | Seed-level inference, turns30 |
| 18.4% | [§6.4] C5 3-seed mean failure30 | training_source/project/experiments/grasp_rotation_v2/README.md | C5 three-seed table, "C5 3-seed mean" |
| −0.068 [−0.122, −0.013], 0.0268 | [§6.4] C5 − A failure30 | training_source/project/experiments/grasp_rotation_v2/results/c5_seeds.md | Seed-level inference, fail30 |
| +0.117 [+0.067, +0.167], 0.0031 | [§6.4] C5 − A yaw rate held | training_source/project/experiments/grasp_rotation_v2/results/c5_seeds.md | Seed-level inference, yaw30 |
| +0.070 [+0.009, +0.131], 0.0338 | [§6.4] C5 − A P(held 60 s) | training_source/project/experiments/grasp_rotation_v2/results/c5_seeds.md | Seed-level inference, held60 |
| −0.005 [−0.070, +0.060] | [§6.4] C5 − R1 failure30 | training_source/project/experiments/grasp_rotation_v2/results/c5_seeds.md | Seed-level inference, fail30, C5 − R1 column |
| 2.361, 1.597, +0.764 [+0.407, +1.120], 0.0047 | [§6.4] grasp 49: C5 3-seed mean, A mean, difference, p | training_source/project/experiments/grasp_rotation_v2/results/c5_seeds.md | "Setting B: nominal grasp 49" table and line below |
| A and C5: 3 seeds; R1, P0, C5-EFFORT: 1 seed (Figure 1) | [§6.4] seeds per bar | docs/figures/readme/headline_numbers.json | `simulation.policies[*].full` (array lengths) |
| 550 mA (Figure 1 C5-EFFORT bars) | [§6.4] actuator used for the C5-EFFORT bars | docs/figures/readme/headline_numbers.json | `simulation.policies` C5-EFFORT `note` ("default actuator") |
| 1.625 | [§6.4, §8.2] C5-EFFORT turns30, decision setting | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §3.1 table, Turns / 30 s |
| five 60-s trials each; four trials of `model_400` | [§7.1] 20 Nov design | docs/hardware_session_2026-11-20.md | Status line; §1 Protocol |
| seed 3 (A3) | [§7.1, Figure 3] A3 is A training seed 3 | docs/hardware_session_2026-11-20.md | §1 Policies table, A3 row |
| 90° | [§7.1] one quarter turn | docs/hardware_policy_actuator_audit_2026-11-20.md | §3, first sentence |
| quarter turns / 4 | [§7.1] conversion to full revolutions | docs/hardware_policy_actuator_audit_2026-11-20.md | §3, first sentence |
| 37 | [§7.2, §8.1] attempts (run logs) on 20 Nov | docs/hardware_policy_actuator_audit_2026-11-20.md | §2 table, Total row; header |
| 4 / 0 / 4 / 0 / 0; 0.50 ± 0.20 (4); 13.5 | [§7.2] `model_400` attempts / completed / dropped / guard / unknown; revolutions; time held | docs/hardware_policy_actuator_audit_2026-11-20.md | §2 table and §3 table, M400 rows |
| 7 / 3 / 2 / 1 / 1; 0.50 ± 0.35 (4); 40.6 | [§7.2] A3 | docs/hardware_policy_actuator_audit_2026-11-20.md | §2 table and §3 table, A3 rows |
| 6 / 0 / 5 / 1 / 0; 0.65 ± 0.34 (5); 9.2 | [§7.2] R1 | docs/hardware_policy_actuator_audit_2026-11-20.md | §2 table and §3 table, R1 rows |
| 9 / 0 / 4 / 3 / 2; 0.00 (5); 10.8 | [§7.2] P0 | docs/hardware_policy_actuator_audit_2026-11-20.md | §2 table and §3 table, P0 rows |
| 11 / 4 / 1 / 6 / 0; 3.48 ± 0.55 (5); 57.3 | [§7.2] C5 | docs/hardware_policy_actuator_audit_2026-11-20.md | §2 table and §3 table, C5 rows |
| 37 / 7 / 16 / 11 / 3 | [§7.2] totals | docs/hardware_policy_actuator_audit_2026-11-20.md | §2 table, Total row |
| P0: 3 drops and 4 stops (Figure 2) | [§7.2] figure's classification of P0 | docs/figures/readme/headline_numbers.json | `real_hand` P0 entry: `dropped`, `safety_stops` |
| +11.90 [+9.85, +14.10], p = 0.009 | [§7.2] C5 − A3 quarter turns per trial | LEAP_Hand_RL_Deploy_Kit/logs/trial_comparison.md | Q1 row |
| n = 5 and 4 (C5, A3) | [§7.2] trials counted | LEAP_Hand_RL_Deploy_Kit/logs/trial_comparison.md | Per-policy table, "rotations per trial" column |
| 3.64; about 3.7 rev/min | [§7.2] C5 rate while held, hand vs simulation | docs/hardware_policy_actuator_audit_2026-11-20.md | §3 table (C5) and "For scale" bullet |
| all 5 recorded trials each; about 10 s | [§7.2, §10] R1 and P0 drops on 20 Nov | docs/hardware_session_2026-11-20.md | Summary, Q2 row and main finding 2 |
| −31.41 s [−51.76, −9.66], p = 0.069 | [§7.2] R1 − A3 time held | LEAP_Hand_RL_Deploy_Kit/logs/trial_comparison.md | Q3 row, time held |
| 3 of 6; 4 of 11 | [§7.2] A3 and C5 attempts that held 60 s (unknown-reason operator stops excluded) | docs/figures/readme/headline_numbers.json | `real_hand` A3 and C5 (20 Nov) entries: `held`, `attempts` |
| 6 of 11; 5 of them the current guard | [§7.2, §10] C5 attempts ending in a guard stop; of which current-guard trips (the sixth has guard type "unrecorded") | docs/hardware_policy_actuator_audit_2026-11-20.md; LEAP_Hand_RL_Deploy_Kit/audit/20261120/attempts.csv | §2 table, C5 row; guard_type column of the C5 rows |
| 256 environments; 350 mA; grasp 49 | [§7.3] seed screen setup | docs/hardware_trials_C5_seeds_2026-11-24.md | "Jam screen in simulation" |
| 3.6%; 7.5% | [§7.3] index MCP/PIP saturated, seed 3 vs seed 1 | docs/hardware_trials_C5_seeds_2026-11-24.md; training_source/project/experiments/c5_hw_v1/results/c5_seed_screen.json | Jam-screen table; JSON `sat_frac_index_mcp_pip` of the `cap350` entries |
| 0 … in 13 attempts; 3 of 5; p = 0.012 | [§7.3] safety stops seed 3 vs seed 1, 24 Nov (Fisher) | docs/hardware_session_2026-11-24.md | "Answer", Guard stops |
| 9 of 16; p = 0.001 | [§7.3, §10] seed 1 over 20 + 24 Nov | docs/hardware_session_2026-11-24.md | "Answer", Guard stops |
| ≥ 330 mA; 0.2–0.3 rad | [§7.3] seed 1 jam signature, 24 Nov | docs/hardware_session_2026-11-24.md | "Same jam as 20 Nov" |
| 20 s; 17.6 (n = 5); about 4.4 rev/min; 13.0 (n = 2) | [§7.3] rotation rate while held ≥ 20 s | docs/hardware_session_2026-11-24.md | Results summary table, last row; "Answer", Speed |
| 11 of 12 | [§7.3, §10] seed 3 drops among valid trials | docs/hardware_session_2026-11-24.md | Results summary table, Drops |
| 5–7 s in 6 trials; 45–60 s in 5 | [§7.3, §8.3, §10] bimodal outcome of seed 3 | docs/hardware_session_2026-11-24.md | "Answer", Retention |
| 0.008 rad | [§7.3] max joint difference after ramp, early-drop vs long-hold | docs/hardware_session_2026-11-24.md | "Start placement", What the logs can show |
| five (trials, 25 Nov); 16 quarter turns; four drops at 5–30 s | [§7.3] 25 Nov outcomes | docs/hardware_session_2026-11-25.md; LEAP_Hand_RL_Deploy_Kit/logs/trials_20261125.csv | Runs table; CSV `drop_time_s` and `notes` |
| 23 counted trials; Spearman −0.42 | [§7.4, §10] estimated yaw vs operator counts | docs/hardware_session_2026-11-20.md | §4 "Cube rotation" |
| 11–17; −1.1 to −2.9 | [§7.4] C5 operator counts vs estimated turns | docs/hardware_session_2026-11-20.md | §4 table, C5 row |
| 5 of C5's 6 guard stops | [§8.1] stops that were sustained index jams | training_source/project/experiments/c5_hw_v1/SPEC.md | §1 table, E1 |
| 0.2–0.6 rad | [§8.1] index shortfall from command during jams | docs/hardware_policy_actuator_audit_2026-11-20.md | Executive summary |
| 453–747 mA | [§8.1] trip transients | docs/hardware_policy_actuator_audit_2026-11-20.md | Executive summary |
| 2–4 s; about 3.5 s | [§8.1] timing of index loading and clearance | docs/hardware_policy_actuator_audit_2026-11-20.md | Executive summary table, "successful vs failed" row |
| 1,450–1,500 mA/rad; 0.23–0.27 rad | [§8.1] current per unit error; saturation error | docs/hardware_policy_actuator_audit_2026-11-20.md | §4 "Current vs tracking error" |
| Four guard stops; 2.9 s; 300–850 mA | [§8.1] start transients of A3, R1, P0 | training_source/project/experiments/c5_hw_v1/SPEC.md; docs/hardware_policy_actuator_audit_2026-11-20.md | SPEC §1 E9 ("4 guard stops"); audit §8 "Two mechanisms", item 2 |
| 1,440 vs 1,493 mA/rad | [§8.1] simulated vs measured gain, index MCP | docs/hardware_policy_actuator_audit_2026-11-20.md | §10 first bullet |
| 7.6%; 3.5%; 0.08%; 0.09% | [§8.1] index MCP/PIP at the limit under a 350 mA cap: C5, P0, A3, R1 | docs/hardware_policy_actuator_audit_2026-11-20.md | §10 table |
| 2.46 → 2.44 | [§8.1] C5 turns per 30 s, default → capped (grasp 49) | docs/hardware_policy_actuator_audit_2026-11-20.md | §10 table, C5 row |
| 80 ms (IQR 75–85 ms) | [§8.1, §10, §11] effective lag on the hand | docs/hardware_policy_actuator_audit_2026-11-20.md | §5 "Effective lag" |
| about 280 ms | [§8.1, §10, §11] effective lag in simulation | docs/hardware_policy_actuator_audit_2026-11-20.md | §5 "The same estimators on simulation" |
| 3.5× | [§8.1] simulated vs hardware tracking error | docs/hardware_policy_actuator_audit_2026-11-20.md | §5, first sub-bullet under simulation |
| two changes | [§8.2] changes from C5 | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §2 table, Task row |
| 0.262–0.321 Nm; 0.167–0.204 Nm; 0.53 Nm/A; ±10% | [§8.2] effort-limit randomisation before / after | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §2 table, Change 1 |
| 0.8; 0.2 | [§8.2] utilisation penalty threshold and scale | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §2 table, Change 2 |
| w = 0.2201; 10% | [§8.2] penalty weight and calibration target | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §2 table, Change 2 |
| One training seed | [§8.2] C5-EFFORT seeds | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | "Scope" |
| 1 s | [§8.2] jam-proxy minimum episode length | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §4 Definition |
| 0.51%; 14.6% | [§8.2] jam proxy (60 s) C5-EFFORT vs C5 3-seed mean | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §1 Headline, jam proxy row |
| ≥ 50%; 96.5% (CI 95.0–97.9%) | [§8.2] criterion 1a and result | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §3.2 table, 1a |
| 1.784 | [§8.2] C5 3-seed mean turns30, decision setting | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §3.1 table |
| ≥ 0.90; 0.911 (CI 0.903–0.919) | [§8.2] criterion 2a and result | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §3.2 table, 2a |
| ≥ 1.20; 1.362 | [§8.2] criterion 2b and result | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §3.2 table, 2b |
| 0.178; 0.180; + 0.03 | [§8.2] failure30 EFFORT vs C5 mean; criterion 3 margin | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §3.1 table; §3.2 table, 3 |
| three C5 training seeds (Figure 4) | [§8.2] C5_1–C5_3 | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §3.1 table |
| 0.899 (seed 6202) | [§8.2] rotation ratio on evaluation seed 6202 | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §3.3 table |
| 0.878; 0.894 | [§8.2] rotation ratio vs C5_1 and C5_2 | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §3.4 table |
| 80%; 0.83%; 3.0–4.1% | [§8.2] share of joint samples with u ≥ 0.8, EFFORT vs C5 seeds | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §4.2 finding 1, band table |
| about 9% | [§8.2] rotation-speed cost | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §1 Headline, Cost row |
| 13.9% vs 14.6% | [§8.2] jam proxy A mean vs C5 mean | training_source/project/experiments/c5_hw_v1/REPORT_EFFORT_V1.md | §4.2 finding 4 |
| 24-point grid | [§8.3] response-fit grid (delay × damping × friction loss) | training_source/project/experiments/c5_hw_v2/REPORT.md | §3 Method |
| ±30% | [§8.3] pre-registered tracking band (0.70–1.30× hardware) | training_source/project/experiments/c5_hw_v2/REPORT.md | §3 Tracking band |
| 2.81× | [§8.3] best simulated tracking error / hardware | training_source/project/experiments/c5_hw_v2/results/response_fit.md | grid table, d0-25_D0.5_F0.5 |
| 218 ms; 79 ms | [§8.3] best-lag point median lag; hardware median lag | training_source/project/experiments/c5_hw_v2/results/response_fit.md | header "Hardware targets"; grid table |
| 95 ms; 364 ms | [§8.3] free-air median lag, hand vs simulation (current setting) | training_source/project/experiments/c5_hw_v2/results/response_fit.md | Free-air cross-check (0.095; 0.36375 s) |
| 4,035 | [§8.3] kit starts kept | training_source/project/experiments/c5_hw_v2/cache/kit_start_cache.json | "kept": 4035 |
| ±4 mm / ±15° | [§8.3] placement noise | training_source/project/experiments/c5_hw_v2/FIT_PLAN.md | §B item 2 |
| 5-s | [§8.3] closing ramp | training_source/project/experiments/c5_hw_v2/FIT_PLAN.md | §B item 4 |
| 95.3–95.8%; 93.0% | [§8.3] 30-s survival from kit starts, C5 seeds 1–3 and C5-EFFORT (effort actuator) | training_source/project/experiments/c5_hw_v2/results/kit_start_survival.md | K survival column, effort rows |
| 99.9–100%; 98.7% | [§8.3] 30-s survival from grasp 49, C5 seeds and C5-EFFORT (effort actuator) | training_source/project/experiments/c5_hw_v2/results/kit_start_survival.md | B survival column, effort rows |
| 0.154 rad | [§8.3] max simulated vs hardware post-ramp joint difference (if_pip) | training_source/project/experiments/c5_hw_v2/cache/kit_start_cache.json | "sim_minus_hw_abs_max_rad" |
| 3001 updates; FAIL (single-seed exploratory) | [Abstract, §1, §8.3, §10] C5-HW-V2 training length and pre-registered verdict | training_source/project/experiments/c5_hw_v2/REPORT.md; training_source/project/experiments/c5_hw_v2/results/v2_comparison.md | §2 table; verdict line |
| 0.5 × both; ≤ C5-EFFORT; ≥ 0.85 × C5 | [§8.3] C5-HW-V2 criteria 1–3 | training_source/project/experiments/c5_hw_v2/PROTOCOL.md | "Decision" |
| 0.045; 0.046; 0.061 | [§8.3] kit-start 30-s failure, C5-HW-V2 / C5 3-seed mean / C5-EFFORT | training_source/project/experiments/c5_hw_v2/results/v2_comparison.json | criteria.1: failure_new 0.0454, failure_C5_mean 0.0461, failure_EFFORT_1 0.0613 |
| 4.39%; 14.6%; 0.51% | [§8.3] jam proxy 60 s, random grasps (setting A), C5-HW-V2 / C5 mean / C5-EFFORT | training_source/project/experiments/c5_hw_v2/results/v2_comparison.json; training_source/project/experiments/c5_hw_v2/results/v2_per_eval_seed.json | criteria.2 (0.0439, 0.0051); settings.A.pooled.jam60.C5_mean 0.146 |
| 0.956; 1.784; 1.625; 0.536 | [§8.3] turns per 30 s, setting A, C5-HW-V2 / C5 mean / C5-EFFORT; ratio vs C5 | training_source/project/experiments/c5_hw_v2/results/v2_comparison.json; training_source/project/experiments/c5_hw_v2/results/v2_per_eval_seed.json | criteria.3; settings.A.pooled.turns30.EFFORT_1 1.625 |
| 0.924 | [§8.3] kit-start turns ratio vs C5 mean | training_source/project/experiments/c5_hw_v2/results/v2_per_eval_seed.json | settings.K.pooled.turns30_ratio_vs_C5_mean 0.9245 |
| 0.347; 0.180 | [§8.3] 30-s failure from random grasps, C5-HW-V2 vs C5 mean | training_source/project/experiments/c5_hw_v2/results/v2_per_eval_seed.json | settings.A.pooled.failure30 (0.3469, 0.1803) |
| 3.4%; 0.2% | [§8.3] 30-s failure from grasp 49, C5-HW-V2 vs C5 mean | training_source/project/experiments/c5_hw_v2/results/v2_per_eval_seed.json | settings.B.pooled.failure30 (0.0344, 0.0019) |
| 2.66%; eight earlier policies; 2.30% | [§8.3] kit starts lost by all 8 baseline policies (A1–A3, R1, C5_1–C5_3, C5-EFFORT); criterion 1 failure bar | training_source/project/experiments/c5_hw_v2/results/posthoc_kit_failures.json | share_failing_for_all_8_baselines 0.0266; criterion1_failure_threshold 0.0230 |
| 1280×720; about 0.40 m | [§9] camera resolution and distance | docs/realsense_setup_2026-11-25.md | §2 table (used stream); §3 "Geometry" |
| 28 fps; 91–97% | [§9] frame rate and RGB-D fusion, static cube | docs/realsense_setup_2026-11-25.md | §3 table, 1280×720 column |
| 180° | [§9] rotation of markers M and I vs `layout.json` | docs/live_vision_digital_twin.md | §2.2 Step 2 table |
| all six faces; two independent recordings | [§9] layout confirmation | docs/realsense_setup_2026-11-25.md | §3d "Faces" |
| 46.5% → 100%; 1,402 | [§9] TRACKING share and layout conflicts, before → after | docs/realsense_setup_2026-11-25.md | §3b table |
| 1% | [§9] wrong-branch probability threshold | docs/realsense_setup_2026-11-25.md | §3c "Resolved or ambiguous" |
| one turn either way | [§9] widening of ambiguous ranges | docs/realsense_setup_2026-11-25.md | §3c "Resolved or ambiguous" |
| 200 sequences of 60 s; 27% hidden; up to 4 s | [§9] synthetic stress test design | docs/realsense_setup_2026-11-25.md | §3c Validation table, synthetic row |
| 0.05; 5.9 | [§9] median error (quarter turns per 60 s), gap-aware vs old rule | docs/realsense_setup_2026-11-25.md | §3c Validation table, synthetic row |
| trial 5; 11.1 s; 2.0 quarter turns; 50°/s | [§9, Figure 5] video frame | docs/hardware_session_2026-11-25.md | Caption under the "estimated frame" image |
| Three of the five trials | [§9] trials with camera video | docs/hardware_session_2026-11-25.md | Runs table, "camera video" column |
| 40%, 26%, 31% | [§9] estimated (hidden) share of the policy window, trials 1/4/5 | docs/hardware_session_2026-11-25.md | "Camera results" table |
| 16.2 [8.2, 24.2]; 0.8; 7.0 [3.0, 11.0] | [§9] camera quarter turns, trials 1/4/5 | docs/hardware_session_2026-11-25.md; LEAP_Hand_RL_Deploy_Kit/logs/trials_20261125_vision.csv | "Camera results" table; CSV `vision_quarter_turns`, `vision_qt_lo/hi` |
| 16; 2; 8 | [§9] operator quarter turns, trials 1/4/5 | docs/hardware_session_2026-11-25.md | Runs table |
| within 1 quarter turn (trials 1 and 5) | [§9] camera–operator agreement | docs/hardware_session_2026-11-25.md | "Result", Turn counts |
| about 11 fps | [§9] live recording rate | docs/hardware_session_2026-11-25.md | "Problems found and fixed", item 3 |
| 1.36–1.53× | [§5, §10] estimator error growth on new gaits vs reference: position 1.36–1.45×, rotation 1.48–1.53× | training_source/project/experiments/hwlimit_ac/README.md | §8.4 "Finding" |
| 4–12 valid attempts | [§10] valid attempts per policy and session | docs/figures/readme/headline_numbers.json | real_hand[].attempts (min 4 model_400, max 12 C5 seed 3 on 24 Nov) |
| ±1–2 s | [§10] uncertainty of operator drop times | docs/hardware_policy_actuator_audit_2026-11-20.md | §11 "Operator counts" |
| Three (20 Nov runs, no recorded reason) | [§10] operator stops with unknown reason | docs/hardware_policy_actuator_audit_2026-11-20.md | §2 table, "Operator stop, reason unknown" total |
| about 0.1 turns | [§10] smallest resolvable difference with 3 seeds | training_source/project/experiments/hwlimit_ac/README.md | §8.6 Limitations, first bullet |

## Notes for the author

Source disagreements (the more primary source was used in REPORT.md):

1. **A/B/C/D differences.** The top-level `README.md` research log rounds the study's values ("+0.26 [+0.15, +0.38]", "+0.15 [−0.01, +0.34]"); the report uses the study README §3.3 values (+0.263 [+0.148, +0.376], +0.150 [−0.006, +0.339]). *README fixed 27 Nov: the research log now quotes the study values.*
2. **A vs C under the kit limits.** `hwlimit_ac/results/tables.md` gives −0.041 [−0.150, +0.055]; the study README and top-level README round the upper bound to +0.06; `hwlimit_ac/figures/test_comparison.png` prints [−0.15, +0.05], and for high_delay prints −0.02 where the README table has −0.03. The report uses `tables.md`. *README fixed 27 Nov: −0.041 [−0.150, +0.055].*
3. **P0 on 20 Nov.** The audit (§2) counts 4 drops, 3 guard stops, 2 unknown-reason operator stops (sensitivity: 5 / 2 / 2). `headline_numbers.json` and Figure 2 count 3 drops and 4 safety stops, because a guard stop after a drop counts as a stop. The original session report double-counted (5 drops + 4 stops + 2 discards = 11 for 9 attempts). The report's table uses the audit and the Figure 2 caption explains the difference.
4. **"Held 60 s" denominators.** Figure 3 / `README.md` give A3 50% and C5 36% (3/6 and 4/11, unknown-reason operator stops excluded); the audit's completion rates include them (A3 3/7, P0 0/9).
5. **Figure 3 footnote.** `docs/figures/readme/sim_vs_real.png` says "training seed 1 of each policy", but the A3 values (2.68 turns in 60 s, 75% held) are A seed 3 (GRV2 Phase 4: A3 turns60 2.677, P(held 60 s) 0.746). The report's caption states the seeds correctly. The footnote is corrected in `scripts/make_readme_figures.py` on branch `task7-consistency` ("A3 = A training seed 3, R1/P0/C5 = training seed 1"); the PNG changes when that branch regenerates the figures. *Fixed and regenerated 26 Nov (task7).*
6. **"R1 and P0 kept the cube best in simulation"** (`README.md`, 20 Nov session report). This is true for 30-s failure (R1 19.0%, P0 11.5%, C5 19.9%, A3 24.0%), but on P(held 60 s) C5 (0.788) is above R1 (0.763). The report states the failure-rate version. *README fixed 27 Nov: "lowest 30-s failure rates"; the Figure 3 alt text now gives all four simulated held-60-s shares.*
7. **C5-EFFORT values.** `sim_progress.png` and `README.md` show C5-EFFORT 1.62 (full distribution) and 2.16 (grasp 49) under the default 550 mA actuator (`headline_numbers.json` note); the pre-registered decision value is 1.625 under the 350 mA-equivalent actuator (`REPORT_EFFORT_V1.md`). Both are single-seed exploratory. *README fixed 27 Nov: Figure 1's footnote and alt text now say "default (550 mA) actuator".*
8. **25 Nov trial 1 camera count.** The session report and `trials_20261125_vision.csv` give 16.2 [8.2, 24.2] (CSV: 16.17 [8.17, 24.17], 2 ambiguous gaps, legacy 14.36). `docs/realsense_setup_2026-11-25.md` §3c gives 17.0 [+5, +29] with 3 ambiguous gaps and legacy 14.1, apparently an earlier pipeline iteration. The report uses the session report.
9. **C5 guard-stop times.** `SPEC.md` E1 says the jam trips came 3.6–15.5 s into the run; the audit's §8 guard-event table lists 4.7–15.6 s for the five jam stops (3.1 s for the sixth); the session report says "3–16 s". The report omits the range.
10. **C5 seed 1 index saturation under the 350 mA cap.** The audit §10 (and `SPEC.md` E5) give 7.6% (`results/sim_effort_diagnostic.json`, 0.0757); the seed screen used for the 24 Nov choice gives 7.5% (`c5_seed_screen.json`, 0.0749). Same settings, separate GPU rollouts. The report uses 7.6% in §8.1 and 7.5% in §7.3, each from its own source.
11. **24 Nov stop-rate test.** The session report gives Fisher p = 0.012 (3/5 vs 0/13); `trials_20261124_comparison.md` gives permutation p = 0.014 on the 12 valid seed-3 trials. The report uses the session report.
12. **20 Nov comparisons.** The session report rounds the comparison file (+11.9 [+9.9, +14.1]; −31 s [−52, −10], p = 0.07); the report uses `LEAP_Hand_RL_Deploy_Kit/logs/trial_comparison.md` (+11.90 [+9.85, +14.10]; −31.41 [−51.76, −9.66], p = 0.069).
13. **C5 − A p-value.** `grasp_rotation_v2/README.md` says p = 0.003; `results/c5_seeds.md` and `README.md` say 0.0025. The report uses `c5_seeds.md`.
14. **Training range of the rot joints.** The 17 Nov session report and `README.md` say ±1.05 rad (mechanical range); `hwlimit_ac/README.md` §2.1 lists the original training range (model soft limits) as −0.995…0.995. The report uses ±1.05 rad; check which is meant.
15. **Estimator degradation factor.** The `README.md` research log says "1.4–1.5× worse"; `hwlimit_ac/README.md` §8.4 gives 1.36–1.45× (position) and 1.48–1.53× (rotation). The report uses the study. *README fixed 27 Nov: 1.36–1.53×.*
16. **Sample-size range.** `README.md` says 4–13 attempts per policy and session; 13 counts the excluded intervention trial of C5 seed 3 on 24 Nov. `headline_numbers.json` gives 4–12 valid attempts. *README and report §10 fixed 27 Nov: "4–12 valid attempts".*
17. **GRV2 limitation wording.** `grasp_rotation_v2/README.md` ends with "the 0.029 m cube only"; 0.029 m is the half side of the 58 mm cube. Not used in the report, but the source wording may confuse readers.

Uncertainties:

- **No number was left out for lack of a source.** Derived counts were avoided; where a range had to be quoted, it was taken from a source that states it.
- **C5-HW-V2** (§8.3) sources are in `training_source/project/experiments/c5_hw_v2/`, committed on branch `task2-c5-hw-v2`. They are not in this worktree until that branch is merged; every §8.3 number was checked against those files.
- **Related work** descriptions are one-line characterisations based on the citation details supplied; check that the descriptions of [3], [4] and [6] match the papers.
- **Software URLs and versions** are taken from the installed packages' metadata (`importlib.metadata`, project URLs of mujoco-warp 3.8.1 and rsl-rl-lib 4.0.1 in the training environment).
- **Word count** of the body (abstract to conclusion, excluding tables, figure captions and references) is about 5,300 words; the PDF (`docs/report/build_pdf.py`) is 8 pages.
