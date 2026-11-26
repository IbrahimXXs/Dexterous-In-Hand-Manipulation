# C5-HW-V2, steps A and B: pre-registered plan (evaluation only, no training)

**Written 2026-11-26, before any rollout of this step.** Hash: `logs/fit_plan.sha256`.
Nothing in this file may be changed after the hash; deviations go into the results file as deviations.

Frozen policies only. No training, no kit file, no guard or limit, no existing checkpoint or log changes.

## A. Response fit (`c5_hw_v1/SPEC.md` §4, made explicit)

**Policy.** Frozen C5 seed 1, `grasp_rotation_v2/checkpoints/C5_1_selected_model_2400.pt` (sha256 prefix `2a932dfe37a5540d`, the checkpoint `sim_actuator_reference.py` uses). **Recorder.** `c5_hw_v1/scripts/sim_actuator_reference.py::record`, imported unchanged, task `Mjlab-Leap-RetV1-A` (C5 physics), grasp 49, 256 environments, 600 steps (30 s), seed 8401.

**Grid (24 points).** Only three event ranges change; everything else is the task's own.

| Factor | Values | Implementation |
|---|---|---|
| Action delay | 0–25, 0–50, 25–50, 25–75 ms (current) | `dr_action_delay.lag_range` = (0,5), (0,10), (5,10), (5,15) physics steps of 5 ms |
| Damping scale s_d | 0.5, 0.75, 1.0 | `dr_motor_damping.ranges` = s_d × (0.6, 1.6) |
| Friction-loss scale s_f | 0.5, 1.0 | `dr_motor_friction.ranges` = s_f × (0.5, 1.8) |

**Primary run:** training effort limit (0.262–0.321 Nm), as in SPEC §4. **Sensitivity run (reported, not used for selection):** the same 24 points with the deployed 350 mA-equivalent limit (0.167–0.204 Nm).

**Scoring.** `sim_metrics` from `LEAP_Hand_RL_Deploy_Kit/kit/audit_hardware_logs.py`, unchanged, `n_envs_lag = 32`.
- **Lag:** per-joint position-estimator lag (`lag[j].pos_s_median`).
- **Hardware lag target:** `audit/20261120/lag_summary.json["C5"]["per_joint"][j]["pos_s"]["median"]`.
- **Tracking:** per-joint median tracking error (`tracking[j].median`). One summary number = the median over the 16 joints.
- **Hardware tracking target:** `audit/20261120/per_joint.csv`, C5 rows with `category_name == completed_60s`. Take the per-joint median of `track_median` over those runs, then the median over joints.

**Selection.**
1. **Constraint:** sim tracking summary within [0.70, 1.30] × the hardware tracking summary.
2. **Objective:** among the points that meet the constraint, the smallest Σ_j |lag_sim,j − lag_hw,j|.
3. **No point meets the constraint:** the fit is declared **failed (tracking)**. The response change is dropped from C5-HW-V2, and the best-lag point is reported for information only.

**Stop rule (lag).** For the selected point, if the median over joints of |lag_sim,j − lag_hw,j| > 150 ms, the fit is declared **failed (structural)**. It is reported, and the response change is not used for training.

**Outcome criterion, reported (from SPEC §7):** the number of joints with |lag_sim − lag_hw| ≤ 30 ms.

**Randomisation around a successful fit.**
- **Delay:** centre c = midpoint of the fitted range (ms). Range = [max(0, c − 25), c + 25] ms, rounded to 5 ms physics steps.
- **Damping:** fitted value = s_d × 1.1 (the midpoint of 0.6–1.6). Range = (0.8, 1.25) × that.
- **Friction loss:** fitted value = s_f × 1.15. Range = (0.8, 1.25) × that.
- Stiffness, gains and armature are unchanged (E4: stiffness already matches).

**Cross-check (reported, does not gate).** The free-air 20 Hz tracking stage of `logs/functest_20261117_094855.npz` (stage 6, no cube).
- **Hardware:** per-joint `lag_position` after the 1 s fade-in.
- **Simulation:** the same command sequence (mapped to server order) replayed as joint targets with the cube moved off the hand, at the current setting and the selected setting (64 environments, seed 8401), with the same estimator.

## B. Kit-start resets

**Procedure (simulation, the deployed 350 mA-equivalent effort range 0.167–0.204 Nm, all other randomisation as in the task):**
1. **Hand:** joints at the kit's home pose (0 rad, clipped to the joint range), held by the motors.
2. **Cube placement:** grasp 49's cube position (`hwlimit_ac/cache/hwlimit_grasp_cache.npz`), plus 10 mm in z so it drops onto the open hand. Orientation is flat (roll = pitch = 0) at grasp 49's yaw. Placement noise: uniform ±4 mm in x and y, uniform ±15° yaw.
3. **Settle:** 1.0 s under gravity, with the motors holding the home pose.
4. **Closing ramp:** 5.0 s linear ramp of the joint targets from home to the policy default pose (the kit's `INIT_DURATION_S = 5`), one target update per 50 ms control step.
5. **Record** the measured joint positions and the cube pose at the end of the ramp. The joint target stored with the start is the **measured** pose: the kit resets its previous command to the measured post-ramp pose (`deploy_direct.py`, mode `kit_reset_measured`).
6. **Validity filter:** cube centre within 25 mm (xy) of grasp 49's position and no lower than 15 mm below it, and ≥ 2 fingertips in contact. The rejection fraction is reported.
- **Seeds and size:** generation seed 8501, 4096 environments.
- **Output:** `cache/kit_start_cache.npz`, in the grasp-cache format, cube size = grasp 49's.

**Validation (reported, does not gate):** the sim post-ramp joint pose (median over starts) against the hardware post-ramp pose. Hardware: `q_hw[0]` of every C5-family run log (C5, C5s2, C5s3) in `LEAP_Hand_RL_Deploy_Kit/logs`; per-joint median, in server order.

**Survival report.**
- **Policies:** frozen C5 seeds 1, 2, 3 and C5-EFFORT seed 1.
- **Evaluation:** `c5_hw_v1/scripts/eval_items.py::rollout` and `summarise`, imported unchanged; the kit-start cache is the reset source. 2048 environments, 600 steps, seed 7001 (validation seed: the held-out seeds 6201/6202 stay unused until the final evaluation).
- **Actuators:** training default and 350 mA effort.
- **Reported:** 30-s survival (`completed`), turns, failure, jam proxy. Also the same policies from the standard grasp-49 start, same seed, for contrast.

**Stop rule (B).** If 30-s survival from kit starts is < 10% for both C5 seed 1 and C5-EFFORT under the 350 mA effort actuator ("nearly every start fails"), stop and propose a 50/50 mix of kit starts and the grasp cache to the user before writing the protocol.
