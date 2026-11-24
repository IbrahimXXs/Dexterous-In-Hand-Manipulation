# C5-HW-EFFORT-V1: single-seed protocol (frozen 2026-11-24, before training)

This addendum to `SPEC.md` fixes how the one C5-HW-effort run is trained, evaluated and judged. It was written and hashed (`logs/frozen.sha256`) before the run started and before any result of it existed.

## Scope (instruction of 2026-11-24)

- **Exactly one training run:** C5-HW-effort with PPO seed 1. Seeds 2 and 3 are not trained, and neither are C5-HW-delay or combined.
- This is a **single-seed exploratory experiment**. It cannot show training-seed reproducibility.
- Confidence intervals below are bootstraps over evaluation environments. They describe evaluation variation for this one trained model, not variation across training runs.

## Condition

The task is `Mjlab-Leap-C5HW-EFFORT`. It is GRV2 C5 with exactly two changes, checked by `tests/test_c5_hw_v1.py` through a full config-tree diff:

1. **Effort limit.** The reset-time effort-limit randomisation is 0.53 Nm/A × 0.35 A × U(0.9, 1.1) = 0.167–0.204 Nm, drawn per environment and per actuator. C5 uses 0.262–0.321 Nm, which is 550 mA. This is the simulator's own mapping. Its reset event overwrites the per-joint effort scales of the actuator config with this uniform per-actuator draw, so the per-actuator variation is the ±10% draw.
2. **Penalty.** The new reward `actuator_utilisation_penalty` is −w Σ_j [max(0, u_j − 0.8)/0.2]², with u_j = |τ_j| / L_j.
   - **w = 0.2201**, at constant weight with no curriculum.
   - It was calibrated once, by `scripts/calibrate_effort_weight.py` → `results/effort_weight_calibration.json`.
   - On frozen C5_1 rollouts, capped, from grasp 49 with seed 8301, the mean is p = 0.2284 per step. The mean rotation reward is 0.5026 per step, so w = 0.1 × 0.5026 / 0.2284.

Training uses 4096 environments, 3001 updates, a zero-initialised actor output and the hwlimit A PPO settings, starting from scratch (`scripts/sbatch_train.sh`).

## Validation, selection and stage check

- **Validation:** `scripts/curves.py`, a copy of the GRV2 validation. It uses seed 7001, nominal preset, 1024 environments, 30 s, a point every 100 updates and a 60-s point every 500. Validation runs in the run's own training task, as C5's did in its own.
- **Selection:** max 30-s validation turns; ties go to the later checkpoint.
- **Stage check at update 500.** Stop only for:
  - a validation 30-s yaw rate below 50% of C5's at update 500. The C5 3-seed mean there is 0.264 rad/s, so the threshold is 0.132 rad/s. For reference, C5_1 alone is 0.239.
  - a reward exploit: the penalty is avoided by not rotating (validation turns ≤ 0.1), or the hand stops moving;
  - NaN or numerical instability, or a collapse of the training reward.

A stopped or failed run is recorded, not removed.

## Held-out evaluation

The evaluation follows GRV2 Phase 4. Rollouts are 60 s (1200 steps), and 30-s metrics come from the same rollouts. The first episode only is used. The metric definitions are those of `grasp_rotation_v2/rollout.summary_per_env`.

**Policies (all selected checkpoints):**
- A1, A2, A3 (the A mean is the reference);
- R1, for context;
- C5_1, C5_2, C5_3 (the original C5 baseline is the mean of the three seeds; each seed is also reported);
- EFFORT_1.

**Settings:**

| Setting | Seeds | Environments | Presets |
|---|---|---|---|
| A, full 239-grasp distribution (**primary**) | 6201, 6202 | 2048 each | nominal on both; heavy_slippery, high_delay, obs_noise_x3 on 6201 |
| B, grasp 49 | 6201, 6202 | 2048 each | nominal |
| C, robust 10-grasp subset | 6201, 6202 | 2048 each | nominal |

**Actuators:** every rollout is run under two actuator models. Rewards play no part in a rollout.
- **default:** the training actuator, task `Mjlab-Leap-RetV1-A`.
- **effort-matched:** task `Mjlab-Leap-C5HW-EFFORT`. Dynamics, observations, actions and randomisation are identical to the default except for the 0.167–0.204 Nm effort range.

**Not available:** SPEC's "hardware-matched" actuator is the 350 mA limit plus a *fitted response* (SPEC Section 4). The response fit is not done in this milestone, so it is not used. No arbitrary response setting is substituted. The decision actuator is therefore the **effort-matched** model, and it is labelled as such. Response matching (80 ms hardware vs ~280 ms simulation) remains unverified.

## Effort and jam metrics

These follow SPEC Section 6 and are recorded per 20 Hz control step for every environment while it is alive in its first episode. Effort samples are taken at reward time, the same instant the penalty sees.

- **Utilisation:** u_j = |τ_j| / L_j. Reported as the mean, p95 and p99 over joint-samples, plus per-joint means. "Saturated" means u ≥ 0.98.
- **Saturation fractions:** per joint, overall, and for index MCP, index PIP and thumb abduction (`th_axl`).
- **Blocked (hardware-aligned):** index MCP or PIP saturated, and |command applied during the step − joint position at the end of the step| ≥ 0.2 rad. The hardware equivalent is current ≥ 340 mA with |q[k] − cmd[k−1]| ≥ 0.2 rad.
- **Jam proxy (primary safety metric):** the share of environments with at least one index-jam episode of 1 s or more. Episodes come from the audit's own detector (`LEAP_Hand_RL_Deploy_Kit/kit/audit_hardware_logs.py: jam_episodes`): in a centred 1-s window, one joint (here only index MCP or PIP) is blocked in at least 50% of samples; gaps under 0.5 s are merged and episodes under 0.5 s dropped. The detector runs per environment over its alive samples of the 60-s rollout, and the proxy counts episodes with duration ≥ 1.0 s. The 30-s version is also reported.
- **Index blocked fraction:** the share of alive samples in which index MCP or PIP is blocked.
- **Other:** joint speed, tracking error |cmd − q|, command-limit occupancy, and turns conditional on surviving 30 s and 60 s.

## Decision (SPEC Section 7, unchanged thresholds)

The comparison uses Setting A (full distribution), nominal preset, seeds 6201 and 6202 pooled, under the **effort-matched** actuator. EFFORT_1 is compared with the **C5 3-seed mean**.

**Success requires all three:**

1. **Jam proxy:** reduced by at least 50% relative to C5, and the 95% CI of the difference excludes 0.
2. **Rotation:** turns per 30 s at least 0.90 × C5 and at least 1.20 × the A mean.
3. **Failure:** failure rate ≤ C5's + 0.03.

**Reject if any one holds:**
- turns per 30 s < 0.80 × C5;
- jam-proxy reduction < 25%;
- failure > C5's + 0.08.

**Otherwise the result is inconclusive.** Inconclusive includes the case where the reduction reaches 50% but the CI does not exclude 0.

**How the CI is computed.** SPEC asks for seed-level Welch tests, which one training seed cannot support. The CI for criterion 1 is therefore a paired bootstrap over evaluation environments: 10,000 resamples of environment indices, the same for all policies, since equal seeds give the same reset draws. It quantifies evaluation variation only.

**The strongest possible verdict** is "single-seed pass (exploratory)", never "confirmed". A pass is also reported against each C5 seed separately, descriptively and not as an extra criterion.

## Deployment readiness gate

A kit-compatible candidate is packaged only after a single-seed pass of all three success criteria, with no rejection criterion triggered.

The package goes in a new directory. It must pass the GRV2 export and verification checks, run offline only, without changing the kit. Even then the candidate is only "ready for supervisor review". Simulation results are not hardware-safety evidence.
