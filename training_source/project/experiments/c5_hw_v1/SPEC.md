# C5-HW-V1: proposed simulation experiment (specification only, nothing trained)

**Status:** proposal for review. No policy has been trained and no training configuration has been changed. The original C5 code, configuration and checkpoints are untouched. The evidence is in the hardware audit `docs/hardware_policy_actuator_audit_2026-11-20.md`. Two simulation evaluations of the frozen checkpoints (no training) support it:
- `sim_reference/` and `scripts/sim_actuator_reference.py`;
- `results/sim_effort_diagnostic.json` and `scripts/sim_effort_diagnostic.py`.

**Aim:** a C5 successor that keeps C5's rotation speed on the real hand and stops tripping the current guard. The design is simulation only, inside the approved joint limits and the runtime's ±1/24 rad command step. The kit's guards and thresholds stay as they are.

## 1. What the hardware data say

All findings are from the 37 logs of 20 Nov 2026.

| # | Finding | Strength |
|---|---|---|
| E1 | 5 of C5's 6 guard stops follow a **sustained index jam**: index MCP and/or PIP held at the 350 mA limit, 0.2–0.6 rad short of its command, unresolved until a 453–747 mA spike trips the 450 mA guard (3.6–15.5 s into the run). The sixth stop (3.1 s) shows the same index + thumb-abduction load before a jam had formed. | Strong |
| E2 | **Every** C5 run loads the index heavily in the first 2–4 s, the completed ones included. Completed runs clear it by about 3.5 s; guard-stop runs never do. Nothing in the first 2 s separates the two groups. | Strong (n = 4 vs 6) |
| E3 | The motor current is a proportional controller clipped at the limit. Quasi-static current ≈ 1,450–1,500 mA/rad × position error (r ≈ 0.99), so the real actuator saturates at a **0.23–0.27 rad** error. | Strong |
| E4 | The training actuator is current-mapped (`leap_right_constants.py`). Its gain equals the hardware's (if_mcp: 0.763 Nm/rad ÷ 0.53 Nm/A ≈ 1,440 mA/rad; hardware 1,493 mA/rad). Its **current limit is 550 mA** (effort 0.262–0.321 Nm ≈ 495–605 mA), against **350 mA** on the hand. In simulation C5 never saturates (≤ 0.05% of samples). | Strong (configuration fact) |
| E5 | With the effort limit set to the deployed 350 mA (0.167–0.204 Nm, frozen checkpoints): **C5 saturates the index MCP/PIP 7.6% of the time** (A3 and R1 0.1%; P0 3.5%), in 79% of environments during the first 2 s. The joint most often in sustained saturation is if_mcp, as on the hand. Rotation barely changes (2.44 vs 2.46 turns per 30 s). | Strong qualitative replication; simulation does not reproduce the *persistence* of the hardware jam |
| E6 | **Effective response lag**, same estimator for both: hardware **80 ms** (IQR 75–85, all policies, all joints 60–100 ms); simulation **~280 ms** (joint medians 160–600 ms). The hand tracks about 3.5× more tightly (median error 0.03 vs 0.10 rad), and C5's real joints move 1.85× faster than in simulation (0.53 vs 0.29 rad/s). | Strong; the premise "the hand is slower than training (100 ms vs 25–75 ms)" compares different quantities and does not hold |
| E7 | Abrupt or fast commands do **not** precede the trips. In the same time windows, guard-stop and completed C5 runs command and move at the same speed before the jam (0–2 s: \|qd\| 0.42–0.47 vs 0.40–0.46 rad/s; full-speed command steps 30–37% vs 28–34%). Completed runs are faster afterwards. Command steps are capped at 1/24 rad, so the 0.05 rad step guard can never trip. | Strong |
| E8 | Loop timing is not a factor. Median interval 50.1 ms; 4–7% of intervals > 60 ms, almost all a single late cycle followed by a catch-up cycle; no excess before stops; no difference between C5 groups. | Strong |
| E9 | 4 guard stops of A3, R1 and P0 are **start transients**: within 0.7–2.9 s, logged currents ≤ 331 mA jump to 549–1,048 mA within one 50 ms period. They differ from C5's jams. | Medium (the trip sample itself is never logged) |

## 2. Hypotheses

- **H-E (effort mismatch, primary).** C5 learned to rely on index forces that are available in simulation but not at the deployed 350 mA. On the hand, the index then stalls against the cube (E1–E5). Train with the deployed effort limit and a smooth cost on near-limit actuator use, and the policy should find a gait that does not force the index: fewer jams, little loss of speed. Basis: in E5, rotation is almost unaffected by the cap.
- **H-R (response mismatch, secondary).** The simulated actuator responds about 3.5× more slowly than the real one (E6), so the policy's commands produce faster, tighter motion on the hand than it learned with. Match the simulated response to the hardware's effective lag and tracking error, and the jam-prone early transient (E2) should shrink. This is "C5-HW-delay", **re-framed**. The data argue against *lengthening* the delay. The target is the measured 80 ms effective lag, not a fixed 100 ms pure delay.
- **Not pursued:** a smoothness or command-change penalty (E7), timing changes (E8), a lower yaw clip (keep ±0.5 rad/s).

## 3. Conditions (each a single, separately controlled change from C5)

| Condition | Change from C5 (everything else identical: obs 320-D, actor, actions, 20 Hz, joint limits, ±1/24 rad step, rewards incl. yaw clip ±0.5, PPO hyperparameters, 4096 envs, 3001 updates, grasp cache, other domain randomisation) |
|---|---|
| **C5 baseline** | The existing C5 (seeds 1–3, `grasp_rotation_v2/checkpoints/C5_{1,2,3}_*`). Not retrained. |
| **C5-HW-effort** | (a) `dr_effort_limits` 0.262–0.321 Nm → **0.167–0.204 Nm** (= 0.53 Nm/A × 0.35 A ± 10%, in the simulator's own current mapping); (b) new reward `actuator_utilisation_penalty = −w Σ_j [max(0, u_j − 0.8)/0.2]²`, with u_j = \|τ_j\| / effort limit_j (0 below 80% of the limit, 1 at it; smooth, per joint, all 16 joints). |
| **C5-HW-delay** (response-matched) | Actuator response parameters (`dr_action_delay` range, damping and friction-loss scales) **refit** so that the simulated effective lag and tracking error match the hardware (Section 4). Effort and rewards unchanged. |
| **C5-HW-combined** | Both. Trained **only if** each single change passes its own criteria (Section 7). |

**Diagnostic ablation**, one seed, run only if C5-HW-effort is promising: **effort-limit-only** (a) without (b). It separates the cap from the penalty.

### Calibrating w (before training, recorded)

On frozen C5 rollouts with the capped actuator, the hinge term averages 0.228 per step, summed over joints (`results/sim_effort_diagnostic.json`, seed 8301).

Rule: w is set so that the mean penalty equals **10% of C5's mean rotation reward per step** on those rollouts. That is the share at which the retention terms were calibrated in Retention-V1/GRV2. Compute it once, write it into the protocol, and do not tune it on validation.

## 4. Response fit for C5-HW-delay (evaluation only, before any training)

1. **Target, from `audit/20261120/lag_summary.json` and `per_joint.csv`:**
   - per joint, effective lag 60–100 ms (median 80 ms);
   - median tracking error 0.004–0.04 rad (C5 completed runs);
   - estimator: `lag_position` / `lag_velocity` in `kit/audit_hardware_logs.py`.
2. **Grid:**
   - delay range ∈ {0–25, 0–50, 25–50, 25–75 (current)} ms;
   - damping scale ∈ {0.5, 0.75, 1.0} × current;
   - friction-loss scale ∈ {0.5, 1.0} × current.

   Evaluate the frozen C5 with `sim_actuator_reference.py`, grasp 49, 256 envs, seed 8401. Score by `sim_metrics` from the audit tool: the same code as for the hardware.
3. **Select:** the setting with the smallest summed per-joint |lag_sim − lag_hw|, subject to median tracking error within ±30% of the hardware. Randomise around it:
   - delay ±25 ms;
   - damping and friction at ×0.8–1.25 of the fitted value.

   If the stiffness range already matches (E4), stiffness is not changed.
4. **Cross-check:** the 17 Nov free-air function-test logs (`logs/functest_*.npz`). They have no cube contact, so they test the actuator alone.
5. **Stop rule:** if no grid point brings the simulated lag within 150 ms of the hardware, the actuator model needs a structural change, not a parameter fit. Report that instead of training C5-HW-delay.

## 5. Training plan

- Three seeds per condition (1, 2, 3), from scratch, never initialised from C5.
- Stage check at update 500: yaw rate ≥ 50% of C5's at 500, no exploit, stable.
- Selection by maximum validation turns (seed 7001), as for C5.
- Run order: C5-HW-effort and C5-HW-delay in parallel (2 ws-ia slots); combined afterwards, only if justified.
- About 4 h per run.

## 6. Evaluation

Held-out seeds 6201/6202, both settings of GRV2 (full grasp distribution; grasp 49 and the robust subset). Every condition, the C5 baseline included, is evaluated under **two actuator models**: the training default, and the **hardware-matched** model (350 mA-equivalent limit plus the fitted response from Section 4). That way no condition is judged only in its own training environment.

| Metric | Definition | Why |
|---|---|---|
| Rotation | 30-s and 60-s turns; yaw rate held | Speed must be kept |
| Retention | failure rate; P(held 60 s) | As in GRV2 |
| **Jam proxy** (primary safety metric) | Share of environments with ≥ 1 index-jam episode ≥ 1 s (index MCP/PIP at ≥ 98% of the effort limit **and** ≥ 0.2 rad from its target, in ≥ 50% of a 1-s window: the audit's definition); index blocked fraction | The hardware failure signature (E1) |
| Utilisation | Share of joint-samples at ≥ 98% of the limit; p99 of u_j | Effort headroom |
| Response | Effective lag and tracking error (audit estimator) | Checks the H-R fit |
| Kit start | The same metrics from the kit's measured start pose (`rehearse_deploy.py`, mode `kit_reset_measured`) | The jam forms in the start transient (E2) |

## 7. Pre-registered success and rejection criteria

Comparisons are against the **C5 baseline, 3 seeds each**, held-out seeds, **hardware-matched evaluation actuator**, seed-level Welch tests.

**C5-HW-effort succeeds if all of these hold:**
1. Jam proxy reduced by ≥ 50% relative to C5, with the 95% CI of the difference excluding 0.
2. Turns per 30 s (full distribution) ≥ 0.90 × C5 **and** ≥ 1.20 × the A mean.
3. Failure ≤ C5's + 0.03.

**C5-HW-delay succeeds if:**
1. The fitted simulation lag is within 30 ms of the hardware on ≥ 12 of 16 joints.
2. Criteria 2 and 3 above hold.
3. The jam proxy is ≤ C5's.

**Reject a condition if:** turns per 30 s < 0.80 × C5; **or** jam-proxy reduction < 25%, for effort; **or** failure > C5's + 0.08.

**Combined** is trained only if both single conditions succeed, or one succeeds and the other is non-inferior on rotation.

**Hardware confirmation** (a separate, supervised, approved session; not part of this experiment):
- C5 vs the best C5-HW condition, ≥ 10 attempts each, same placement and kit settings;
- primary metric: guard-stop rate (C5 on 20 Nov: 6/11);
- secondary: quarter-turns per trial, reported as full revolutions = quarter-turns / 4; completions.

## 8. Constraints honoured

- Joint limits, the command step, gains, current limit and guard thresholds of the kit: unchanged.
- The effort change *lowers* the simulated limit to the physical one.
- No privileged observations; actor and observation unchanged.
- No estimator work; no camera work.
- C5 code, configuration and checkpoints unchanged.
- New code would live in new task configs (e.g. `Mjlab-Leap-C5HW-{effort,delay}`) and a new reward function, with tests that each differs from C5 by exactly its one change.
