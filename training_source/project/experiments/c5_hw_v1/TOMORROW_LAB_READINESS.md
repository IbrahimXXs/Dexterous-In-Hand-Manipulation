# Lab readiness: C5-HW-EFFORT-V1 candidate (prepared 2026-11-24)

> **Superseded (note added 2026-11-26).** The next hardware session is planned in `docs/hardware_protocol_final_DRAFT.md` (DRAFT, not approved). Two kit hashes recorded below are **stale**: the lab changed `leap.sh` and `configs/hand.env` on 24/25 Nov (camera integration; commits `b2e0cd7`, `7bcbd59`). At commit `8bdcbee` they are `77dca20769710205…` (`leap.sh`) and `68cc0663e9cb200f…` (`hand.env`). The guard and limit values in `hand.env` did not change. The 24 Nov hardware result for C5 seed 3 (0 safety stops in 13 attempts, `docs/hardware_session_2026-11-24.md`) also postdates this note. The rest of the document is kept as written.

> **This document does not authorise any powered test.** Simulation results are not hardware-safety evidence. Any run with torque enabled needs explicit approval from the lab supervisor and must be supervised. Nothing here was, or will be, started automatically.

## Recommendation

**GO for supervisor review. The candidate may be *proposed* for a supervised, shadow-first comparison against C5.** It is **not** approved for hardware use by this document.

| Question | Answer |
|---|---|
| Did the candidate pass the pre-registered simulation criteria? | **Yes: a single-seed pass (exploratory, not confirmed).** Jam proxy −96.5% vs C5; turns 0.911 × C5 (bar 0.90) and 1.36 × A; failure 0.178 vs C5 0.180. See `REPORT_EFFORT_V1.md`. |
| Is it packaged and verified offline? | **Yes.** It is in kit format and loads in the kit runtime. The runtime matches the training actor (max \|Δ\| 2.3e-5), the export is bit-exact, and kit inspection reports OK. The closed-loop simulation rehearsal (kit runtime in the loop) is within its CI of the training actor in all three modes (below). |
| Is it hardware-safe? | **Unknown.** Simulation cannot show it (see "Why a simulation pass is not enough"). |
| Should it be tested only because a session is booked? | **No.** It qualifies on its pre-registered result, not on the booking. If the supervisor has doubts, the right action is not to run it. |
| What would turn this into a NO-GO? | Any of: the supervisor declines; a hash check below fails; the kit safety settings differ from the recorded values; the shadow run fails or looks abnormal; the bench is not in the state used on 20 Nov. |

Read the "pass" with three caveats:
1. **Rotation clears its bar narrowly.** It is at 0.911 × C5, and below 0.90 against C5_1 (the policy run on 20 Nov: 0.878), on evaluation seed 6202 (0.899), and in the heavy_slippery and high_delay presets. Expect it to be roughly 10% slower than C5 on the hand.
2. **There is one training seed.** Whether a retrained seed behaves the same is unknown.
3. **The simulated jam proxy is not validated against hardware.** In simulation it does not separate A from C5, yet on 20 Nov only C5 jammed.

## Why a simulation pass is not enough

- **Response mismatch.** The simulated actuator lags about 280 ms; the hand lags about 80 ms and tracks about 3.5× more tightly (SPEC E6). This was **not** fitted or corrected here.
- **Persistence.** The hardware's persistent index jam and its 453–747 mA spikes are not reproduced in simulation (SPEC E5).
- **Torque constant.** 350 mA ↔ 0.1855 Nm assumes Kt = 0.53 Nm/A; this was not measured on this hand.
- **Limits gap.** The kit's MCP/PIP limits differ from training by up to 0.042 rad. C5 has the same gap, which is unchanged.
- **History.** C5 had **6 guard stops in 11 runs** on 20 Nov, 5 of them sustained index jams. The candidate was trained to avoid near-limit effort, so in simulation its p99 torque demand is about 275 mA-eq. against C5's 350. Whether that carries over to the hand is exactly what a supervised test would have to show.

## What is prepared (offline only)

| Item | Location | sha256 |
|---|---|---|
| Candidate policy folder | `training_source/project/experiments/c5_hw_v1/deploy/policies/C5HW_EFFORT_V1_seed1/` | — |
| Checkpoint `model_2900.pt` | in the folder | `d1d46f0558eb958d78bfb3e84e9a0575659d7550df1feb3fe7aeb4c3c41a4469` |
| `params/env.yaml` | in the folder | `14cd28d04d323eeda3f74fda2b6935e55c1ac2de666b6f5084644997a1627b41` |
| `params/agent.yaml` | in the folder | `3f1fe0d598efdd7635625f17fc06fef1291a51c5ee937fd66dd9fb1a55d247f3` |
| NumPy export (reference only, not needed to run) | `.../c5_hw_v1/deploy/exports/C5HW_EFFORT_V1_seed1_actor.npz` | `d3deae0882ce8228413fb6b05c3f7c8477321e58721325cd3715133b170f2c29` |
| Manifest | `.../c5_hw_v1/deploy/MANIFEST.txt` | — |

**Nothing in the deployment kit was changed.** The candidate is **not** copied into `LEAP_Hand_RL_Deploy_Kit/policies/` and is **not** added to the trial scheduler (`kit/trials.py` accepts only A3, C5, R1, P0 and M400). Both would be kit changes and need the supervisor's decision.

### Kit state to verify at the bench (recorded 2026-11-24; must be unchanged)

| File | sha256 |
|---|---|
| `LEAP_Hand_RL_Deploy_Kit/configs/hand.env` | `323eb5fd5ea1e0a50e53308ae3a9380f598869d18a5d63f8788fafb2cc26a252` |
| `LEAP_Hand_RL_Deploy_Kit/leap.sh` | `dc606948e3ec547882a76f106484a503a4bd7ade155e3a8cfd22065fb417c79e` |
| `LEAP_Hand_RL_Deploy_Kit/kit/policy_runtime.py` | `35c35b18154094f519ef4532e42ef2ffa689df6ae556532f603e77ffe98faac3` |

The safety values in `hand.env` stay exactly as they are:
- `CURRENT_LIMIT_MA=350`, `CURRENT_ABORT_MA=450`;
- `TRACKING_ERROR_ABORT_RAD=0.65`, `VELOCITY_ABORT_RAD_S=8.0`, `TEMPERATURE_ABORT_C=60`;
- `MAX_COMMAND_STEP_RAD=0.05`, `LEAP_GAIN_PROFILE=H1`.

Do not change them, do not bypass a guard, and do not re-run a guard-stopped trial with altered settings.

## Proposed supervised session (only after supervisor approval)

This is the SPEC's hardware confirmation, proposed and not scheduled. The supervisor decides whether to run it and may shorten or stop it at any time.

**0. Before power-on (no motion)**
1. The supervisor approves the plan in writing and is present.
2. Check the three kit hashes above and the four candidate hashes (`sha256sum`). Check that `git status` shows no kit changes.
3. Run `./leap.sh inspect <candidate folder>`. It is offline and must print "OK".

**1. Shadow run (torque OFF)**
- Run `./leap.sh shadow <candidate folder>`, with the policy on live encoders.
- The kit only unlocks `run` for this exact policy hash after a passed shadow run. That gate stays in place.

**2. Comparison (powered, supervised)**
- Alternate C5 and the candidate (ABAB…), with the same cube, the same placement and the same kit settings.
- Aim for ≥ 10 attempts each, with C5 started through its existing kit folder.
- Run the candidate with `./leap.sh live run <candidate folder>`. The operator notes the turns, and the log goes to `LEAP_Hand_RL_Deploy_Kit/logs/physical_*.npz`.
- Primary metric: the guard-stop rate (C5 on 20 Nov: 6/11). Secondary: quarter-turns per trial (report revolutions = quarter-turns / 4), completions, and index current behaviour.

**3. Stopping rules (proposed; the supervisor may tighten them)**
- Every guard stop is handled by the kit as designed. **No override, no restart with changed settings.**
- Stop the candidate's arm of the session if:
  - it has **3 guard stops in its first 5 attempts** (at least C5's rate);
  - any temperature warning appears;
  - the supervisor sees abnormal motion, sound or contact.
- Stop the whole session on any hardware fault.

**4. Afterwards (offline)**
- Run `kit/audit_hardware_logs.py` on the new logs, using the same jam detector as the simulation proxy.
- Compare the hardware jam and guard-stop rates with C5's from 20 Nov.

**Statistical power.** 10 vs 10 runs is exploratory. Even a drop from 6/11 to 1/10 guard stops is borderline significant (Fisher exact p ≈ 0.06). A favourable result would justify further seeds and runs, not a conclusion.

## Status: done vs not done

| Done (simulation and offline) | Not done |
|---|---|
| One policy trained (seed 1), stage check passed, checkpoint frozen | Seeds 2 and 3 (not requested) |
| Held-out evaluation, 144 rollouts; pre-registered verdict: single-seed pass | Response fit; a hardware-matched response actuator |
| Kit-format package; runtime and export equivalence; kit inspection OK | Any hardware run; any kit change |
| Closed-loop simulation rehearsal: 3 modes, all within CI (below) | Supervisor approval |
| Tests: c5_hw_v1 6/6; root 37/37; training suite 87/88 (the intermittent, pre-existing `test_actor_observation_layout`); preservation all OK | — |

### Closed-loop simulation rehearsal

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
