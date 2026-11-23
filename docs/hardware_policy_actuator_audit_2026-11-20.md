# Hardware policy and actuator audit: 20 Nov 2026 session

Offline analysis of all 37 run logs (`LEAP_Hand_RL_Deploy_Kit/logs/physical_20261120_*.npz`) and the operator record (`logs/trials.csv`). The hand was not used. Raw logs, CSV, kit code, guard settings and all checkpoints are unchanged.

Reproduce:

```bash
cd LEAP_Hand_RL_Deploy_Kit
python kit/audit_hardware_logs.py --date 20261120 --out audit/20261120 \
  --sim-dir ../training_source/project/experiments/c5_hw_v1/sim_reference
```

Tests: `pytest tests/test_audit_hardware_logs.py` (30 tests).

Out of scope, as requested: estimator B and the force estimator (the `mon_*` fields are ignored), and camera or video work.

## Executive summary

**Why C5 rotates well but trips the guard.** C5's guard stops are not caused by speed, abrupt commands, loop timing or lag. They are **index-finger jams**:
- Index MCP and/or PIP are held at the 350 mA current limit, 0.2–0.6 rad short of their command, for seconds.
- The policy keeps running its rhythmic gait into the blocked finger.
- A transient of 453–747 mA then trips the 450 mA guard.

| Question | Answer | Strength |
|---|---|---|
| Current, tracking, speed, timing or a combination? | **Current saturation and tracking error together. They are the same event:** on this hand the current is a proportional controller of the position error, clipped at 350 mA. **Not** speed, abrupt commands or timing. | Strong |
| Which joints and fingers? | **Index MCP (if_mcp) and PIP (if_pip)**, with **thumb abduction (th_axl)** as the counter-loaded joint. In C5's guard stops the index carries 43% of the total current, against 19% in completed runs. | Strong |
| Do successful and failed C5 runs differ? | **Yes, but only after about 2–4 s.** Every C5 run loads the index early. Successful runs clear it by about 3.5 s; failed runs never do. Nothing in the first 2 s predicts which way a run goes. | Strong (n = 4 vs 6); the cause of the split is undetermined |
| Why can simulation not catch this? | Training assumed a **550 mA** actuator current limit; the hand ran at **350 mA**. In simulation C5 never saturates. With the effort limit set to 350 mA (frozen checkpoints), **C5 alone saturates the index**, the same joint as on the hand. | Strong |
| Actuator lag? | **80 ms** effective lag on the hand (IQR 75–85 ms, all policies). The **same estimator on simulation gives ~280 ms.** The real hand responds faster and tracks 3.5× more tightly than the simulator. Lengthening the simulated delay would widen the gap. | Strong |
| Corrected counts | 37 attempts: 7 completed, 16 drops, 11 guard stops, 3 operator stops with unknown reason (mutually exclusive). C5: 11 attempts = 4 completed, 1 drop, 6 guard stops. The earlier P0 row double-counted two trials. | Exact |
| C5 speed in full revolutions | 13.9 recorded quarter-turns per trial = **3.48 full revolutions** per 60 s trial (3.64 full rev/min while held). | Operator counts |

**Recommendation for C5-HW-V1** (`training_source/project/experiments/c5_hw_v1/SPEC.md`):
- **Primary (C5-HW-effort):** train C5 with the effort limit at the deployed 350 mA plus a smooth near-limit utilisation penalty.
- **Secondary (C5-HW-delay):** refit the actuator response so the simulated lag matches the measured 80 ms. This reframes the condition: do not lengthen the delay.
- **Do not add a smoothness penalty, and keep the ±0.5 rad/s yaw clip.**

## 1. Signal semantics, established from the code

Sources: `kit/deploy_direct.py`, `leap.sh`, `configs/hand.env`.

| Field | Meaning |
|---|---|
| Record k | One per 20 Hz policy step. The state (q, qd, current) is **read at loop start**. The guards check that read. The policy then runs and `command_hw[k]` is **written**. A read that trips a guard raises before its record is appended, so **the tripping sample is never logged**. Logged peaks (max 428 mA) are therefore below the trip values (452–1,048 mA), and so are the CSV's `max_current_ma`. |
| `elapsed_s` | Monotonic seconds since the bridge started (includes the 5 s placement delay and the 5 s ramp), taken **after the write**. `loop_compute_s` = read → write duration. Read time = `elapsed_s − loop_compute_s`. Analyses use policy time = read time − first read time. |
| `q_hw`, `command_hw` | rad, **hardware order** `[if_rot, if_mcp, if_pip, if_dip, mf_…, rf_…, th_cmc, th_axl, th_mcp, th_ipl]`. Server/training order swaps rot and mcp of the three fingers (`_HW_TO_SERVER = [1,0,2,3,5,4,…]`). `rot` = side joint (±0.52 rad); `mcp` = knuckle flexion. `command_hw` is the **goal actually written**: the policy's ±1/24 rad delta step, clipped to the joint limits minus 0.01 rad. |
| `qd_hw` | Dynamixel present velocity × 0.229 rpm → rad/s. Agrees with the finite difference of `q_hw` (r = 0.95–0.98 per run). |
| `current_ma` | Signed present-current register, treated by the kit as mA. Not converted to force here. |
| `temperature_c` | Read every 20th step; NaN otherwise. This is by design, not missing data. |
| `tracking_error_rad` | **Scalar**: max over joints of \|q_hw[k] − command_hw[k−1]\|. Recomputed per joint here; the recomputation reproduces the logged scalar exactly (max difference 0.0 in all 37 logs). At k = 0 the previous command is the post-ramp pose, which is not logged. |
| `stop_reason` / `completed` | `max_steps` = full 60 s; `interrupted` = Ctrl+C; `error: <guard message>`. One log (09:46, C5) predates the session patch that stored the message and says only `error`. |
| Guards (hand.env, unchanged) | Current limit 350 mA (motor register); abort if any \|I\| > 450 mA; tracking error > 0.65 rad after step 10; velocity > 8 rad/s; 60 °C; command step > 0.05 rad. Stall recovery was off. |

Two limits of the logs:
- **The step command can never trip the 0.05 rad guard.** The policy runtime limits each step to 1/24 rad = 0.0417 rad, which is the maximum observed.
- **Runs that trip a guard during the 5 s ramp write no log at all.** Such attempts would be invisible in this dataset.

## 2. Attempt inventory and reconciliation

All 37 files have readable metadata. Policy is identified from `run_meta_json` (policy folder plus checkpoint sha256), not from file order. The sha256 matches the CSV for all 24 recorded trials. There are 13 files without a CSV row, no CSV row without a file, and no duplicates.

Categories (mutually exclusive):
1. **Completed:** 60 s, no drop reported.
2. **Dropped:** the operator reported a drop. This includes runs that continued, or that the operator stopped, after the drop.
3. **Guard stop.**
4. **Operator stop, documented reason:** none.
5. **Operator stop, reason unknown.**
6. **Invalid:** none.

Rule for drop and guard together: if the reported drop time is more than 2 s before the stop, the attempt is a drop (flag `guard_after_operator_drop`). If the two are within 2 s, it is a guard stop with low confidence (`drop_coincident_with_guard`).

| Policy | Attempts | Completed 60 s | Dropped | Guard stop | Operator stop, reason unknown | Any guard event | Completion rate |
|---|---|---|---|---|---|---|---|
| A3 | 7 | 3 | 2 | 1 | 1 | 1 | 3/7 |
| **C5** | **11** | **4** | **1** | **6** | 0 | 6 | **4/11** |
| R1 | 6 | 0 | 5 | 1 | 0 | 1 | 0/6 |
| P0 | 9 | 0 | 4 | 3 | 2 | 4 | 0/9 |
| M400 | 4 | 0 | 4 | 0 | 0 | 0 | 0/4 |
| **Total** | **37** | **7** | **16** | **11** | **3** | **12** | |

**Discrepancies with the session report** (`docs/hardware_session_2026-11-20.md`):

1. **P0 row: "9 attempts, 5 drops, 4 safety stops, 2 discards" sums to 11, not 9.** Two recorded trials were counted as both a drop and a stop.
   - Trial 4 (`095740`): drop reported at about 5 s; current guard 462 mA at 15.8 s. The cube was already out, so this audit counts it as a **drop**.
   - Trial 11 (`101853`): drop reported at 30 s; guard 452 mA at 30.1 s. The order cannot be resolved from the logs, so it is a **guard stop, low confidence**.

   Corrected P0: 4 drops, 3 guard stops, 2 unknown, 0 completed = 9.

   Sensitivity: counting trial 11 as a drop gives 5 / 2 / 2. No other policy is affected.
2. **The 10 unrecorded safety stops and 3 discards are confirmed.** Guard stops not in the CSV: A3 1, C5 6, R1 1, P0 2. Discards: A3 at 22.6 s, P0 at 18.1 s and 5.1 s. These are Ctrl+C stops with no reason recorded; their logs show nothing unusual: peaks ≤ 383 mA, no saturation episode.
3. **C5's "tracking error 0.68 rad at 4.7 s"** (09:46) **cannot be verified from the log** (pre-patch, `stop_reason = "error"`). The last logged sample (4.68 s) has a max tracking error of 0.637 rad on if_mcp, against the 0.65 rad abort, with if_mcp at the 350 mA limit. That is consistent with a tracking-error trip on the next read and with the report's "index knuckle jamming".
4. **Guard stops by type:** all 11 guard-stop attempts, and the guard event inside P0 trial 4, are **current** trips, apart from that one unrecorded (probably tracking) C5 stop.

Per-attempt table: `LEAP_Hand_RL_Deploy_Kit/audit/20261120/attempts.csv` (also `summary.md`).

## 3. Quarter-turns and full revolutions

The operator's counts are **quarter-turns** (90°). **Full revolutions = quarter-turns / 4.** Rate = full revolutions / time held × 60, only for trials held ≥ 20 s. Recorded trials only; no trial had an intervention.

| Policy | Trials counted | Quarter-turns per trial (mean ± s.d.) | **Full revolutions per trial** | Full rev/min while held (n) | Mean time held [s] |
|---|---|---|---|---|---|
| A3 | 4 of 5 (trial 1 blank) | 2.0 ± 1.4 | 0.50 ± 0.35 | 0.37 (2) | 40.6 |
| **C5** | 5 | **13.9 ± 2.2** (17, 13, 11, 13.5, 15) | **3.48 ± 0.55** | **3.64 (5)** | 57.3 |
| R1 | 5 | 2.6 ± 1.3 | 0.65 ± 0.34 | — (0) | 9.2 |
| P0 | 5 | 0.0 | 0.00 | 0.00 (1) | 10.8 |
| M400 | 4 | 2.0 ± 0.8 | 0.50 ± 0.20 | 1.20 (1) | 13.5 |

- In the operator's notes, two of A3's trials rotated about a horizontal axis. Their counts (blank, and 0) are not vertical rotations.
- **For scale:** simulated C5 makes 1.87 full turns per 30 s on the full grasp distribution (3.7 rev/min) and 2.46 per 30 s from grasp 49 (4.9 rev/min). C5's 3.6 rev/min on the hand is therefore the same order as the simulation, now that the unit is corrected.

## 4. Current by joint and finger

Figure: `audit/20261120/figures/fig_current_per_joint_by_policy.png`. Data: `per_joint.csv`, `per_finger.csv`.

**Finger share of the summed |current|** (mean over attempts; summed mean / p99 in mA):

| Group | Index | Middle | Ring | Thumb |
|---|---|---|---|---|
| C5 guard stop (6) | **0.43** (553 / 894) | 0.13 | 0.12 | 0.32 (407 / 747) |
| C5 completed (4) | 0.19 (150 / 570) | 0.25 | 0.26 | 0.30 (232 / 645) |
| C5 dropped (1) | 0.20 | 0.26 | 0.25 | 0.29 |
| A3 (7) | 0.28 | 0.17 | 0.24 | 0.31 |
| R1 (6) | 0.25 | 0.22 | 0.26 | 0.28 |
| P0 (9) | 0.34 | 0.22 | 0.18 | 0.26 |
| M400 (4) | 0.16 | 0.28 | 0.29 | 0.28 |

**Most loaded joints** (share of samples at ≥ 340 mA, i.e. at the 350 mA limit; mean p99 current):

| Group | 1st | 2nd | 3rd |
|---|---|---|---|
| C5 guard stop | **if_mcp 27%** (p99 380 mA) | **if_pip 23%** (397) | **th_axl 15%** (383) |
| C5 completed | rf_mcp 2.4% | th_axl 1.3% | if_pip 0.5% |
| A3 | th_axl 1.2% | th_mcp 0.9% | if_mcp 0.5% |
| R1 | if_pip 3.0% | th_axl 1.8% | rf_pip 0.8% |
| P0 | if_mcp 4.5% | th_axl 2.4% | if_dip 2.0% |
| M400 | ≈ 0 | | |

- **Concentration.** Demand during C5's failures is concentrated in two index joints, if_mcp and if_pip, with thumb abduction pushing back. The index and thumb then carry 75% of all current. In completed C5 runs, load is spread over all four fingers.
- **Sustained, not only spikes.** In C5's guard stops a joint sits at the limit for most of the run: index blocked fraction 0.11–0.43, against 0.001–0.008 in completed runs. On top of this plateau, the trip itself is a jump of 41–420 mA within one 50 ms period, which is never logged.
- **Phase.** Same time windows for every attempt (`fig_c5_current_completed_vs_guard.png`): the index is at the limit 5–17% of the time in 0–2 s for both groups, and 25–60% in 2–4 s for **both**. From 4 s onward, completed runs are at **0%** and guard-stop runs at **56–97%**.
- **Current vs tracking error.** Quasi-static current ≈ **1,450–1,500 mA/rad × position error** on every joint (r = 0.93–1.00), clipped at 350 mA (`fig_current_vs_tracking_error.png`). The limit is reached at an error of **0.23–0.27 rad**. Within runs, Spearman(|error|, |current|) = 0.97–0.99.
- **What current means here.** It is mainly the position controller's effort, so high current means the joint is far from its command. Contact, gravity and friction enter only through that error. Current is not a calibrated force.

## 5. Tracking and effective lag

Figures: `fig_tracking_per_joint_by_policy.png`, `fig_c5_tracking_completed_vs_guard.png`, `fig_lag_hardware_vs_sim.png`.

- **Tracking error** = \|command(k−1) − q(k)\|. Median per joint in completed C5 runs: 0.004–0.041 rad.
  - In C5's guard stops: p95 index 0.33–0.44 rad and th_axl 0.29 rad. In the five jam stops, max over the last 2 s: 0.49–0.64 rad, always on if_mcp.
  - The tracking guard (0.65 rad) was only approached, by the pre-patch C5 run (0.637 rad at its last logged sample).
- **The error is an offset, not a delay.** During a jam, the measured index MCP/PIP stay flat (about 0.15 and 1.6 rad) while the command keeps oscillating 0.3–0.6 rad away (guard-event figures, panel 3). That is contact resistance at the current limit, not slow tracking.
- **In completed runs, by contrast,** index PIP curls fully to the kit limit (1.885 rad) and the thumb sweeps with close tracking.
- **Effective lag.** Two estimators, both on the real timestamps (read and write times), using only unsaturated samples (|I| < 200 mA on that joint), for runs ≥ 10 s:
  - (i) τ minimising mean |q(t) − c(t − τ)|, with the command linearly interpolated between write times;
  - (ii) τ maximising corr(measured velocity, command rate shifted by τ), which is insensitive to offsets.

  Both give **80 ms** (IQR 75–85 ms) for every policy, and 60–100 ms on every joint. The estimators recover known delays on synthetic data (tests).

  This is an *effective* lag. It contains the write → read convention (≈ 4 ms), the Dynamixel position loop, inertia and contact. It is not a pure transport delay. It is consistent with the 17 Nov estimate of "about 2 steps (~100 ms)", which was quantised to whole steps.
- **The same estimators on simulation**, using the frozen checkpoints, grasp 49, 32 environments and training actuator settings, give a median of **~280 ms** (joint medians 160–600 ms). Side joints that hardly move (e.g. C5's if_rot) are poorly determined.
  - Simulated tracking error is about **3.5× larger** than on the hand (median of joint medians 0.10 vs 0.03 rad).
  - The real hand is therefore faster and stiffer in tracking than the training simulator. The premise "real lag ≈ 100 ms > training delay 25–75 ms" compares an effective lag with a pure delay, and **is not supported**.

## 6. Joint velocity and movement

Figure: `fig_velocity_per_policy.png`.

| Measure | A3 completed | C5 completed | Simulated A3 (grasp 49) | Simulated C5 (grasp 49) |
|---|---|---|---|---|
| Mean \|joint velocity\| [rad/s] | 0.37 | **0.53** | 0.22 | 0.29 |
| Mean \|command rate\| [rad/s] | 0.48 | 0.66 | 0.41 | 0.51 |
| Command steps at the ±1/24 rad maximum | 18% | 41% | 16% | 37% |

- **C5 is faster on the hand too.** All attempts, mean |qd| 0.46 vs 0.39 rad/s (+18%); completed runs +42%. On the hand, joints move 1.7–1.9× faster than in simulation for the same policy.
- **Speed does not precede C5's failures.** Same windows, 0–2 s: guard-stop runs |qd| 0.42–0.47 rad/s and 30–37% full-speed steps; completed runs 0.40–0.46 rad/s and 28–34%.
  - After 2 s, completed runs are faster (4–8 s: 0.52–0.53 vs 0.34–0.40 rad/s), because the jam stops the index.
  - Across all samples, Spearman(|qd|, tracking error) = 0.02–0.41 per run, against 0.97–0.99 for current vs tracking error.
- **No abrupt command transitions.** Every command step is ≤ 1/24 rad. The maximum-step fraction is *higher* in completed C5 runs (0.40–0.41) than in guard stops (0.28–0.35). No evidence for a smoothness or command-change penalty.
- **Command limits.** In completed C5 runs, index PIP spends long stretches at the kit's upper command bound (1.885 rad, above training's 1.825 rad; the known PIP gap, `hwlimit_ac` §8.5). In jammed runs it never gets there.

## 7. Control-loop timing

Figure: `fig_timing.png`.

| Policy | Median interval | p95 | p99 | Max | > 50 ms | > 60 ms | > 100 ms | Compute median / p99 |
|---|---|---|---|---|---|---|---|---|
| A3 | 50.1 ms | 53.2 | 88.5 | 127 | 75% | 6.6% | 0.34% | 4.2 / 42 ms |
| C5 | 50.1 | 53.9 | 88.0 | 126 | 79% | 5.4% | 0.35% | 4.2 / 42 |
| R1 | 50.1 | 71.7 | 88.7 | 127 | 77% | 4.4% | 0.14% | 4.1 / 42 |
| P0 | 50.1 | 52.9 | 88.0 | 127 | 74% | 3.8% | 0.12% | 4.3 / 42 |
| M400 | 50.1 | 52.4 | 88.1 | 126 | 75% | 4.6% | 0.26% | 4.1 / 42 |

- **"> 50 ms" is not informative.** The nominal period is 50.1 ms.
- **Long intervals are single late cycles.** 1,067 of the 1,111 intervals > 60 ms are followed immediately by a < 40 ms catch-up interval. The cadence shifts once rather than losing a step. The mean is 50.2–50.35 ms in every policy.
- **Timing is not linked to failure.** C5 guard stops have 3.1–10.9% of intervals > 60 ms, completed runs 4.2–5.8%. The rate in the last 2 s before a stop (2.5–10%) matches the rest of the run.
- **Re-evaluation of the session report:** confirmed. 3.8–6.6% > 60 ms per policy (report: 4.9% mean, max 8.1% per trial).

## 8. Guard stops, one by one

One figure per event (current, tracking error, the involved finger's q vs command, loop interval, trip marker): `audit/20261120/figures/guard_events/`. Details: `guard_events.json` and `guard_events.csv`.

Trip classes:
- **Sustained jam:** a jam episode on one joint is unresolved at the stop. A jam is a 1 s window with ≥ 50% of samples at ≥ 340 mA **and** ≥ 0.2 rad from the command.
- **Start transient:** the trip comes ≤ 3 s into the run, with the last logged current below the limit.
- **Other.**

| Policy | Log | Trial | Outcome | Trip | t [s] | Class | Last logged max \|I\| → trip | Loaded joints (share at the limit, last 2 s) | Max tracking, last 2 s |
|---|---|---|---|---|---|---|---|---|---|
| C5 | 094641 | – | guard stop | not logged (pre-patch) | 4.7 | **sustained jam** | 350 → ? | if_pip 45%, if_mcp 28%, th_axl 10% | 0.64 if_mcp |
| C5 | 095913 | – | guard stop | 677 mA | 6.5 | **sustained jam** | 349 → 677 | if_mcp 38%, if_pip 22%, th_axl 10% | 0.49 if_mcp |
| C5 | 100007 | – | guard stop | 747 mA | 15.6 | **sustained jam** | 327 → 747 | if_mcp 52%, if_pip 30%, th_axl 25% | 0.59 if_mcp |
| C5 | 102433 | – | guard stop | 453 mA | 8.1 | **sustained jam** | 369 → 453 | if_mcp 92%, if_pip 30%, th_axl 28% | 0.61 if_mcp |
| C5 | 102505 | – | guard stop | 713 mA | 7.6 | **sustained jam** | 325 → 713 | if_mcp 50%, if_pip 38%, th_ipl 18% | 0.50 if_mcp |
| C5 | 102011 | – | guard stop | 454 mA | 3.1 | other (early load, no jam formed yet) | 413 → 454 | th_axl 28%, if_pip 18%, if_mcp 15% | 0.33 th_axl |
| P0 | 095740 | 4 | drop (guard after the drop) | 462 mA | 15.8 | sustained jam | 285 → 462 | if_mcp 42%, if_pip 22%, th_axl 22% | 0.42 if_mcp |
| P0 | 101853 | 11 | guard stop (drop coincident) | 452 mA | 30.1 | other | 281 → 452 | if_pip 8% | 0.29 if_pip |
| P0 | 100412 | – | guard stop | 848 mA | 2.9 | start transient | 234 → 848 | if_dip 10% | 0.28 if_dip |
| P0 | 103327 | – | guard stop | 1048 mA | 0.8 | start transient | 195 → 1048 | none | 0.14 if_pip |
| R1 | 101248 | – | guard stop | 810 mA | 1.0 | start transient | 331 → 810 | if_pip 15%, th_axl 10% | 0.25 if_pip |
| A3 | 102817 | – | guard stop | 549 mA | 1.7 | start transient | 246 → 549 | none | 0.18 if_mcp |

Two common features of all events:
- Joint velocity before each stop is ordinary: max 1.0–2.1 rad/s against the 8 rad/s guard.
- The maximum loop interval in the last 2 s is 51–126 ms, the same as elsewhere in the runs.

Two mechanisms:
1. **Jam trips: C5 ×5, P0 ×1, plus P0 trial 4 after its drop.** A blocked index finger (MCP and PIP) is pressed at the current limit while thumb abduction is also loaded, consistent with the cube wedged between index and thumb. The trip is a current transient on top of that plateau. **The physical trigger of the transient cannot be determined at 20 Hz.** Candidates, all hypotheses: a slip, a command reversal into the obstacle, or the motor's current-loop overshoot.
2. **Start transients: A3, R1, P0 ×2, all ≤ 2.9 s.** A jump of 300–850 mA within one control period, from logged currents well below the limit. They happen in the handover from the closing ramp to the first policy actions. They are not specific to C5, and their mechanism is not resolvable from these logs.

## 9. C5: successful vs failed runs

Data: `c5_comparison.json`. Figures: `fig_c5_*`.

| Window | Index at the limit (completed / guard) | Index current, summed [mA] | Mean \|qd\| [rad/s] | p95 tracking (any joint) [rad] |
|---|---|---|---|---|
| 0–2 s | 0.03–0.17 / 0.05–0.15 | 330–425 / 380–435 | 0.40–0.46 / 0.42–0.47 | 0.30–0.36 / 0.25–0.31 |
| 2–4 s | 0.00–0.33 / 0.40–0.60 | 205–440 / 510–715 | 0.42–0.54 / 0.38–0.43 | 0.12–0.32 / 0.28–0.57 |
| 4–8 s | **0.00** / **0.56–0.77** | 145–185 / 545–675 | 0.52–0.53 / 0.34–0.40 | 0.16–0.20 / 0.47–0.61 |

- **The groups separate after the start transient.** Every C5 run loads the index in its first seconds. Completed runs resolve it within about 1 s; guard-stop runs stay jammed until the trip.
- **Same commands, different trajectories.** Commands and joint speeds in 0–2 s are statistically indistinguishable. The proprioceptive logs do not show *why* a given run escapes.
- **Undetermined:** sub-millimetre placement differences or contact geometry may decide it, but that cannot be tested without cube tracking. The cube placement followed the simulation reference throughout, and the data give no reason to blame placement.

## 10. Simulation comparison

Evaluation only; frozen checkpoints; no training. Files: `training_source/project/experiments/c5_hw_v1/sim_reference/`, `results/sim_effort_diagnostic.json`.

- **The training actuator is current-mapped** (`leap_right_constants.py`: τ = 0.53 Nm/A × i, i = Kp·error − Kd·qd, limit 550 mA).
  - The SysID-scaled stiffness matches the hand: if_mcp 0.763 Nm/rad ÷ 0.53 = 1,440 mA/rad, measured 1,493 mA/rad.
  - The **current limit does not match**: training uses 550 mA ± 10% (0.262–0.321 Nm), the hand ran at 350 mA.
  - Simulated saturation error: 0.28–0.62 rad. Hardware: 0.23–0.27 rad.
- **Training actuators:** no policy ever saturates (≤ 0.05% of samples at ≥ 98% of the limit).
- **Effort limit set to the deployed 350 mA** (0.167–0.204 Nm, same checkpoints, grasp 49, 256 envs):

| Policy | Index MCP/PIP at the limit | Envs with ≥ 1 s index saturation | Most frequently saturated joint | Turns / 30 s (default → capped) |
|---|---|---|---|---|
| **C5** | **7.6%** (79% of envs in the first 2 s) | 2.7% | **if_mcp** | 2.46 → 2.44 |
| P0 | 3.5% | 1.6% | if_mcp | 1.68 → 1.67 |
| A3 | 0.08% | 0 | rf_mcp | 1.78 → 1.77 |
| R1 | 0.09% | 0 | rf_mcp | 1.61 → 1.60 |

- **The policy ranking and the joint match the hand.** C5, then P0, load the index; A3 and R1 do not.
- **Persistence is not reproduced.** In simulation, episodes recur but do not lock up, and there is no guard. The persistent hardware jam is therefore not reproduced.
- **Speed survives the cap.** Rotation of the frozen C5 is essentially unchanged under the cap. That suggests training with the real limit need not cost speed (a hypothesis to test).

## 11. Data quality and limitations

- **All 37 files** are complete, readable and monotonic, with no non-finite values, no chunk/part files, no CSV mismatch in policy or checksum, and the tracking scalar reproduced exactly.
- **Unlogged trip samples.** The tripping sample of every guard stop is missing by design. Trip values come only from the stop message, and one log has no message.
- **Invisible ramp trips.** Attempts that tripped during the ramp would have no log; none can be counted.
- **Sample size.** 4 completed vs 6 failed C5 runs. All between-group statements are descriptive.
- **No ground truth.** There is no cube pose or force measurement: no claims about forces, cube motion or contact location. "Wedged between index and thumb" is an interpretation of joint data, not an observation.
- **20 Hz sampling.** Transients shorter than 50 ms, including the trip spikes, cannot be resolved.
- **Lag is effective, not transport.** The effective lag mixes delay, servo dynamics and contact. It is not a pure transport delay, and it is undefined for joints that barely move.
- **Operator counts.** Quarter-turn counts are by eye; drop times are approximate (±1–2 s), which matters for P0 trial 11.

## 12. Conclusions

| Strength | Conclusion |
|---|---|
| **Strongly supported** | C5's guard stops are index (MCP/PIP) jams at the current limit, with thumb abduction loaded. |
| **Strongly supported** | Current and tracking error are the same signal on this hand, saturating at ≈ 0.24 rad. |
| **Strongly supported** | Speed, abrupt commands and loop timing do not precede the failures. |
| **Strongly supported** | The real hand's effective lag is ≈ 80 ms and *shorter* than the simulator's (≈ 280 ms). |
| **Strongly supported** | Training's actuator current limit (550 mA) exceeds the deployed 350 mA. With it matched, C5 alone loads the index in simulation. |
| **Strongly supported** | Corrected counts (Section 2) and full revolutions (Section 3). |
| Tentative | The current-limit mismatch is *why* C5 learned an index-forcing gait. |
| Tentative | The simulator's slow response contributes to the early transient. |
| Tentative | Start transients (A3/R1/P0) come from the ramp→policy handover. |
| Cannot be determined | Why some C5 runs escape the early jam. |
| Cannot be determined | The physical trigger of each trip spike. |
| Cannot be determined | Cube motion or contact location. |
| Cannot be determined | Force magnitudes. |

## 13. Recommendation: C5-HW-V1

Full specification with pre-registered criteria: `training_source/project/experiments/c5_hw_v1/SPEC.md`.

| Condition | Change | Basis |
|---|---|---|
| C5 baseline | unchanged C5, seeds 1–3 | reference |
| **C5-HW-effort** (primary) | Effort limit 0.262–0.321 → **0.167–0.204 Nm** (350 mA in the simulator's own current mapping), plus a smooth utilisation penalty −w Σ_j [max(0, u_j − 0.8)/0.2]². w is calibrated before training to 10% of C5's rotation reward. | §4, §10 |
| **C5-HW-delay** (re-framed as response-matched) | Refit delay range, damping and friction so the simulated effective lag (same estimator) matches the hardware's **80 ms**. Do **not** extend the delay. | §5 |
| C5-HW-combined | both; only if each passes its criteria | |
| Not recommended | smoothness or command-change penalty; reduced yaw clip; timing changes | §6, §7 |

- **Evaluation:** under both the training and a hardware-matched actuator.
- **Primary safety metric:** the jam proxy (environments with a ≥ 1 s index jam). Rotation must stay ≥ 90% of C5 and ≥ 1.2 × A.
- **Plan:** three seeds per condition.
- **Not started:** no training until this design is reviewed. Any hardware confirmation must be separately approved and supervised; guard-stop rate is the primary metric.

## Files

| Path | Contents |
|---|---|
| `LEAP_Hand_RL_Deploy_Kit/kit/audit_hardware_logs.py` | offline audit tool (no hardware access) |
| `tests/test_audit_hardware_logs.py` | 30 tests: mapping vs `deploy_direct.py`, missing and malformed fields, truncated files, timestamps, classification, CSV matching, quarter-turns, current and tracking metrics, lag estimator, jam episodes, trip classes, real-log reconciliation (raw logs unmodified) |
| `LEAP_Hand_RL_Deploy_Kit/audit/20261120/` | `attempts.csv`, `per_joint.csv`, `per_finger.csv`, `policy_summary.{csv,json}`, `guard_events.{csv,json}`, `c5_comparison.json`, `current_vs_error.json`, `lag_summary.json`, `sim_reference.json`, `data_quality.json`, `reconciliation.json`, `audit.json` (everything), `summary.md` |
| `LEAP_Hand_RL_Deploy_Kit/audit/20261120/figures/` | per-joint current and tracking by policy; C5 completed vs guard (current, tracking); velocity; timing; current vs error; lag hardware vs simulation; `guard_events/` (12 event plots, 6 for C5) |
| `training_source/project/experiments/c5_hw_v1/` | `SPEC.md`; `scripts/sim_actuator_reference.py`, `scripts/sim_effort_diagnostic.py`; `sim_reference/` (8 evaluation rollouts); `results/sim_effort_diagnostic.json` |
