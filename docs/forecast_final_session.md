# Forecast for the final real-hand session (M400, A3, C5s3)

> **Status: FORECAST, written 27 Nov 2026 before the session, from existing data only.**
> - Hashed before the session: sha256 and timestamp in `training_source/project/experiments/final_session_forecast/logs/`.
> - **Not edited after hashing.** After the session it is scored against the results (section 5). Any later comment goes in a separate file.
> - Protocol: `docs/hardware_protocol_final_DRAFT.md` (12 attempts per arm; metric definitions in its section 6).
> - All numbers below come from `training_source/project/experiments/final_session_forecast/results/forecast.json`, written by `scripts/forecast.py` in the same folder.

## 1. Forecast

Each cell gives the point forecast and a **90% prediction interval** [5th, 95th percentile] for the session's **per-arm statistic**, not for single attempts.

| Arm | Camera rev/min while held (mean over attempts held ≥ 10 s) | Quarter turns per finger cycle (camera, window 3 s → drop; ratio of sums, held ≥ 20 s) | Finger rhythm, Hz (median, held ≥ 20 s) | Attempts held 60 s (of 12) |
|---|---|---|---|---|
| M400 | 2.63 [1.42, 3.79] | 0.25 [0.18, 0.36] | 0.26 [0.18, 0.36] | 0 [0, 5] |
| A3 | 0.84 [0.12, 1.80] | 0.05 [0.00, 0.10] | 0.50 [0.17, 0.73] | 7 [2, 11] |
| C5s3 | 4.09 [3.15, 5.30] | 0.95 [0.70, 1.35] | 0.29 [0.22, 0.36] | 1 [0, 4] |

**Undefined quantities.** A quantity can have no value in the session when no attempt is held long enough. The simulated probability of that:
- M400 turns per cycle 0.20 and rhythm 0.20 (only 1 of 4 earlier M400 attempts was held ≥ 20 s);
- every other cell ≤ 0.03.

If it happens, the cell is scored as "not available", not as a miss.

**In words.**
- C5s3 is forecast to turn fastest while held, near 1 quarter turn per finger cycle at about 0.29 Hz. Like M400, it is forecast to rarely hold 60 s; C5s3's early drops were attributed to placement (24 Nov).
- A3 is forecast to hold the cube most often but to barely turn it.
- M400's rate rests on three short holds (10–25 s), so its interval is wide.

## 2. Definitions (as in the protocol, section 6)

- **Camera rev/min while held:** camera quarter turns from policy start to the drop (or the end of the run) ÷ 4 ÷ held minutes. Per-arm mean over attempts held ≥ 10 s.
- **Quarter turns per finger cycle:** camera quarter turns inside the window 3 s → drop ÷ encoder finger cycles in the same window. The cycles come from `count_cycles` in `gait_efficiency/scripts/gait.py`, unchanged. Per-arm ratio of sums over attempts held ≥ 20 s.
- **Finger rhythm:** FFT peak of the first principal component of the joint positions, same window and counter. Per-arm median over attempts held ≥ 20 s.
- **Held 60 s:** no drop and no stop (the run ended at `max_steps`, not intervened).

## 3. Inputs and their sources

| Input | Value | Source |
|---|---|---|
| Earlier attempts per arm | M400 4 (20 Nov trials 21–24); A3 5 (20 Nov trials 1, 7, 10, 16, 17); C5s3 18 (24 Nov 13, 25 Nov 5) | `LEAP_Hand_RL_Deploy_Kit/logs/trials.csv`, `trials_20261124.csv`, `trials_20261125.csv` |
| Operator quarter turns, drop time, stop reason, intervened | per attempt | the same CSVs |
| Encoder cycles and rhythm | per attempt held ≥ 20 s, recomputed with `gait.py`. The 18 rows with an operator count equal `gait_efficiency/results/real_hand.json` (variant `held`) | run logs `physical_*.npz` (`q_hw`, `elapsed_s`) |
| Camera / operator, whole hold (for rev/min) | 1.01 (25 Nov trial 1: 16.17 vs 16), 0.87 (trial 5: 6.95 vs 8) | `LEAP_Hand_RL_Deploy_Kit/logs/trials_20261125_vision.csv` |
| Camera in window 3 s → drop / operator, whole hold (for turns per cycle) | 0.97 (trial 1), 0.76 (trial 5) | `gait_efficiency/results/real_hand.json`, `camera_check_25nov.trial{1,5}.camera_qturns_in_window` |
| Per-attempt spread (log SD, pooled within session over all policies) | rev/min 0.37; turns per cycle 0.24; rhythm 0.31 | `forecast.json`, `dispersion_inputs.s_run` |
| Session-to-session shift (log SD) | rev/min 0.072; turns per cycle 0.105; rhythm 0.028 | `forecast.json`, `dispersion_inputs.session` (from C5s1 20 vs 24 Nov and C5s3 24 vs 25 Nov) |

**Prior values per arm, operator-based, before conversion and before the session simulation:**

| Arm | Rev/min (mean, held ≥ 10 s; n) | Quarter turns per cycle (operator; n) | Rhythm Hz (median; n) | Held 60 s |
|---|---|---|---|---|
| M400 | 2.65 (3) | 0.29 (1) | 0.25 (1) | 0 of 4 |
| A3 | 0.92 (3) | 0.06 (2) | 0.53 (3) | 3 of 5 |
| C5s3 | 4.25 (8) | 1.08 (7) | 0.30 (7) | 2 of 18 |

A3's rhythm here includes 20 Nov trial 1. That attempt has no operator count, but rhythm needs none. Its three runs are 0.18, 0.53 and 0.70 Hz. The `gait_efficiency` value of 0.35 Hz is the median of the two counted runs.

## 4. Derivation

For every arm and quantity, 20,000 sessions of 12 attempts are simulated (`scripts/forecast.py`, seed 20261127):

1. **Parameter uncertainty.** Dirichlet(1, …, 1) weights are drawn over the arm's earlier attempts (a Bayesian bootstrap).
2. **New attempts.** 12 attempts are drawn with those weights. Each keeps its held time, so the share that passes the 10-s or 20-s rule is drawn as well.
3. **Small-n smoothing.** Each drawn value is multiplied by exp(N(0, h²)), with h = per-attempt spread × n^(−1/5) and n the number of eligible earlier attempts.
   - This lets an arm with one or two earlier runs produce values it has not shown.
   - The spread is pooled over every policy and session with at least 2 positive values (all six policies). Zeros cannot enter a log SD.
4. **Camera conversion** (rev/min and turns per cycle only). Each attempt is multiplied by a log-normal factor fitted to the two 25 Nov ratios above:
   - rev/min: log mean −0.065, log SD 0.107;
   - turns per cycle: log mean −0.152, log SD 0.167.

   The turns-per-cycle factor covers two things: the camera instead of the operator, and the window starting at 3 s.
5. **Session shift.** The arm statistic is multiplied by exp(N(0, s²)), where s is the RMS of the two C5 session-to-session log ratios ÷ √2. That RMS still contains sampling noise, so s is an overestimate (conservative).
6. **Result.** Point forecast = median; interval = 5th–95th percentile.

**Held 60 s:** a Jeffreys beta-binomial. The share is drawn from Beta(k + ½, n − k + ½), then 12 attempts are drawn; the cell gives the median and the 5th–95th percentile.

### Assumptions and their direction

1. **Same conditions as 20/24/25 Nov:** the same kit, guards and checkpoints, and placement by eye.
   - If the placement template (protocol section 4) is used and works, C5s3 should end up **above** its held-60-s interval and have more attempts that pass the 10-s and 20-s rules.
   - Its while-held rates should change less.
   - This direction is stated now; the numbers above are not adjusted for it.
2. **Camera agrees with the operator as in 25 Nov trials 1 and 5.** In the one short hold with both counts (25 Nov trial 4, held 5 s), the camera read 0.82 quarter turns against the operator's 2. M400's rate rests mostly on two holds of 10–12 s with 2–3 counted quarter turns, so its forecast is more likely **too high** than too low.
3. **Direction:** every counted rotation is assumed to go the trained way. An arm that turns the other way reads negative (protocol section 6) and would fall **below** its interval.
4. **A3 tumbles:** in 20 Nov trials 1 and 10 the operator noted rotation about another axis. The camera counts only rotation about the session axis.
5. **Small n:** M400's turns per cycle and rhythm rest on one run (20 Nov trial 23), and A3's turns per cycle on two. Their intervals come mostly from the borrowed spread, not from their own data.
6. **What the intervals cover:** sampling of 12 attempts, uncertainty from the earlier attempts, a session shift, and camera conversion. They do **not** cover a change in the hand, cube or kit.

## 5. Scoring after the session

- For each of the 12 cells: is the session value inside its 90% interval? Report the session value next to the forecast.
- If the forecast is calibrated, about 1 of 12 cells falls outside (12 × 10%).
- Cells that cannot be computed are reported as not available.
- Every cell is reported, including misses. The forecast and its hash are not changed.
- Score against the session's per-arm statistics exactly as defined in section 2, with the protocol's exclusions.

## Reproduction

```bash
uv run --no-project --with numpy --with scipy python \
    training_source/project/experiments/final_session_forecast/scripts/forecast.py
```

This writes `results/forecast.json` and `results/forecast_tables.md`. The run is deterministic: a re-run gives the same file.
