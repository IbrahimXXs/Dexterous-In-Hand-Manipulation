# Addendum: 1-hour session (written 27 Nov 2026, before the session)

> **Status: ADDENDUM to `docs/forecast_final_session.md`, written and hashed before the session.**
> The original forecast and the protocol draft are **not edited**; their hashes in `training_source/project/experiments/final_session_forecast/logs/forecast.sha256` still match.
> sha256 and timestamp of this addendum: `training_source/project/experiments/final_session_forecast/logs/addendum_1h.sha256`, `addendum_1h_hashed_at.txt`.

## 1. What changes

The hand is available for **one hour**, not the ~2 h that the protocol's 36 attempts need (protocol section 2: about 3 min per attempt; earlier sessions averaged 2.4–2.6 min per attempt from first to last trial, `LEAP_Hand_RL_Deploy_Kit/logs/trials.csv`, `trials_20261124.csv`).

- **Attempts:** `TRIAL_BLOCKS=6` instead of 12: **18 attempts, 6 per arm.** This is each of the six Williams sequences once, so the design stays balanced for position and carry-over. Everything else in the protocol (arms, run length, camera on every attempt, metrics, exclusions, guards, stopping rules in section 8) is unchanged.
- **Stopping rule, fixed now and independent of results:** if 50 minutes after the first attempt the 18 are not done, finish the current 3-attempt sequence and stop. The number of attempts run and the reason are reported.
- **Not run in this session:** optional arms, the fingertip-tape block, the grasp-49 start and the camera-to-palm calibration.
- **Placement:** a camera placement check replaces placement by eye (section 2), if it passes its validation before the first attempt. Otherwise placement is by eye, as on 24/25 Nov, and the report says so.

## 2. Camera placement check (placement aid, protocol section 4)

- **What it is:** a fixed ArUco reference marker above the hand. The camera measures the cube pose relative to that marker (marker PnP and depth, as in the tracker). The trial's live camera panel compares it with a taught target pose and shows **GO (green)** or **NOT YET (red)** with the offset in mm and degrees.
- **Target:** taught once, before the first attempt. The cube is placed as in `docs/figures/cube_placement.png`, frames are averaged, and the operator and supervisor confirm by eye that it matches the figure.
- **Rule and tolerances:** fixed in `docs/placement_check.md` and hashed before the first attempt, together with the target file and a short validation:
  - noise of the check with the cube still;
  - five re-placements read GO;
  - deliberate 5 mm and 15° offsets read NOT YET.
- **During the session:**
  - The closing ramp starts only after the operator sees GO and presses Enter. The check is advisory: an operator override is allowed and logged.
  - The check never changes what the kit commands, its guards, `configs/hand.env` or the policies.
- **Logged for every attempt:** placement error (position in mm, yaw and tilt in degrees) at GO and at policy start (end of the closing ramp), in a new file next to the trial CSV.
- **Exploratory analysis (descriptive, small n):** placement error at policy start against time to drop, per arm. The question is whether early drops (< 10 s) start from larger placement errors than attempts held ≥ 20 s.
- **Expected direction:** already stated in assumption 1 of the original forecast, and still not applied to the numbers. If placement improves, C5s3 should end **above** its held-60-s interval, and more attempts should pass the 10-s and 20-s rules. The comparison between arms is unaffected, because all three arms use the same check.

## 3. Forecast for 6 attempts per arm

`scripts/forecast.py` re-run, **unchanged**, with 6 attempts per arm instead of 12 (`scripts/forecast_6.py`: same inputs, method and seed 20261127). Output: `training_source/project/experiments/final_session_forecast/addendum_1h/results/forecast.json` and `forecast_tables.md`. Point forecast and **90% prediction interval** for the per-arm statistic:

| Arm | Camera rev/min while held (mean, held ≥ 10 s) | Quarter turns per finger cycle (camera, held ≥ 20 s) | Finger rhythm Hz (median, held ≥ 20 s) | Attempts held 60 s (of 6) |
|---|---|---|---|---|
| M400 | 2.60 [1.23, 4.05] | 0.25 [0.17, 0.38] | 0.26 [0.17, 0.38] | 0 [0, 3] |
| A3 | 0.84 [0.00, 1.97] | 0.05 [0.00, 0.11] | 0.50 [0.16, 0.78] | 4 [1, 6] |
| C5s3 | 4.04 [2.88, 5.70] | 0.94 [0.65, 1.56] | 0.29 [0.21, 0.38] | 1 [0, 3] |

The intervals are wider than for 12 attempts, as expected. Simulated probability that a cell has no value (no attempt held long enough): M400 turns per cycle 0.33 and rhythm 0.34; A3 turns per cycle 0.08; every other cell < 0.07 (`forecast.json`, `arms.*.forecast.*.p_undefined`).

## 4. Scoring

- **If 18 attempts are run:** score this addendum's table, as in section 5 of the original forecast.
- **If fewer are run** (the stopping rule above): still score this table, and state the number of attempts per arm next to each cell.
- The 12-attempt forecast is reported too, marked "designed for 12 attempts per arm; session ran 6".
