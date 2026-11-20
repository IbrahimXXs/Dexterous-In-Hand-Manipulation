# LEAP Hand hardware session, 20 Nov 2026: A3 vs C5 vs R1 vs P0 (+ model_400)

**Status:** the planned 20 trials are done (5 per policy, balanced order), plus 4 reference trials of the professor's model_400 (M400). All runs used the live estimator monitor. The operator observed every recorded trial. 13 further attempts are in the logs but not in the trial record (Section 3); they are counted in the reliability results.

## Summary

| Question | Answer on the real hand | Strength |
|---|---|---|
| Q1: A3 vs C5, which rotates faster? | **C5, by far.** 13.9 vs 2.0 rotations per trial (+11.9 [+9.9, +14.1], p = 0.009). | Clear. Matches the simulation's ranking. |
| Q2: R1 vs P0, which keeps the cube more reliably? | **Neither.** Both dropped the cube in 5 of 5 trials, after about 9–11 s. | No difference; both fail. |
| Q3: A3 vs R1, did the retention reward improve real-world stability? | **No; R1 was worse.** A3 dropped in 2/5 trials and held 40.6 s on average; R1 dropped in 5/5 and held 9.2 s (−31 s [−52, −10], p = 0.07). | Opposite of the simulation. |
| Q4: C5 vs P0, speed vs retention | **C5 better on both** in the recorded trials: 13.9 vs 0 rotations, 1/5 vs 5/5 failures, 57 vs 11 s held. **But** C5 also had 6 safety stops in unrecorded attempts: 5 of 11 C5 attempts completed. | Speed is clear. Reliability is mixed. |

**Main sim-to-real findings:**
1. **Rotation speed transfers.** C5 is the fastest, as in simulation.
2. **The retention reward does not.** R1 and P0 had the best retention in simulation, but on the hand they lose the cube within about 10 s every time.
3. **C5 trips the current guard often.** Its fast, forceful finger motion, already flagged as a sim-to-real risk, produced 6 safety stops in 11 attempts.
4. **The cube-rotation estimate does not match what the operator saw.** Its output is anti-correlated with the counts (Spearman −0.42).
5. **The force estimate agrees with motor current only moderately**, best for the thumb (median r = +0.39).

## 1. Setup

- **Policies** (all use 320 proprioceptive inputs, no estimator in the actor; kit-format folders verified against the training actors to 1.5e-5):

  | Name | Policy | Source |
  |---|---|---|
  | A3 | baseline A, seed 3, trained inside the kit limits | `experiments/hwlimit_ac` |
  | R1 | A + retention reward | `experiments/retention_v1` |
  | P0 | R1 without the position part of the retention penalty | `experiments/grasp_rotation_v2` |
  | C5 | R1 with the yaw-reward clip at ±0.5 rad/s | `experiments/grasp_rotation_v2` |
  | M400 | the professor's original `model_400` | trained without the kit limits; reference only |

- **Simulation reference** (held-out, 30 s): turns A3 1.33, C5 1.87, R1 1.16, P0 1.27; failure 24 / 20 / 19 / 12%. Simulated failure includes the tilt/drift rule. A closed-loop rehearsal through the kit runtime came within 0.06 turns of these.
- **Protocol** (`docs/hardware_trials_A3_C5_R1_P0.md`):
  - 60 s trials in a balanced Latin-square order, 5 blocks;
  - fixed cube placement (`docs/figures/cube_placement.png`, training grasp 49);
  - H1 gains, 350 mA current limit, abort at 450 mA, USB latency timer 1 ms;
  - after each run the operator recorded the rotations seen, drop and drop time, any help, and notes.
- **Timing:** mean 4.9% of steps over 60 ms (max 8.1%), down from about 20% on 18 Nov after the latency timer was fixed.
- **Hardware load:** peak current 422 mA, peak temperature 29 °C.

## 2. Results: recorded trials

From `LEAP_Hand_RL_Deploy_Kit/logs/trials.csv` and `logs/trial_comparison.md`. Rotations are the operator's count until the cube dropped or the run stopped.

| Policy | Trials | Rotations per trial (mean ± s.d.) | Drops | Safety stops | Failure (drop or stop) | Mean time held [s] |
|---|---|---|---|---|---|---|
| A3 | 5 | 2.0 ± 1.4 (n = 4) | 2/5 | 0 | 40% | 40.6 |
| **C5** | 5 | **13.9 ± 2.2** | 1/5 (at 45 s) | 0 | 20% | **57.3** |
| R1 | 5 | 2.6 ± 1.3 | 5/5 (at 5–15 s) | 0 | 100% | 9.2 |
| P0 | 5 | 0.0 | 5/5 (at 4–30 s) | 2 (current 452 and 462 mA) | 100% | 10.8 |
| M400 | 4 | 2.0 ± 0.8 | 4/4 (at 7–25 s) | 0 | 100% | 13.5 |

**Notes:**
- **A3:** in 2 of its 5 trials the cube **tumbled about a horizontal axis** ("rotating down to top") instead of spinning about the vertical axis. These are counted as 0 or no vertical rotations.
- **Speed measure changed after seeing the data.** The pre-specified rate (turns per minute while held) is not meaningful when the cube drops after a few seconds. It gave R1 an inflated 18/min from a few seconds of holding. Rotations per trial is used instead; the rate for trials held at least 20 s is in `trial_comparison.md`.
- **Counting unit unknown.** C5's 11–17 per 60 s is about 4× the simulation's C5 speed (about 3.7 turns per 60 s). If the operator counted quarter turns (faces passing), the real speed is about 3.5 turns per 60 s, close to the simulation. Comparisons between policies are unaffected as long as the same unit was used throughout.

## 3. All attempts, including those not recorded as trials

The logs contain 13 runs that were not entered in the trial record: 10 safety stops and 3 runs the operator interrupted. Leaving out failed attempts favours the policies that failed most, so the reliability picture must include them.

| Policy | Attempts | Completed 60 s without drop | Drops | Safety stops (all attempts) | Interrupted, not recorded |
|---|---|---|---|---|---|
| A3 | 7 | 3 | 2 | 1 (549 mA at 1.6 s) | 1 |
| C5 | 11 | 4 | 1 | **6** (tracking error 0.68 rad at 4.7 s; current 453–747 mA at 3–16 s) | 0 |
| R1 | 6 | 0 | 5 | 1 (810 mA at 1.0 s) | 0 |
| P0 | 9 | 0 | 5 | 4 (452–1048 mA at 0.7–30 s) | 2 |
| M400 | 4 | 0 | 4 | 0 | 0 |

- **Where the stops come from:** most trips happen in the first seconds of the policy, while the fingers move from the closing ramp to the policy's first actions with the cube in contact. Brief spikes reach 450–1050 mA even though the motors are limited to 350 mA.
- **C5:** the one tracking-error stop was the index knuckle jamming against the cube.
- **Including the attempts:**
  - **C5's reliability falls from 4/5 to 4/11** completed runs. When it runs, it holds the cube and spins it fast, but it often triggers the guard first.
  - **A3 is the most consistent:** 3/7 full runs and 1 stop.

## 4. Estimator monitor vs physical observation

**Cube rotation (estimator B, integrated yaw): not grounded.**

| Policy | Operator's counts | Estimated turns |
|---|---|---|
| C5 | 11–17 | −1.1 to −2.9 (opposite sign, ~10× smaller) |
| A3, R1 | 0–5 | 0.0 to +0.5 |
| P0 | 0 every trial | −2.5 to +0.7, still "rotating" after the drop |
| M400 | 1–3 | +0.2 to +1.6 |

- Across 23 counted trials: Spearman(observed, estimated) = **−0.42**.
- The estimate follows finger motion, not the cube. This matches the simulation audit, where B is about 1.5× worse on the new gaits (`hwlimit_ac` README §8.4), plus the additional sim-to-real shift.

**Fingertip force vs motor current: partial agreement.**
- **Thumb:** median r(F_est, current) = +0.39, positive in 83% of trials.
- **Ring:** +0.18 (79% of trials).
- **Index:** +0.17 (62%).
- **Middle:** +0.11 (75%).
- In most policy and finger combinations, current is higher when the estimator reports contact; e.g. C5 ring 327 vs 160 mA, R1 ring 345 vs 167 mA.
- Per-run files: `logs/physical_20261120_*_analysis.json`; summary: `logs/force_current_summary_20261120.json`.
- **This is indirect.** Current also reflects gravity, friction and tracking effort, and after a drop the hand moves without a cube. No force sensor was used.

## 5. Possible explanations and limits

- **Retention reward failing on the hand (R1, P0):**
  - **Start condition:** drops cluster within about 10 s. The kit closes to its own default grasp pose, not the training grasp's finger angles, so this could be a start-condition effect that hurts these policies more.
  - **Hardware limits:** the hand's current limit (350 mA) and lag (about 100 ms, against 25–75 ms in training) are not modelled in the retention studies.
- **C5's guard trips:** its 18% faster finger motion (simulation audit) meets the real current limit. The trips are evidence of that sim-to-real risk.
- **Small samples:** n = 4–5 recorded trials per policy; p-values and CIs are indicative only.
- **Operator effects:** counts are by eye with no video, and the unit is uncertain. Placement was by hand, and the operator also judged the drop times.
- **One seed per policy:** seed variance in simulation was larger than some of the policy differences.

## 6. Next steps

1. **Confirm the counting unit and direction**, and video every trial.
2. **Separate the start effect from the policy:** re-test R1 and P0 with the kit closing to the training grasp's finger angles (grasp 49). That is a kit start-pose change; it needs the TA's approval.
3. **Fix C5's guard trips:** retrain or fine-tune with an effort or current cost and the real lag, or test a slower ramp into the policy.
4. **Ground truth for the estimators:**
   - camera tracking of the cube (ArUco / RealSense) for the cube estimate;
   - retraining B on the new gaits, as recommended in `hwlimit_ac`;
   - a fingertip force sensor for the force estimator.

## Files

| Path | Contents |
|---|---|
| `LEAP_Hand_RL_Deploy_Kit/logs/trials.csv` | operator record, 24 trials |
| `logs/trial_comparison.md`, `logs/trial_comparison.json` | Q1–Q4 tables |
| `logs/physical_20261120_*.npz` | all 37 runs: recorded trials and unrecorded attempts |
| `logs/physical_20261120_*_analysis.json` | per-run comparison of estimated forces with motor current |
| `docs/hardware_trials_A3_C5_R1_P0.md` | protocol |
| `docs/figures/cube_placement.png` | cube placement and expected rotation direction |
| `kit/trials.py` | schedule, recorder, analysis |
| `training_source/project/experiments/grasp_rotation_v2/deploy/` | kit folders for P0 and C5 |
| `grasp_rotation_v2/results/deploy_verification.json`, `grasp_rotation_v2/results/deploy_rehearsal.json` | offline checks and simulation rehearsal |
