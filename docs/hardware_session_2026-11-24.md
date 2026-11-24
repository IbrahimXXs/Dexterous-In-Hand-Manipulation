# LEAP Hand hardware session, 24 Nov 2026: C5 seed 1 vs C5 seed 3

**Question:** does C5 seed 3 avoid the current-guard stops (index jams) that stopped C5 seed 1 on 20 Nov, without losing speed?

**Answer:**
- **Guard stops: yes.**
  - Today: C5 seed 3 had **0 safety stops in 13 attempts**; seed 1 had **3 in 5** (Fisher p = 0.012).
  - Including 20 Nov: seed 1 tripped a guard in **9 of 16** attempts, seed 3 in 0 of 13 (p = 0.001).
- **Speed: yes, when it holds the cube.** While held for ≥ 20 s, seed 3 turned about 18 quarter turns/min (≈ 4.4 revolutions/min), against 13 for seed 1.
- **Retention: no.** Seed 3 dropped the cube in 11 of 12 valid trials, but the outcome is **bimodal**:
  - early drops at 5–7 s (6 trials);
  - long holds of 45–60 s (5 trials), where it rotates well.

  The operator attributes the early drops to the cube's start placement.

## Setup

- **Candidate choice:** no "low-jam" C5 has been trained; C5-HW-V1 is a specification only. Seed 3 is the C5 recipe with a different training seed. It was chosen **before** the hardware session, by a simulation screen with the motors capped at the deployed 350 mA (`experiments/c5_hw_v1/results/c5_seed_screen.json`):

  | | Index saturated | Environments with the index stuck ≥ 1 s | Turns per 30 s |
  |---|---|---|---|
  | Seed 1 | 7.5% | 2.7% | 2.45 |
  | Seed 3 | 3.6% | 0.4% | 2.23 |
  | Seed 2 | 1.7% | 0% | not tested on the hand: 42% of environments have the *middle* finger stuck ≥ 1 s |

- **Kit folders:** `policies/C5` (seed 1) and `policies/C5s3`. Kit = training to ≤ 1.9e-5. The kit rehearsal in simulation costs seed 3 −0.17 ± 0.12 turns (the PIP-limit gap) and seed 1 −0.03.
- **Protocol** (`docs/hardware_trials_C5_seeds_2026-11-24.md`):
  - ABBA order for trials 1–10, then 8 further C5s3 trials (11–18);
  - 60 s runs, the 20 Nov placement (`docs/figures/cube_placement.png`) and kit settings;
  - every attempt recorded;
  - rotations counted in **quarter turns**.
- **Hardware:** loop jitter 3–6% of steps > 60 ms. Guard trips at 461–887 mA (spikes; the motors are limited to 350 mA).
- **Correction:** trial 15 was saved as "no drop", but the operator's note says the cube went out at 50 s. It is corrected to drop = yes at 50 s, and the correction is marked in the record.

## Results (`LEAP_Hand_RL_Deploy_Kit/logs/trials_20261124.csv`, `trials_20261124_comparison.md`)

| # | Policy | Ran [s] | Ended | Quarter turns | Drop (time) | Notes |
|---|---|---|---|---|---|---|
| 1 | C5s3 | 60.3 | full | 4 | 15 s | |
| 2 | C5 | 13.5 | **guard 887 mA** | 1 | 8 s | |
| 3 | C5 | 13.9 | **guard 461 mA** | 1 | 8 s | |
| 4 | C5s3 | 60.4 | full | **20** | no | |
| 5 | C5s3 | 9.6 | stopped after drop | 2 | 7 s | |
| 6 | C5 | 2.8 | **guard 722 mA** | – | 3 s | |
| 7 | C5 | 34.1 | stopped after drop | 7 | 30 s | |
| 8 | C5s3 | 7.4 | stopped | 1 | – | operator intervened; excluded |
| 9 | C5s3 | 17.2 | stopped after drop | 2 | 7 s | "OUT" |
| 10 | C5 | 60.4 | full | 12 | no | |
| 11 | C5s3 | 60.4 | full | 12 | 45 s | |
| 12 | C5s3 | 10.9 | stopped after drop | 3 | 7 s | "OUT" |
| 13 | C5s3 | 20.8 | stopped after drop | 7 | 20 s | |
| 14 | C5s3 | 7.5 | stopped after drop | 2 | 5 s | "OUT" |
| 15 | C5s3 | 60.3 | full | 14 | 50 s (corrected) | |
| 16 | C5s3 | 10.0 | stopped after drop | 2 | 5 s | "initial placement is a bit off" |
| 17 | C5s3 | 52.1 | stopped after drop | 12 | 50 s | |
| 18 | C5s3 | 8.4 | stopped after drop | 2 | 5 s | "OUT" |

| | C5 seed 1 | C5 seed 3 |
|---|---|---|
| Attempts (valid) | 5 (5) | 13 (12) |
| **Safety stops** | **3/5** | **0/13** |
| Drops | 4/5 | 11/12 |
| Held ≥ 45 s | 1/5 | 5/12 (trials 4, 11, 15, 17, and 4 at 60 s) |
| Early drop ≤ 7 s | 1/5 (guard stop) | 6/12 |
| Quarter turns per trial (mean ± s.d.) | 5.2 ± 5.3 (n = 4) | 6.8 ± 6.2 |
| Quarter turns/min while held ≥ 20 s | 13.0 (n = 2) | 17.6 (n = 5) |

**Same jam as 20 Nov:** in seed 1's guard stops, the **index MCP/PIP** joints are at ≥ 330 mA for 35–72% of the last 2 s, 0.2–0.3 rad behind their commands (trials 2 and 3). Trial 6 tripped early (2.8 s), with lighter, mixed index and thumb-abduction load. Seed 3 shows no such stop.

## Start placement

- **What the operator saw:** seed 3's early drops at 5–7 s ("OUT"; trial 16: "initial placement is a bit off") contrast with long holds when the start was good.
- **What the logs can show:** nothing. The hand pose right after the closing ramp is **identical** in early-drop and long-hold trials (every joint within 0.008 rad), and so is the estimated cube position at the start.
- **What that implies:** the kit's default grasp pose barely constrains the cube, so the fingers stop at the same angles wherever the cube sits. The placement is invisible to the policy at the start, and a few millimetres decide whether the first motions keep the cube.
- **Simulation agrees:** with the kit's default pose and grasp 49, only the middle and ring finger bases touch the cube, by 1–2 mm (17 Nov placement analysis). The GRV2 study found that the start grasp alone changes simulated failure from about 24% to about 1%.

## Interpretation and limits

- **Supported:** seed 3 does not trip the current guard on the hand (0/13), unlike seed 1 (9/16 over two sessions). This matches its lower index loading in the 350 mA simulation screen. It is a seed difference, not a design change.
- **Supported, small n:** when seed 3 keeps the cube past the start, it rotates at least as fast as seed 1 (about 4.4 revolutions/min while held).
- **Not supported:** better retention. Early drops dominate, and they are attributed to placement, which cannot be checked objectively from the logs.
- **Limits:** one session, one operator, visual counts; only 5 seed-1 attempts today; the placement quality of each trial was not measured.

## Next

1. **Make the start repeatable:** a placement jig or template on the palm, or a kit start pose that grips the cube, such as grasp 49's finger angles; that needs approval. Placement currently decides retention more than the policy does.
2. **Record placement per trial** (a photo before each run), so early drops can be attributed objectively.
3. **Train C5-HW-V1** (`experiments/c5_hw_v1/SPEC.md`) to remove the jam by design instead of relying on seed variance.
