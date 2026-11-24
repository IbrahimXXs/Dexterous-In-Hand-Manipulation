# LEAP Hand hardware session, 24 Nov 2026: C5 seed 1 vs C5 seed 3

**Question:** does C5 seed 3 avoid the current-guard stops (index jams) that stopped C5 seed 1 on 20 Nov, without losing speed?

**Answer (pilot, n = 5 each):**
- Seed 3 had **no safety stops in 5 attempts**; seed 1 had **3 in 5** (Fisher p = 0.17).
- Including 20 Nov, seed 1 tripped a guard in **9 of 16** attempts against 0 of 5 for seed 3 (p = 0.045).
- **Drops remain frequent for both:** 3 of 4 for seed 3, 4 of 5 for seed 1. The operator reports that some cubes were placed incorrectly; which trials is not recorded.

## Setup

- **Candidate choice:** no "low-jam" C5 has been trained; C5-HW-V1 is a specification only. Seed 3 is the C5 recipe with a different training seed. It was chosen **before** the hardware session, by a simulation screen with the motors capped at the deployed 350 mA (`experiments/c5_hw_v1/results/c5_seed_screen.json`):

  | | Index saturated | Environments with the index stuck ≥ 1 s | Turns per 30 s |
  |---|---|---|---|
  | Seed 1 | 7.5% | 2.7% | 2.45 |
  | Seed 3 | 3.6% | 0.4% | 2.23 |
  | Seed 2 | 1.7% | 0% | not tested on the hand: 42% of environments have the *middle* finger stuck ≥ 1 s |

- **Kit folders:** `policies/C5` (seed 1) and `policies/C5s3`. Kit = training to ≤ 1.9e-5. The kit rehearsal in simulation costs seed 3 −0.17 ± 0.12 turns (the PIP-limit gap) and seed 1 −0.03.
- **Protocol** (`docs/hardware_trials_C5_seeds_2026-11-24.md`):
  - ABBA order (C5s3, C5, C5, C5s3, …), 60 s trials;
  - the 20 Nov placement and kit settings;
  - every attempt recorded;
  - rotations counted in **quarter turns**.
- **Hardware:** loop jitter 3–6% of steps > 60 ms. Guard trips at 461–887 mA (spikes; the motors are limited to 350 mA).

## Results (`LEAP_Hand_RL_Deploy_Kit/logs/trials_20261124.csv`, `trials_20261124_comparison.md`)

| # | Policy | Ran [s] | Ended | Quarter turns | Drop (time) | Notes |
|---|---|---|---|---|---|---|
| 1 | C5s3 | 60.3 | full | 4 | yes (15 s) | |
| 2 | C5 | 13.5 | **guard 887 mA** | 1 | yes (8 s) | |
| 3 | C5 | 13.9 | **guard 461 mA** | 1 | yes (8 s) | |
| 4 | C5s3 | 60.4 | full | **20** | no | |
| 5 | C5s3 | 9.6 | stopped after drop | 2 | yes (7 s) | |
| 6 | C5 | 2.8 | **guard 722 mA** | – | yes (3 s) | |
| 7 | C5 | 34.1 | stopped after drop | 7 | yes (30 s) | |
| 8 | C5s3 | 7.4 | stopped | 1 | – | operator intervened; excluded |
| 9 | C5s3 | 17.2 | stopped after drop | 2 | yes (7 s) | |
| 10 | C5 | 60.4 | full | 12 | no | |

| | C5 seed 1 | C5 seed 3 |
|---|---|---|
| Attempts (valid) | 5 (5) | 5 (4) |
| **Safety stops** | **3/5** | **0/5** |
| Drops | 4/5 | 3/4 |
| Completed 60 s without a drop | 1/5 | 1/4 |
| Quarter turns per trial (mean) | 5.2 (n = 4) | 7.0 (n = 4) |
| Best run | 12 quarter turns = 3 revolutions / 60 s | 20 quarter turns = 5 revolutions / 60 s |

**Same failure as 20 Nov:** in the logs of seed 1's guard stops, the **index MCP/PIP** joints are at ≥ 330 mA for 35–72% of the last 2 s, 0.2–0.3 rad behind their commands. That is the sustained index jam found in the audit (trials 2 and 3). Trial 6 tripped early (2.8 s), with lighter, mixed load on the index and the thumb abduction joint.

## Interpretation and limits

- **Supported, tentatively:** seed 3 trips the current guard less than seed 1, consistent with its lower index loading in the 350 mA simulation screen. This is a seed difference, not a design change. Seed variance is the explanation until C5-HW-V1 is trained.
- **Not supported:** better retention or speed. Drops are as frequent for both, and the rotation difference is within noise.
- **Placement errors** in some trials (not identified) can cause early drops for either policy. They make the drop comparison unreliable, but they are unlikely to create the jam signature.
- n = 5 per policy, one operator, one session, counts by eye.

## Next

1. Repeat with **≥ 10 attempts each** and a placement check before each run (e.g. a photo); note any misplacement in the record.
2. Train **C5-HW-V1** (`experiments/c5_hw_v1/SPEC.md`) to target the jam directly rather than relying on seed variance.
3. Look into the early drops (5–15 s) seen for every policy since 20 Nov: start pose and placement, current limit.
