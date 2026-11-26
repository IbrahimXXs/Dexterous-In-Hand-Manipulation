# Gait efficiency: analysis plan for the simulation half

**Status:** written and hashed (`logs/analysis_plan.sha256`, `logs/analysis_plan_hashed_at.txt`) **before any simulation rollout of this study**. Simulation only: no training, no hardware.

**What is and is not pre-registered.** The real-hand numbers (`results/real_hand.json`, `results/real_hand.md`, from `scripts/real_hand.py`) were computed **before** this plan was written. Their sha256 at the time of writing is recorded in `logs/real_hand_at_plan.sha256`. The real-hand side is therefore exploratory and post hoc. Two things in this plan were chosen after seeing those numbers: which real-hand window is compared with simulation (below), and the interpretation thresholds. Only the simulation half (settings, seeds, metrics and how sim and real are compared) is fixed here before its data exist.

## 1. Question

Rotation speed while the cube is held factorises exactly as

> quarter turns per second = finger cycles per second × quarter turns per finger cycle.

On the real hand the joint encoders give the cycles, and the operator's counts give the quarter turns. In simulation, the same cycle counter runs on the simulated joint positions, and the simulator's cube yaw gives the quarter turns. For the policies that ran on the hand, which factor changed between simulation and the hand?

## 2. Cycle counter (identical on both sides)

`scripts/gait.py::count_cycles`, unchanged from the version that produced `results/real_hand.json`:
- joint positions at the policy rate (20 Hz); the first 3 s dropped;
- centred and projected on the first principal component, fitted per run or environment;
- band-pass 0.12–1.2 Hz, 2nd-order Butterworth, `filtfilt`; Hilbert phase;
- cycles = |unwrapped phase change| / 2π;
- rhythm = the largest FFT peak above 0.12 Hz of the centred PC1 (Hann window, zero-padded to ≥ 4096 samples);
- stroke amplitude = the median per-cycle peak-to-peak of the band-passed PC1 (rad).

## 3. Policies (same checkpoints as on the hand)

`scripts/sim_gait.py::check_kit_identity` asserts sha256 equality with `LEAP_Hand_RL_Deploy_Kit/policies/<name>/model_*.pt`.

| Label | Checkpoint | Kit folder |
|---|---|---|
| M400 | `checkpoint/model_400.pt` (run with the action squash it was trained and deployed with: **clip**) | `M400` |
| A3 | `hwlimit_ac/checkpoints/A_seed3_selected_model_2600.pt` | `A3` |
| R1 | `retention_v1/checkpoints/R1_selected_model_2600.pt` | `R1` |
| P0 | `grasp_rotation_v2/checkpoints/P0_1_selected_model_1500.pt` | `P0` |
| C5s1 | `grasp_rotation_v2/checkpoints/C5_1_selected_model_2400.pt` | `C5` |
| C5s3 | `grasp_rotation_v2/checkpoints/C5_3_selected_model_3000.pt` | `C5s3` |
| EFFORT_1, C5HW2_1 (optional extras, simulation only) | `c5_hw_v1/…/EFFORT_1_selected_model_2900.pt`, `c5_hw_v2/…/C5HW2_1_selected_model_3000.pt` | – |

## 4. Simulation settings

- **Harness:** `c5_hw_v1/scripts/eval_items.py::rollout` under `c5_hw_v2/scripts/eval_items.py::setup`, both unchanged.
  - That is GRV2's `make_env`: first episode only, timeout beyond the horizon, frozen penalty weights, nominal preset, real (approved) joint limits.
  - `scripts/sim_gait.py` only wraps the environment constructor, as `c5_hw_v2` does. The wrapper records `robot.data.joint_pos` after every control step and sets model_400's squash to "clip".
- **Primary setting:** **K**, the c5_hw_v2 kit-start cache (`c5_hw_v2/cache/kit_start_cache.npz`, environment *i* → entry *i* mod 4,035), with the **350 mA-equivalent effort actuator** (`Mjlab-Leap-C5HW-EFFORT`). This is the closest simulated match to how the policies ran on the hand.
- **Reference setting:** **A**, random grasps from the 239-grasp cache, with the **default actuator** (`Mjlab-Leap-RetV1-A`). These are the conditions behind the earlier simulation numbers. Descriptive only.
- **Seeds:** 9101 and 9102 (new; not 5001/5002, 6201/6202, 7001, 8301, 8401, 8501 or 9201). 256 environments per policy, setting and seed, so 512 per policy and setting.
- **Episode:** 60 s (1,200 control steps).
- **Logged per step:** joint positions and the rollout's genuine cube-yaw increment. Unwrapped yaw is its cumulative sum.

## 5. Per-environment quantities (simulation)

- **Held:** samples before the first termination (the drop); the terminal step is excluded. An environment never terminated is held for 60 s.
- **Cycles:** `count_cycles` on the window from 3 s to the last held sample.
- **Quarter turns:** 4 × |Σ yaw increments over the whole held time| / 2π. This mirrors the operator, who counts from the policy start to the drop. Sensitivity: quarter turns over the counting window only.
- **Inclusion:** held ≥ 20 s, mirroring the real-hand rule. Sensitivity: held ≥ 10 s.

Per policy and setting (both seeds pooled):
- quarter turns per cycle as a ratio of sums, with a bootstrap 95% CI over environments (10,000 resamples);
- median rhythm;
- cycle rate (Σ cycles / Σ window seconds);
- quarter turns per held second (Σ quarter turns / Σ window seconds);
- median stroke amplitude;
- 60-s survival and the share of environments held ≥ 20 s.

## 6. What is compared with what

- **Real-hand comparison window:** variant `held` of `results/real_hand.json`, i.e. window 3 s to the operator's drop time or run end, runs held ≥ 20 s, no intervention, count present. It is the only real-hand window counted on the same rule as simulation, which counts only while the cube is held.
- Variant `run` (window to the end of the run, after the drop too; the procedure of the instruction) is reported next to it. It is not used for the transfer labels, because after a drop the hand keeps cycling with no cube, which lowers its quarter turns per cycle.
- R1 has no run held ≥ 20 s, so it gets no primary comparison. Its `run` and `held ≥ 10 s` values are reported as descriptive.
- **Simulation side of the comparison:** the primary setting (K, effort actuator).

**Ratios (real / sim), per policy:**
- *r_rhythm* = median FFT rhythm, real / sim;
- *r_rate* = cycle rate, real / sim;
- *r_tpc* = quarter turns per cycle, real / sim;
- *r_speed* = quarter turns per held second, real / sim.

Exactly, *r_speed* = *r_rate* × *r_tpc*.

## 7. Interpretation rules (fixed now)

1. **Rhythm transferred** if 0.75 ≤ *r_rhythm* ≤ 1.25. The cycle rate *r_rate* is reported next to it. If the two fall on different sides of a bound, both are stated and no label is given.
2. **Turns per cycle transferred** if 0.75 ≤ *r_tpc* ≤ 1.25.
3. **Speed decomposition**, only when *r_speed* is outside [0.75, 1.25]:
   - share of the speed change from rhythm = ln *r_rate* / ln *r_speed*;
   - share from turns per cycle = ln *r_tpc* / ln *r_speed*;
   - the two shares sum to 1 by construction.
   - Inside the band: "speed transferred"; no shares.
   - A policy with 0 real quarter turns (P0) has *r_tpc* = 0. It is labelled "no rotation on the hand", and its shares are not computed.
4. **Strokes or twitching.** A policy's real cycles are called *finger strokes* if its real median stroke amplitude is ≥ 0.2 rad **and** ≥ 50% of its simulated median stroke amplitude. Otherwise they are called *small/twitching motion*, and its turns per cycle is not interpreted as "turns per stroke".
   - 0.2 rad (about 11° along the joint-space PC1) is far above encoder resolution and loop jitter, and below any deliberate finger stroke.
5. **Why ±25%.** Each source of real-hand imprecision is ≤ ~16% (sources below), and the simulated speed of C5's three training seeds spans about ±5% (GRV2 `results/c5_seeds.md`: 1.685–1.869 turns per 30 s). A ratio outside ±25% is therefore unlikely to come from measurement imprecision alone. A ratio inside it cannot be distinguished from "unchanged" with these data.
   - Operator vs camera quarter turns, 25 Nov: 16 vs 16.17 and 8 vs 7.04, up to 12% (`LEAP_Hand_RL_Deploy_Kit/logs/trials_20261125_vision.csv`).
   - Camera slope vs operator-based turns per cycle, trial 1: 0.88 vs 16 / 17.5 = 0.91.
   - The bootstrap CI over runs of real turns per cycle: C5s1 0.73 [0.67, 0.80]; C5s3 1.08 [0.97, 1.25] (`results/real_hand.json`, variant `held`).
   - FFT resolution of a 57-s window: 0.018 Hz, about 6% of 0.3 Hz.
6. **Small n is stated, not tested.** Real-hand runs per policy in the comparison window: C5s3 7, C5s1 7, A3 2, M400 1, P0 1 (R1 0). There are no p-values for sim-vs-real. Simulation CIs are over environments and describe simulation variation only.
7. **Validation limit.** The cycle count is checked against the camera only for C5 seed 3 (25 Nov, trials 1 and 5). It is not presented as a validated turn counter for any other policy. The logs cannot separate slip, missed contact and back-and-forth rocking.

## 8. Outputs

- `results/items/<key>.json` and `data/<key>.npz` (git-ignored).
- `results/sim.json` (`scripts/sim_gait.py --analyse`).
- `results/decomposition.json`/`.md` (sim vs real, rules 1–4).
- `figures/gait_efficiency{,_dark}.png`.

## 9. Compute

32 rollouts: 8 policies × 2 settings × 2 seeds, 256 environments × 1,200 steps each. Two Slurm evaluation workers (`c5_hw_v2/scripts/sbatch_job.sh`, partition ws-ia, one GPU each). Run only after the user approves the GPU job.
