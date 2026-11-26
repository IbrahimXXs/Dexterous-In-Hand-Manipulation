# C5-HW-V2: single-seed protocol (frozen 2026-11-26, before training)

This protocol fixes how the one C5-HW-V2 run is trained, evaluated and judged. It was written and hashed (`logs/frozen.sha256`) **before** the run started and before any result of it existed. Steps A and B (`FIT_PLAN.md`, hashed earlier in `logs/fit_plan.sha256`) were evaluation-only and used frozen policies; their results are summarised in "Inputs" below. Style and statistics follow `c5_hw_v1/PROTOCOL_EFFORT_V1.md`.

## Scope (instruction of 2026-11-26)

- **Exactly one training run**, PPO seed 1. It is a **single-seed exploratory experiment** and cannot show training-seed reproducibility.
- **Confidence intervals are bootstraps over evaluation environments.** They describe evaluation variation for this one trained model, not variation between training runs.
- **Goal:** one policy trained under the real hand's conditions, aimed at the hardware failure modes. These are jams at 350 mA, a simulated actuator ~3.5× slower than the hand, and placement-caused early drops.

## Inputs fixed before training (from steps A and B)

**Step A, response fit (`results/response_fit.json`, `.md`): FAILED (tracking).**
- **Grid:** frozen C5_1, 24 settings of delay × damping × friction loss.
- **Tracking:** no setting meets the pre-registered tracking band. Simulated median tracking error is 2.81–3.82× the hardware's (band 0.70–1.30×).
- **Lag:** the best-lag setting (delay 0–25 ms, damping ×0.5, friction ×0.5) gives a median simulated lag of 218 ms against the hardware's 79 ms (median per-joint error 134 ms), with 0 of 16 joints within 30 ms.
- **Sensitivity:** the same grid at the 350 mA-equivalent limit also fails.
- **Free-air cross-check:** functest stage 6, no cube. Hardware median lag 95 ms; simulation 364 ms at the current setting and 275 ms at the best-lag setting. The mismatch is in the simulated actuator, not in contact.
- **Consequence (pre-registered rule):** the response change is **not** applied. C5-HW-V2 addresses the jam and placement failure modes only; the ~3.5× response mismatch remains. A structural actuator-model change is out of scope here.

**Step B, kit-start resets (`cache/kit_start_cache.npz`, sha256 `10b198029f50b780…`; record `cache/kit_start_cache.json`).**
- **Starts:** 4,035 of 4,096 simulated kit starts kept.
  - **Procedure:** home pose; cube at grasp 49's position + 10 mm, flat, at grasp 49's yaw, ±4 mm x/y and ±15° yaw noise; 1 s settle; 5 s linear closing ramp to the policy default pose at the 0.167–0.204 Nm effort range.
  - **Stored joint target:** the measured post-ramp pose, as the kit resets it.
- **Recorded deviation (made before any policy evaluation):** the planned "≥ 2 fingertip contacts" filter was not applied. The reference grasp-49 state itself shows 0–1 fingertip-sensor contacts while held, and the filter rejected 100% of starts.
- **Validation against the hardware:** the simulated post-ramp joint pose differs from the hardware's (median `q_hw[0]` of 35 C5-family run logs) by a median of 0.068 rad and at most 0.154 rad (if_pip). In simulation the fingers stop less flexed.
- **Bar to beat** (`results/kit_start_survival.md`; validation seed 7001, 2048 environments, 30 s, effort actuator). Stop rule B not triggered.

  | Policy | Survival from kit starts | Survival from grasp 49 |
  |---|---|---|
  | C5_1 | 0.955 | 1.000 |
  | C5_2 | 0.958 | 0.999 |
  | C5_3 | 0.953 | 0.999 |
  | C5-EFFORT | 0.930 | 0.987 |

- **Limitation, stated before training:** in simulation the kit start costs C5 only about 4–5 percentage points of 30-s survival. On the real hand, C5s3 dropped the cube at 5–7 s in 6 of 12 valid trials on 24 Nov. The simulated kit start does not reproduce the real placement failure, so a simulated gain here is weak evidence about the hand.

## Condition

The task is `Mjlab-Leap-C5HW2` (`src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_right_c5_hw_v2/`): C5-HW-EFFORT, unchanged, plus one change.
- **Kept from C5-EFFORT:** GRV2 C5; effort range 0.167–0.204 Nm; utilisation penalty with w = 0.2201, constant.
- **Change: kit-start resets** (`apply_kit_start_resets`). Every training episode starts from the step-B cache (reset event `reset_from_grasp_cache`, joint state and stored command restored).
- **Response:** `apply_response` exists but `RESPONSE = None`, per step A.
- **Test:** `tests/test_c5_hw_v2.py` checks with a full config-tree diff that:
  - the task differs from `Mjlab-Leap-C5HW-EFFORT` by exactly the two reset-cache leaves (`events.reset_from_grasp_cache.params.cache_file`, `actions.joint_pos.required_grasp_cache`);
  - the PPO config differs only in the experiment name;
  - the frozen `RESPONSE` equals the fit's outcome;
  - episodes start from the cache, with the measured pose as the command.

## Training

- **Settings:** PPO seed 1, 4096 environments, 3001 updates (0..3000), from scratch, never initialised from C5 or C5-EFFORT.
- **PPO config:** zero-initialised actor output, the hwlimit A PPO configuration (C5's and C5-EFFORT's).
- **Command** (`scripts/sbatch_train.sh`; preflight checks `logs/frozen.sha256` and the grasp cache):

```bash
cd training_source/project && SEED=1 sbatch experiments/c5_hw_v2/scripts/sbatch_train.sh
# runs: scripts/train.py Mjlab-Leap-C5HW2 --env.scene.num-envs 4096 --agent.seed 1 --agent.run-name c5hw2_seed1 --zero-actor-output True
```

## Validation, selection and stage check

- **Validation:** `scripts/curves.py`, a copy of C5-EFFORT's, with only the run, task and log paths changed.
  - Seed 7001, nominal preset, 1024 environments, 30 s, a point every 100 updates, a 60-s point every 500.
  - Runs in the run's own task, `Mjlab-Leap-C5HW2` (kit starts).
- **Selection:** max 30-s validation turns; ties go to the later checkpoint.
- **Stage check at update 500.** Stop only for:
  - a validation 30-s yaw rate below 0.132 rad/s (50% of the C5 3-seed mean at 500, as in PROTOCOL_EFFORT_V1);
  - a reward exploit: validation turns ≤ 0.1, or the hand stops moving;
  - NaN, numerical instability or a collapse of the training reward.

  The validation starts differ from C5's (kit starts, not the grasp cache); this is noted, and the threshold is unchanged.

A stopped or failed run is recorded, not removed.

## Held-out evaluation (seeds 6201 and 6202, used once)

The rollout and metrics are those of C5-HW-EFFORT-V1 (`c5_hw_v1/scripts/eval_items.py`: `rollout`, `summarise`, jam proxy), imported unchanged by `scripts/eval_items.py`.
- 60-s rollouts (1200 steps), 2048 environments per seed, first episode only.

**Policies (selected checkpoints):**
- A1–A3;
- R1;
- C5_1–C5_3 (the C5 baseline is the per-environment mean of the three seeds);
- EFFORT_1;
- C5HW2_1.

**Settings:**

| Setting | Seeds | Presets | Source of the baseline rollouts |
|---|---|---|---|
| A, full 239-grasp distribution | 6201, 6202 | nominal; heavy_slippery, high_delay, obs_noise_x3 on 6201 | C5-HW-EFFORT-V1 items (same code, tasks and seeds); C5HW2_1 new |
| B, grasp 49 | 6201, 6202 | nominal | as above |
| C, robust 10-grasp subset | 6201, 6202 | nominal | as above |
| **K, kit start** (environment i → cache entry i mod 4035) | 6201, 6202 | nominal | **new for every policy** |

**Actuators:**
- **default**, task `Mjlab-Leap-RetV1-A`;
- **effort-matched**, task `Mjlab-Leap-C5HW-EFFORT`, the 350 mA-equivalent limit.
- **Not available:** the "hardware-matched" actuator of SPEC §6, because the response fit failed (step A). No arbitrary response setting is substituted.

**Item lists:** `items/heldout_baselines_K.json` and `items/heldout_new.json` (`scripts/eval_items.py --make-items`).

## Decision

- **Basis:** effort-matched actuator; seeds 6201 and 6202 pooled; nominal preset.
- **Computed by:** `scripts/compare_v2.py decision` → `results/v2_comparison.{json,md}`.

**Success requires all three** (the instruction's criteria, made explicit before training):
1. **Kit-start survival clearly above C5 and C5-EFFORT.**
   - **Measure:** setting K, 30-s failure (1 − survival).
   - **Threshold:** ≤ 0.5 × the C5 3-seed mean's **and** ≤ 0.5 × C5-EFFORT's.
   - **CIs:** both paired-bootstrap 95% CIs of the survival difference (new − C5 mean, new − C5-EFFORT) have lower bound > 0.
   - **Why "halve the failures":** the step-B survival of the frozen policies is already 0.93–0.96, so an absolute margin like +0.05 is unreachable. Halving the remaining failures is the pre-registered meaning of "clearly".
2. **Jam proxy ≤ C5-EFFORT.** Setting A, 60-s jam proxy (share of environments with ≥ 1 s index jam, the audit's detector), point estimate. The CI and the setting-K value are reported.
3. **Rotation.** Setting A, 30-s turns ≥ 0.85 × the C5 3-seed mean (point estimate; CI reported).

**Verdicts:**
- **Pass:** all three hold → **"single-seed pass (exploratory)"**. That is the strongest possible verdict, never "confirmed".
- **Fail:** any one fails → **"fail (single-seed exploratory)"**, naming the criteria that failed.

Everything else in `v2_comparison.md` (other settings, actuators, presets, per-policy rows) is descriptive.

**How the CI is computed:** paired bootstrap over evaluation environments. 10,000 resamples of environment indices within each evaluation seed, the same for all policies, since equal seeds give equal reset draws.

## Additional reports (descriptive, not criteria)

- **350 mA jam screen** (`c5_hw_v1/scripts/seed_screen.py` procedure: capped motors, grasp 49, 256 environments, 30 s) of C5HW2_1, next to C5 seeds 1–3 and C5-EFFORT. Seed differences decided the jam behaviour on the real hand (C5 seed 1 vs 3).
- **Deploy candidate.** Built in a new directory (`deploy/`), never in `LEAP_Hand_RL_Deploy_Kit/policies/`:
  - kit-format export with offline verification (kit = training to ≤ 3e-5);
  - a closed-loop kit rehearsal in simulation (kit reset "measured" and "target").

  It is labelled "ready for supervisor review" only after a single-seed pass, otherwise "not recommended". Simulation results are not hardware-safety evidence.
