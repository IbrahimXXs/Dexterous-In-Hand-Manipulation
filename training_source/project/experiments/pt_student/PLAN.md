# Privileged teacher → transformer student (PT-student): pre-registered plan

**Status:** written 2026-11-27 before any training. It is hashed (sha256 + timestamp in `logs/`) and committed before the first training job.

**Scope:** simulation only. There is ONE training seed (seed 1) for everything trained here.

**Question.** Does the recipe of Yao et al. (arXiv 2605.21330, v1, May 2026; preprint) beat C5 in simulation, and which part is responsible? The recipe is a PPO teacher that sees the true cube state, distilled into a small transformer student that sees only joint history, with an auxiliary cube-state loss.

## 0. What the paper says (checked in the arXiv HTML, v1)

**Real ORCA hand** (3 trials of 60 s per policy, 20 Hz, joint position targets):
- 55 mm cube: proprioceptive PPO 3.83 ± 0.51 rev/min, 0.33 drops per trial; joint-sensor transformer student ("PT-Joint") 11.83 ± 0.52, 0 drops (Table I).

**Ablations** (Table III) are also real-hand on the 55 mm cube, but in the *motor-sensor* configuration ("PT-Motor"), not PT-Joint:

| Variant | rev/min | drops per trial |
|---|---|---|
| no reconstruction loss | 4.75 ± 0.75 | 5.0 ± 1.7 |
| MLP | 5.08 ± 0.38 | – |
| LSTM | 7.00 ± 0.25 | – |
| transformer | 9.33 ± 0.63 | – |

**The paper's recipe, and where this study differs:**

| Paper | This study (as instructed) |
|---|---|
| distillation is behaviour cloning (L2 to teacher actions) | DAgger with a β schedule |
| sinusoidal positional embedding | learned positional embedding |
| auxiliary targets: object position (0.5), clean joint positions (0.3), clean joint velocities (0.2) | cube position (0.5), cube orientation (0.3), clean joint positions (0.2) |
| inputs include joint velocities and a goal command | the same 320-D observation as C5 |
| teacher reward includes an action-rate penalty (−0.075), "critical for producing smooth motions that transfer to hardware" | the teacher keeps C5's reward exactly; the action-rate variant is optional (4.3) |
| teacher: 4-layer MLP, 10,000 PPO iterations | teacher: C5's network and C5's 3001 updates |

## 1. Teacher (Step 1; training seed 1)

**Environment.** The task `Mjlab-Leap-GRV2-C5` environment, unchanged:
- rewards, the ±0.5 rad/s yaw clip, curriculum and terminations;
- domain randomisation, observation noise and delay;
- resets from the 239-grasp hardware-limited cache;
- approved command limits, the tanh delta action (±1/24 rad per step), 20 Hz, 20-s episodes.

It is registered as `Mjlab-Leap-PT-Teacher` in a new package (`tasks/hand_cube/config/leap_right_pt_student/`). No existing task changes.

**What the teacher actor sees: 403-D** = observation groups `actor` + `critic`, concatenated and normalised by the actor's running normaliser:

| Group | Contents | Dims |
|---|---|---|
| `actor` | C5's observation: 10 frames × (16 noisy, delayed, biased joint positions relative to default + 16 previous commands) | 320 |
| `critic` | clean joint positions (relative), joint velocities, command − position error, previous command | 16 + 16 + 16 + 16 |
| `critic` | cube pose in the palm frame (position 3, quaternion 4), cube linear and angular velocity in the palm frame | 3 + 4 + 3 + 3 |
| `critic` | cube size, cube mass, cube centre-of-mass offset (body frame), cube friction coefficient | 1 + 1 + 3 + 1 |

- The critic already includes the randomised friction, cube mass and centre of mass, so no extra teacher-only inputs are added.
- The critic is unchanged and sees `critic` (83-D).

**PPO: C5's configuration** (`hwlimit_ppo_cfg("A")`):
- actor and critic 512-512-256 ELU, InstrumentedPPO, adaptive-KL LR from 1e-3;
- 32 steps per environment, 5 epochs × 4 minibatches, γ 0.99, λ 0.95, clip 0.2, entropy 0.003, init noise std 0.7;
- `clip_actions` None (tanh in the environment), 4096 environments, zero-initialised actor output;
- **3001 updates**, training seed 1, launched with `scripts/train.py` exactly as `grasp_rotation_v2/scripts/sbatch_train.sh` launches C5.

**Checkpoint:** the final `model_3000.pt` (no validation selection; the teacher is an upper bound and a labeller, never deployed).

## 2. Students (Step 2 main, Step 4 controls; distillation only, training seed 1)

**Input:** the SAME 320-D observation as C5 (group `actor`, with the simulator's noise and delay), so the kit's observation code is unchanged.
- **Tokens:** 10, one per frame: [16 joint positions, 16 previous commands] (the 320 vector is term-major: 10 × 16 positions, then 10 × 16 commands).
- **Normalisation:** per token feature (32), mean and std over all frames of the first DAgger rollout (β = 1, teacher-driven), then frozen. ε = 1e-2 as in rsl_rl.

**S-TF (main).**
- Linear embedding 32 → 256, plus a learned positional embedding (11 positions: CLS + 10 frames).
- Transformer encoder: 3 layers, 4 heads, width 256, feedforward 1024, GELU, pre-LayerNorm, dropout 0.
- **Heads from the CLS output:** action (16, raw output u; the environment applies tanh, exactly as C5), cube position (3), cube orientation (4), clean joint positions (16).
- The action head is zero-initialised. The auxiliary heads are used in training only, never as inputs.

**Auxiliary targets** (from the simulator, `cube_state` and `critic` groups):
- **Cube position:** palm-frame position standardised with the fixed scales of `scratch_study/features.py` (P_MEAN, P_STD; clamp ±5).
- **Cube orientation:** [sin 4ψ, cos 4ψ, t_x, t_y].
  - R* = the canonical palm-frame rotation (`scratch_study.features.canonical_rotation`, the cube symmetry closest to the palm frame).
  - ψ = atan2(R*₁₀, R*₀₀), the yaw about the palm normal.
  - (t_x, t_y) = the palm-plane components of the cube's canonical normal axis R*[:, 2] (the tilt).
  - **Why 4ψ** (an interpretation of "sin/cos of yaw"): a cube is symmetric under 90° turns, so joint data can determine yaw only modulo 90°. sin/cos of 4ψ is that quantity, and it is continuous.
- **Clean joint positions:** the critic's noise-free joint positions relative to default (16, rad), at the current step.

**Loss (per sample, each term a mean over its dimensions):**

L = MSE(tanh u_student, tanh u_teacher) + 0.5 MSE(position) + 0.3 MSE(orientation) + 0.2 MSE(clean joints)

- The action term compares executed (squashed) actions, the quantity the environment applies.
- The teacher label is the teacher's deterministic action (mean).

**DAgger.**
- **Environment:** the training environment of the teacher (= C5's: same domain randomisation, resets, noise, curriculum), 4096 environments, environment seed 1.
- **Each update:**
  1. A rollout of 32 steps. Per environment and step, the executed action is the teacher's with probability β and the student's otherwise.
  2. The teacher labels every visited state.
  3. The rollout joins a FIFO buffer of the last 4 rollouts (≤ 524,288 samples).
  4. One pass over the buffer in shuffled minibatches of 16,384.
- **β** = 1 → 0 linearly over the first 20% of updates (300 updates), then 0.
- **Budget:** **1500 updates** (196.6 M labelled environment steps).
- **Optimiser:** AdamW, lr 3e-4, weight decay 0.01, linear warm-up over the first 50 updates, gradient-norm clip 1.0, torch seed 1.
- **Checkpoint:** the final one (update 1500). A checkpoint every 100 updates, plus a log of losses, β and the environment's episode statistics.

**Controls (Step 4; same teacher, same data budget, same schedule):**
- **S-TF-noaux:** S-TF with the auxiliary weights set to 0.
- **S-MLP:** C5's actor architecture (320 → 512-512-256 ELU), with the same normalisation, the action head and the three auxiliary heads on the last hidden layer, and the same loss.

## 3. Evaluation (Step 3; harness unchanged)

**Harness:** `c5_hw_v1/scripts/eval_items.py::rollout` and `summarise` under `c5_hw_v2/scripts/eval_items.py::setup`. Only the policy loader is wrapped (the student and the teacher's 403-D observation), and the step is wrapped to record joint positions and the command target, as in `seed_jury/scripts/sim_jury.py`.

**Policies:**
- S-TF;
- the teacher T;
- C5s1, C5s2, C5s3: `grasp_rotation_v2/deploy/policies/C5_seed{1,2,3}`, sha256 `2a932dfe…`, `49579bdf…`, `c221af89…`; evaluation only.
- Step 4 adds S-TF-noaux and S-MLP.

**Conditions:**
- **K** (primary): kit-start cache, effort (350 mA-equivalent) actuator, nominal;
- **A:** default starts, default actuator, nominal;
- **heavy_slippery**, **high_delay**, **obs_noise_x3:** setting A, default actuator.

**Seeds:** new evaluation seeds **9501 and 9502**, 256 environments × 1200 steps (60 s) each, so 512 environments per policy and condition.

**Metrics:** the `seed_jury` definitions:
- survival at 30 s and 60 s;
- turns per 30 s (mean signed turns, first 600 steps);
- jam proxy (60 s);
- rhythm and quarter turns per cycle (`gait_efficiency/scripts/gait.py`, unchanged; environments held ≥ 20 s);
- mean |command change| per step.

**C5 reference:** "C5" is the mean over the three seeds of each cell's value. Every seed is also its own row.

### Pre-registered success (S-TF vs the C5 three-seed mean, condition K)

1. **Turns:** turns per 30 s ≥ 1.15 × C5 mean, **and** the 95% bootstrap CI of the ratio S-TF / C5 mean lies above 1. The CI uses 10,000 resamples of environments within each policy and evaluation seed, rng 20261130.
2. **Survival:** survival at 30 s ≥ C5 mean − 0.01.
3. **Stress:** mean survival at 60 s over the three stress presets ≥ the C5 mean's − 0.02.

**PASS** = 1 and 2 and 3.

**Stricter test, reported separately:** does S-TF beat every C5 seed on turns per 30 s, i.e. is the bootstrap CI of the ratio to each seed above 1?

**Expected direction, stated before any training.**
- The teacher clearly above C5 on turns and survival (it sees the cube).
- S-TF **not** passing. The most likely outcome is turns within 0.9–1.15 × C5, for three reasons:
  - our earlier proprioceptive estimator gave no established gain (REPORT §5);
  - C5 already uses the same 320-D history;
  - the paper's large gains are real-hand results, where robustness matters more than simulated speed.
- Attribution, if S-TF > C5: the auxiliary loss is the part that matters least in simulation.

## 4. Controls, real-log check, hand-test preparation

**4.1 Attribution (Step 4).** S-TF vs S-TF-noaux vs S-MLP, evaluated like Step 3. Descriptive (no test). Order: after the main result.

**4.2 Real-log check (Step 5; CPU; exploratory).**
- **Replay:** rebuild the observations of every real C5 seed 1 and seed 3 run with the kit's code, as in `seed_jury/scripts/real_replay.py`. The logged policy must reproduce its logged raw actions to ≤ 1e-4; otherwise this step is skipped.
- **Prediction:** feed the observations to S-TF and read its orientation head: 4ψ = atan2(sin, cos), unwrapped, ÷ 4 → cumulative yaw.
- **Comparison** on 25 Nov trials 1, 4 and 5, against the camera's cumulative quarter turns (`LEAP_Hand_RL_Deploy_Kit/logs/trials_20261125_trial{1,4,5}_vision.csv`), from the policy start to the operator's drop time (or the end):
  - Pearson r of the yaw rates (1-s centred slope, on the camera's sample times);
  - cumulative quarter turns. Signs are set so that each net rotation is positive, as in the gait camera check.
- **Reference:** C5's estimator B "follows finger motion rather than the cube" (REPORT §7.4).
- **Caveat:** these runs were driven by C5, not the student.

**4.3 Optional action-rate variant** (only if everything else is done; labelled as a variant):
- a teacher with an added action-rate term (L2 of the change in the squashed action), weight −0.075 as in the paper, all else equal;
- plus an S-TF distilled from it;
- the same evaluation.

**4.4 Hand-test preparation (Step 6), only if Step 3 PASSES.**
- `kit/policy_runtime.py` loads a "transformer" folder (weights + config, numpy inference).
- Existing policies give bit-identical actions (test); there is no guard, limit or `hand.env` change.
- CPU latency median and p99 (p99 must be ≤ 10 ms; the control period is 50 ms).
- Offline smoke test on replayed real observations.
- `docs/hardware_protocol_pt_student_DRAFT.md`: shadow run first; S-TF vs C5s3, ABBA × 5, camera on; primary camera rev/min while held; predictions hashed before.

If Step 3 fails, none of 4.4 is done.

## 5. Reporting

- **Results:** `results/sim.{json,md}`, `results/real_log.{json,md}`, figures in the README style (light and dark).
- **Labels:** every result is marked pre-registered or exploratory and **single training seed** (the student and the teacher); the C5 rows are three training seeds.
- **Deviations** from this plan are listed with reasons. The plan is not edited after hashing.
- **GPU hours** are summed from the job logs.
