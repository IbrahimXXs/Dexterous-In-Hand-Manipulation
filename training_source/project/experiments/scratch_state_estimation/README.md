# Sensorless cube-state estimation for LEAP Hand cube rotation: a from-scratch study

**Question.** Can explicit cube-state estimation from LEAP Hand proprioception alone improve continuous in-hand cube rotation, compared with proprioception-only reinforcement learning?

**Status: complete (2026-11-13).** Every stage ran to completion. Every number in this report comes from an executed experiment; the sources are in `results/` and `logs/`. The first C runs were invalid because of a bug; they were discarded and retrained (Section 1.8).

---

## 1. Stage 0: environment and PPO corrections

Every correction lives in new modules; the original tasks and checkpoints are untouched.

| Module | Contents |
|---|---|
| `src/in_hand_rotation_mjlab/scratch_study/geometry.py` | Physical tilt; genuine yaw increment |
| `src/in_hand_rotation_mjlab/scratch_study/mdp.py` | Corrected termination, reward, penalty, metrics, cube-state observation |
| `src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_right_scratch_study/` | Task registration, corrected environment, tanh action, PPO config |
| `src/in_hand_rotation_mjlab/scratch_study/ppo.py` | `InstrumentedPPO`: unchanged optimization, extra diagnostics |
| `src/in_hand_rotation_mjlab/scratch_study/{features,actors}.py` | Cube-state features and the C/D actors |

The study tasks are `Mjlab-Leap-Scratch-{A,C,D}`; the pilot-only variant is `Mjlab-Leap-Scratch-A-ClipAction`.

### 1.1 Reset bug (verified)

**Bug.** The original `cube_pose_deviation` termination stored the episode-initial cube pose inside its `reset()` hook. MJLab calls manager resets before `sim.forward()`, so that pose was the previous episode's final pose. New episodes were then compared against a stale anchor and terminated at once, cascading.

**Other terms with the same pattern:**
- The reward's drift gate was anchored at reset but re-anchored on the first post-reset step, so it was correct.
- The `rotation_progress`, `position_error` and `tilt_error` metrics and the joint `pose_diff_penalty` anchored only at reset.

**Fix, applied to every anchored term.** `reset()` only marks an environment. The anchor is taken on the term's next evaluation, after the first physics step of the new episode, so it always uses fresh kinematics. This also covers the runner's random initial episode lengths. The anchoring step itself never terminates.

### 1.2 Euler-angle tilt bug (verified)

**Bug.** The original tilt was `‖(wrap(roll−roll₀), wrap(pitch−pitch₀))‖` in XYZ-Euler angles. Near |pitch| = 90° that is singular: a few degrees of real tilt change roll/pitch by tens of degrees.

**Measured consequence** (previous diagnostics, `docs/state_estimation_diagnostics.md`): every episode whose cube spawned with |pitch| ≥ 30° (about half of the grasp-cache starts) was terminated, 98–100% of them, after only 7–32° of real tilt.

**Fix: physical tilt.** Let u(q) = R(q)ᵀ e_z be the world vertical expressed in the cube body frame. Then

  tilt(q, q₀) = arccos( u(q) · u(q₀) )

Properties:
- It is invariant to any rotation about world Z (the task motion), because (R_z R)ᵀ e_z = Rᵀ e_z.
- It equals the angle through which the cube is tipped away from its reference attitude.
- It has no singular orientations.

The thresholds are unchanged: termination at a tilt of 0.8 rad (45.8°) or 0.08 m of drift; the reward's drift gate at 0.35 rad or 0.02 m.

Validated by tests (`tests/test_scratch_study.py`):
- A full world-Z turn never terminates, including from near-gimbal-lock starts (|pitch| = 88–89.9°).
- The tilt equals the applied horizontal rotation angle for any start orientation.
- 30° of tilt does not terminate; 50° does.
- Drift above 8 cm terminates.

### 1.3 Rotation measurement

**Bug.** The original reward and evaluation used finite differences of the Euler yaw angle. Euler yaw is ill-defined near |pitch| = 90°, and differencing it across resets counted teleports as rotation.

**Fix: genuine yaw increment.** It is the z-component of the rotation vector of q_t ⊗ q_{t−1}⁻¹ (taken with w ≥ 0):
- exact for rotations about Z;
- continuous;
- unaffected by quaternion sign flips;
- zero on the first step of each episode.

Evaluation sums increments only over transitions inside one episode. It reports:
- net signed rotation;
- positive-direction rotation;
- turns of the first, uninterrupted episode (the primary metric);
- turns including later episodes after a reset (secondary), where the reset jump itself is never counted.

Tests check exactness, sign-flip invariance, that horizontal rotation contributes zero, and that a 120° reset teleport yields zero reward.

### 1.4 Action saturation

**Mechanism.**
- The policy outputs a 16-D Gaussian sample u. The original wrapper clipped it to ±1, and the action term maps that to a per-step joint-target change of ±1/24 rad, about 0.83 rad/s at 20 Hz. The physical range is unchanged in this study.
- PPO stores the *unclipped* sample and its Gaussian log-probability, but the environment executes `clip(u)`.
- When moving at the maximum rate is rewarded, samples beyond ±1 earn the same return as samples at ±1. Their likelihood gradient still pushes the mean outward, and nothing pushes it back.
- The mean therefore drifts outside the clip range, where exploration no longer changes the executed action.
- In the fine-tuned `model_400`: 95% of entries were saturated, the median |μ| was 12.8, and 94% of exploration samples left the executed action unchanged.

**Fix:** the action term executes `tanh(u)` instead of `clip(u)`, with the same delta mapping.
- tanh is a smooth bijection, so the PPO likelihood ratio of the stored Gaussian sample equals the ratio of the executed squashed action: the Jacobian term cancels (unit-tested).
- KL is also invariant under the bijection, so the adaptive schedule sees the true policy change.
- There is no zero-sensitivity region, and no physical limit is changed.

**Pilot evidence** (Section 1.6): with clipping, from scratch, saturation reappears and grows. The mean-action saturation went from 0 to 5.2% in 200 iterations, and executed saturation from 16% to 28%. With tanh, both stayed below 1%.

### 1.5 Learning rate

**Diagnosis.** RSL-RL's adaptive schedule divides the learning rate by 1.5 whenever the minibatch KL exceeds 2 × 0.01, with a floor of 1e-5. When fine-tuning `model_400`, the raw outputs were around 10–30, so even tiny parameter steps moved the Gaussian means by a lot relative to σ = 0.75. KL therefore kept exceeding the target, and the learning rate sat at its floor for all 300 iterations.

**From scratch** (pilot), the same schedule behaves as designed: the learning rate is 5e-4 to 1.5e-3, and KL is about 0.017, inside its band. A fixed 3e-4 learned similarly.

**Decision:** keep the original schedule (adaptive, initial 1e-3, desired KL 0.01) for all runs. It is the smallest change, and its behavior is logged for every run.

### 1.6 Pilot

Three runs, each with seed 1, 200 iterations and 4096 environments, all with the corrected termination, reward and metrics:

| | P1 clip + adaptive (original) | P2 tanh + adaptive | P3 tanh + fixed 3e-4 |
|---|---|---|---|
| Reward, iteration 0 → 199 | −0.11 → 0.43 | −0.12 → 0.38 | −0.12 → 0.37 |
| Genuine yaw rate in training episodes [rad/s] | 0.08 → 0.19 | 0.06 → 0.21 | 0.06 → 0.17 |
| Executed saturation | 0.16 → **0.28** | 0.0002 → 0.007 | 0.0002 → 0.004 |
| Mean-action saturation | 0 → **0.052** | 0 → 0.0002 | 0 → 0.00001 |
| Learning rate | 1e-3 → 5.3e-4 | 1.5e-3 → 5.4e-4 | 3e-4 |
| KL after update | 0.009 → 0.018 | 0.009 → 0.017 | 0.009 |
| Effective exploration (share of samples changing the executed action by > 0.05) | 0.94 → 0.91 | 0.94 → 0.93 | 0.94 → 0.93 |

Figure: `figures/pilot_stage0.png`.

- **PPO optimizes** in every variant.
- **Exploration changes the executed action** in more than 90% of samples.
- **Genuine rotation improves** during training.
- **No spurious terminations:** the GPU test resets 3 × 64 environments repeatedly; no termination fires on the anchoring step, and only 2 of 2048 resets end within 0.15 s, both genuine topples.
- **Frozen configuration:** after the pilot, the environment, reward, action interface and PPO configuration were frozen.

### 1.7 Tests

| File | Hardware | Tests | Status |
|---|---|---|---|
| `tests/test_scratch_study.py` | CPU | 17 | pass |
| `tests/test_scratch_env.py` | GPU | 5 | pass |

`test_scratch_study.py` covers:
- the physical-tilt invariances;
- yaw-increment exactness;
- the termination with a fresh anchor near gimbal lock;
- the reward never counting reset jumps;
- the drift gate using physical tilt;
- the tanh PPO ratio being exact;
- the symmetry invariance of the features;
- privileged-information isolation: A and C are unaffected by the critic and cube-state groups, and D is affected only by the cube-state group;
- the ablation switches.

`test_scratch_env.py` covers:
- the observation groups (actor 320, critic 83, cube_state 17);
- no spurious termination after repeated resets;
- the cube-state group matching the critic's ground truth;
- the environment executing exactly `tanh(u)`;
- a cube drop terminating.

**Deterministic evaluation:** the same seed reproduces the per-environment randomization exactly. MuJoCo-Warp contact dynamics are not bit-reproducible between runs, so paired comparisons use a repeated baseline run as their noise floor.

### 1.8 Bug found and fixed during Stage 3 (first C runs invalid)

**Symptom:** the first C ablation showed that replacing the estimated features with their mean, or shuffling them across environments, changed the executed actions by exactly 0. For D, the same changes moved the actions by about 0.4. The estimated features had zero variance across all environments and time steps.

**Cause:**
- `scripts/train.py --zero-actor-output` zero-initializes "the last `nn.Linear` in `actor.modules()`".
- In the C actor, the frozen estimator was registered as a submodule after the policy MLP, so the zeroed layer was **the estimator's output head**, not the policy's output layer.
- Every C run therefore:
  1. received constant "estimated" features carrying no state information, so it was effectively A plus a constant input;
  2. started from a default-initialized output layer, unlike A and D.
- A and D were unaffected: their policy output layers were verified to start at zero.
- The closed-loop estimator metrics were unaffected, because evaluation runs its own copy of B.

**Fix** (`scratch_study/actors.py`):
- The frozen estimator is kept outside the actor's module tree, so it is not in `modules()`, `parameters()`, `state_dict()`, the optimizer or `.train()`.
- It is pinned to eval mode and moved to the input's device on use.
- The original training script is unchanged.
- Regression test: `test_zero_actor_output_targets_policy_head_not_estimator`. It runs the real `_zero_actor_output` on the A, C and D actors and checks four things: (a) the policy output layer is zeroed; (b) the estimator is unchanged; (c) no estimator weights are in the checkpoint; (d) the features vary. It fails on the old code and passes on the fix.
- A 2-iteration GPU smoke run confirmed that the closed-loop feature std is non-zero.

**Consequence:**
- All C runs trained before the fix are archived as invalid (`logs/rsl_rl/scratch_C_INVALID_estimator_head_zeroed/`, `results/_INVALID_C_estimator_head_zeroed/`) and are not used anywhere in this report.
- C seeds 1–3 were then retrained from scratch with the fix (2026-11-12, 19:06 to about 22:10), using exactly the A/D settings and budget, uninterrupted (`scripts/sbatch_C_fresh.sh`). Their actor checkpoints contain no estimator weights, and their output layer starts at zero, as in A and D.

**Interrupted runs policy:** a gpu-01 node failure at 08:47 interrupted two of the invalid C runs. Interrupted runs are never resumed, because RSL-RL's resume does not restore the adaptive learning rate, the reward-curriculum EMA or the RNG state. They are retrained from scratch, so every run in the comparison is uninterrupted.

## 2. Methods

### 2.1 Conditions

| Experiment | Actor input | Deployable |
|---|---|---|
| A: baseline | 320-D proprioceptive history: 10 frames × (16 joint positions, with ±0.01 rad noise and a 0–1 step delay, + 16 previous joint-position commands) | yes |
| B: estimator | Not a policy: an MLP mapping the same 320-D input to the cube state | yes |
| C: estimated state | 320-D history + 19 features from the frozen estimator B | yes |
| D: oracle state | 320-D history + the same 19 features computed from MuJoCo ground truth | no: needs external object sensing |

**Features** (`scratch_study/features.py`), identical for C and D:

| Feature | Dims | Definition |
|---|---|---|
| Position | 3 | Palm-frame position, standardized with fixed scales |
| Rotation | 6 | First two columns of the canonical cube rotation, i.e. the one of the 24 cube-symmetric representatives closest to the palm frame |
| Linear velocity | 3 | Palm-frame linear velocity, divided by fixed scales |
| Angular velocity | 3 | Palm-frame angular velocity, divided by fixed scales |
| Contacts | 4 | Fingertip contacts: probabilities for C, 0/1 for D |

The scales are fixed constants, so C and D see exactly the same mapping, and D does not depend on B.

**Architecture and training:**
- **Actor input:** the 320-D part is normalized by the actor's running normalizer, as in A. The features enter unnormalized, since they are already on an O(1) scale.
- **Critic:** identical for all conditions (83-D privileged state, unchanged). Policy and value networks are 512-512-256 ELU; the actor's last layer starts at zero, and the initial noise std is 0.7.
- **PPO:** 32 steps per environment, 4096 environments, 5 epochs × 4 minibatches, γ 0.99, λ 0.95, clip 0.2, entropy coefficient 0.003, maximum gradient norm 1.0, adaptive learning rate starting at 1e-3 with desired KL 0.01.
- **Budget:** 1001 PPO updates (131.2M environment steps) per run.
- **Seeds:** training seeds 1, 2 and 3 for every condition.

### 2.2 Environment

The original continuous +Z rotation task for the right LEAP Hand with the 58 mm MBZUAI cube, unchanged except for the Section 1 corrections.

**Reward** (weights unchanged; the penalty weights follow the original progress curriculum during training and sit at their final values during evaluation):

| Term | Weight |
|---|---|
| Drift-gated yaw rate (clipped to ±0.25 rad/s; ×0.1 outside 2 cm / 0.35 rad) | 1.25 |
| Cube linear-velocity L1 | −0.3 |
| Joint-pose deviation beyond 0.4 rad | −0.1 |
| Torque² | −0.1 |
| Work | −0.05 |
| Cube fallen | −10 |

**Terminations:** time-out after 20 s; cube below 0.2 m (drop); cube faster than 1 m/s; physical tilt above 0.8 rad or drift above 0.08 m.

**Domain randomization** (unchanged):

| Parameter | Range | Applied |
|---|---|---|
| Cube mass | ×0.7–1.4 | per reset |
| Friction | 0.6–1.4 | per reset |
| Cube center of mass | ±2 mm | per reset |
| Link mass | ×0.8–1.2 | per reset |
| Joint friction | ×0.5–1.8 | per reset |
| Joint damping | ×0.6–1.6 | per reset |
| Armature | ×0.7–1.3 | at startup |
| PD gains | ×0.9–1.1 | at startup |
| Actuator delay | 5–15 physics steps | per reset |
| Effort limit | ×0.9–1.1 | per reset |

Episodes start from grasp-cache states with random yaw.

### 2.3 Estimator dataset (Stage 2)

Data was collected with `scripts/stage2_estimator.sh` (the plan is logged in `data/collection_plan.txt`) using the training environment: observation noise, actuator delays and the full DR.
- **Episodes:** each collection runs 600 control steps (30 s) per environment, with automatic resets, so every episode, failures included, runs until it terminates or the window ends. Half the environments sample actions from the policy Gaussian, half take the mean.
- **"std ×2" runs:** these double the sampling std, producing more slips, drift and drops.

| Split | Policies (Experiment A) | Checkpoints | Envs × steps | Sim seeds | Rows |
|---|---|---|---|---|---|
| train | A seed 2, A seed 3 | 100, 300, 600, 1000, plus 1000 at std ×2 | 4×1024×600 + 512×600, per policy | 11000–11009 | 5,529,600 |
| val | A seed 2, A seed 3 | 300, 1000 | 256×600 each | 12000–12003 | 614,400 |
| test | **A seed 1** (held-out policy) | 100, 300, 600, 1000, plus 1000 at std ×2 | 4×512×600 + 512×600 | 13000–13004 | 1,536,000 |

**Separation:**
- No simulation seed is shared between splits.
- The test policy never produced training or validation data, so the test set measures generalization to a new policy's state distribution, not only to new rollouts.
- Estimator inputs are exactly the 320-D actor observation, verified per run (`leakage` in `results/estimator/estimator_report.json`).
- The maximum absolute input–label correlation is 0.63, which comes from the joint positions; no label is copied into the input.

### 2.4 Estimator B

- **Model:** MLP 320 → 512-512-256 (LayerNorm, ELU), with input normalization from training-split statistics.
- **Heads:** palm-frame position, rotation, linear velocity, angular velocity, and four fingertip-contact logits.
- **Rotation:**
  - predicted as a 6-D representation;
  - trained with the **symmetry-aware** chordal loss min_S ‖R̂ − R S‖²_F, with S over the 24 cube rotations;
  - all orientation errors are reported modulo the cube's symmetry.
- **Loss:**
  - position is standardized and velocities are divided by their training RMS, so the Huber terms are on comparable scales; weights are pos 1, rot 1, vel 1 (raised from the default 0.5 so velocity is not under-weighted), contact 0.5;
  - contact BCE is unweighted (no positive-class weight), and the first step after a reset is masked out, since its contact sensor value predates the reset.
- **Training:** AdamW at 1e-3 with weight decay 1e-4, cosine schedule, 60k steps at batch 4096.
- **Checkpoint selection:** on validation loss only; the best checkpoint was the last one (step 60000).
- **Frozen artifact:** `models/estimator_B.pt` (sha256 in `models/estimator_B.sha256`). Training logs are in `runs/estimator/*_estimator_B/`, and the test split was not touched during training (`--skip-test`).

### 2.5 Evaluation protocol

**Checkpoint selection** (fixed before any test evaluation):
- Every 100 updates, each run's checkpoint is evaluated on the **validation** simulation seed 7001: nominal preset, 1024 environments, 600 control steps (30 s), deterministic actions.
- The selected checkpoint has the highest mean *uninterrupted turns*, with ties going to the later checkpoint.
- Test seeds are never used for selection.

**Test:** the selected checkpoints, on simulation seeds 5001 and 5002, for each preset below.

| Preset | Change from training DR |
|---|---|
| nominal | Training DR |
| heavy_slippery | Cube mass ×1.3–1.8, friction 0.4–0.7 (outside training range) |
| light_grippy | Mass ×0.5–0.8, friction 1.2–1.6 (outside training range) |
| high_delay | Actuator delay 14–20 physics steps |
| weak_motors | Effort limit ×0.65–0.8 |
| obs_noise_x3 | Joint-position noise ±0.03 rad |

**Per-environment metrics** (`scratch_study/evaluation.py`):
- **Turns:** net signed +Z rotation until the first termination, computed with the singularity-free yaw increment and divided by 2π.
- **Rotation:** net rotation in rad and desired-direction rotation.
- **Yaw rate:** mean yaw rate.
- **Survival:** time until first failure.
- **Failure type:** drop, pose (tilt or drift) failure, or too fast.
- **Reward:** episode reward.
- **Drift:** maximum and mean drift (mm).
- **Tilt:** maximum and mean physical tilt (°).
- **Saturation:** executed-action saturation.

**Statistics:**
- The unit of replication is the training seed. Each run's metric is the mean over all environments of both test seeds.
- A condition's value is the mean of its 3 run values, reported with the across-seed s.d.
- 95% CIs come from a bootstrap over training seeds. With n = 3 they are coarse, so the individual seed values are always shown too.
- Within-run variation across test seeds is reported separately as a noise reference.

**Estimator in closed loop:** while each policy runs, estimator B also runs on its observations. This gives B's accuracy on A, C and D's own state distributions; for C, it measures the distribution shift its own policy induces.

## 3. Results

### 3.1 Estimator B: held-out-policy test split (Stage 2)

The test data comes from A seed 1, a policy whose data was never used for training: 1.536M rows (`results/estimator/estimator_report.json`).

| Metric | **MLP (B)** | Ridge (linear) | Prior (training mean) | Initial-hold pose |
|---|---|---|---|---|
| Position RMSE [mm] | **10.2** | 15.2 | 24.4 | 23.3 |
| Orientation error, symmetric, mean / median [°] | **16.8 / 13.7** | 39.2 / 40.6 | 38.6 / 39.5 | 29.8 / 30.4 |
| Linear-velocity RMSE [m/s] | **0.0205** | 0.0251 | 0.0267 | 0.0267 |
| Angular-velocity RMSE [rad/s] | **0.63** | 0.74 | 0.79 | 0.79 |
| Yaw-rate RMSE [rad/s] | **0.31** | 0.37 | 0.43 | 0.43 |
| Contact micro-F1 / macro-F1 | **0.79 / 0.69** | 0.71 / 0.43 | 0.62 / 0.19 | 0.21 / 0.13 |

Per-finger contacts (precision / recall / F1):

| Finger | Precision | Recall | F1 |
|---|---|---|---|
| Index | 0.72 | 0.68 | 0.70 |
| Middle | 0.68 | 0.62 | 0.65 |
| Ring | 0.71 | 0.49 | 0.58 |
| Thumb | 0.80 | 0.90 | 0.85 |

The ring finger is rarely in contact (0.7% positive rate).

**Validation versus test** (same estimator):
- Validation (training policies, new sim seeds): 7.3 mm and 10.6°.
- Test (new policy): 10.2 mm and 16.8°.

Validation already uses unseen simulation seeds, so most of this gap reflects the change of policy. The comparison is confounded, though: the test split also contains checkpoints 100 and 600 and the std ×2 rollouts, which validation lacks. Either way, the estimator degrades on a new policy's state distribution. This matters for C, whose own policy creates yet another distribution (Section 3.4).

**Events** (`results/estimator_events/estimator_events.json`; MLP versus ridge):

| Subset | Share of rows | Position [mm] | Orientation [°] | Drift abs. error [mm] | Yaw-rate RMSE (R²) |
|---|---|---|---|---|---|
| Stable (low speed, small drift) | 3% | 6.8 | 11.4 | 3.9 | 0.14 (−0.02) |
| Translational slip | 10% | 10.4 | 16.8 | 4.6 | 0.77 (0.38) |
| Tumbling slip | 6% | 10.6 | 16.8 | 4.6 | 0.99 (0.39) |
| Drift > 20 mm | 39% | 12.4 | 19.5 | 6.0 | 0.31 (0.20) |
| Within 0.5 s of failure | 3% | 13.2 | 23.3 | 6.0 | 0.58 (0.14) |

**Detection AUROC:**

| Event | Estimator | Ridge | Proprioception |
|---|---|---|---|
| Drift > 10 mm | **0.89** | 0.74 | 0.43 |
| Failure within 0.5 s, best estimated cue (estimated off-axis angular speed) | **0.81** | 0.67 | 0.52 |
| Same, ground truth | 0.79 | — | — |
| Drop within 0.5 s | 0.96 | — | — |

The drop-within-0.5 s result rests on only 44 positive samples.

**Velocity is not reliable.**
- R² is 0.41 for linear velocity, 0.33 for off-axis angular velocity and 0.35 for yaw rate.
- In stable phases, the velocity R² is ≤ 0 (the estimate is no better than a constant).
- The velocity features given to C are therefore mostly a coarse "something is moving or slipping" signal, not an accurate measurement.
- Position, drift and orientation are substantially better than every baseline.

**Robustness to dynamics:** errors barely change across cube mass, friction, actuator delay and damping quartiles (position 9.9–10.5 mm). They grow with time in episode: 4.7 mm in the first 10 steps (the cube is near its known start pose), against about 11 mm after 100 steps.

Figures: `results/estimator/baselines.png`, `results/estimator/timeseries_*.png`, `results/estimator_events/lead_time_before_failure.png`.

Loss curves: `figures/estimator_training.png`.

### 3.2 Learning curves (A, C, D)

Figures:
- `figures/validation_curves.png`: validation turns, failure rate, survival and yaw rate against environment steps, every 100 updates, for each run and each condition mean.
- `figures/training_curves.png`: training reward, genuine yaw rate, episode length, termination rates, saturation, exploration, learning rate, KL, policy std, entropy and gradient norm.

All curves are cut at update 1000, the study budget. The training budget was identical: 1001 updates, or 131.2M environment steps, per run.

**Learning speed** (validation seed 7001, nominal; per-seed values for seeds 1 / 2 / 3; from `results/tables.md`):

| Measure | A | C | D |
|---|---|---|---|
| First update with validation turns ≥ 0.5 | 900 / 600 / 600 | 500 / 300 / 300 | 500 / 300 / 400 |
| First update with validation failure ≤ 0.5 | never / 700 / 700 | 500 / 400 / 500 | 600 / 400 / 500 |
| Mean validation turns over updates 100–1000 | 0.386 / 0.495 / 0.537 (mean 0.473) | 0.584 / 0.828 / 0.647 (mean 0.686) | 0.590 / 0.824 / 0.751 (mean 0.721) |
| Validation turns at update 1000 | 0.67 / 0.84 / 0.86 | 0.88 / 1.15 / 0.82 | 0.96 / 1.11 / 1.08 |

- **C and D learn faster than A.** Every C seed and every D seed has a larger mean-over-curve than every A seed. With 3 vs 3 seeds, complete separation has an exact one-sided permutation probability of 1/20 = 0.05 under "no difference", so this is the strongest separation n = 3 can show.
- **C learns about as fast as D on this measure.** The means are 0.686 and 0.721, with strongly overlapping seeds.
- **No run had converged.** Every run, in every condition, was still improving at update 1000, so all comparisons are at a fixed and limited budget. The ranking at convergence is unknown.

**Selected checkpoints** (by validation): every run selected `model_1000` except A seed 3, which selected `model_900`. Copies, with their sha256, are in `checkpoints/` (`checkpoints/MANIFEST.txt`).

### 3.3 Held-out test: final performance

All 9 selected checkpoints were evaluated on test sim seeds 5001 and 5002 × 6 presets × 1024 environments × 30 s, with deterministic actions. Full tables with s.d. and bootstrap CIs for every metric and preset are in `results/tables.md`. Machine-readable versions are in `results/final_table.json` and `results/pairwise_differences.json`, and the figure is `figures/final_comparison.png`.

**Uninterrupted turns in 30 s** (condition mean ± across-seed s.d.; per-seed values for seeds 1 / 2 / 3):

| Preset | A | C | D |
|---|---|---|---|
| nominal | 0.785 ± 0.100 (0.671 / 0.823 / 0.859) | 0.927 ± 0.168 (0.864 / 1.117 / 0.799) | 1.041 ± 0.090 (0.937 / 1.099 / 1.086) |
| heavy_slippery | 0.814 ± 0.075 | 0.972 ± 0.179 | 1.064 ± 0.108 |
| light_grippy | 0.679 ± 0.094 | 0.836 ± 0.173 | 0.973 ± 0.084 |
| high_delay | 0.707 ± 0.085 | 0.843 ± 0.158 | 0.961 ± 0.100 |
| weak_motors | 0.778 ± 0.111 | 0.930 ± 0.174 | 1.036 ± 0.085 |
| obs_noise_x3 | 0.773 ± 0.103 | 0.926 ± 0.162 | 1.036 ± 0.091 |
| **Pooled** (per-seed mean over the 6 presets) | **0.756** (0.648 / 0.793 / 0.826) | **0.906** (0.849 / 1.095 / 0.773) | **1.018** (0.913 / 1.080 / 1.062) |

**Other nominal-preset metrics:**

| Metric | A | C | D |
|---|---|---|---|
| Tilt/drift failure rate | 0.425 ± 0.110 | 0.323 ± 0.075 | 0.285 ± 0.021 |
| Survival [s] | 20.9 ± 3.0 | 23.0 ± 1.8 | 24.4 ± 0.1 |
| Turns including after resets | 1.12 ± 0.03 | 1.21 ± 0.17 | 1.27 ± 0.11 |
| Mean yaw rate [rad/s] | 0.235 | 0.245 | 0.255 |
| Mean drift [mm] / mean tilt [°] | 18.3 / 17.0 | 17.6 / 17.3 | 17.7 / 17.5 |
| Episode reward | 0.62 ± 0.23 | 0.97 ± 0.34 | 0.62 ± 0.25 |
| Executed saturation | 0.055 | 0.071 | 0.080 |

**Condition differences, pooled over presets** (difference of means, with a 95% bootstrap CI over training seeds; n = 3 per condition):

| Metric | C − A | D − A | D − C |
|---|---|---|---|
| Turns | +0.150 [−0.006, +0.339] | **+0.263 [+0.148, +0.376]** | +0.113 [−0.075, +0.276] |
| Failure rate | −0.107 [−0.229, +0.002] | **−0.147 [−0.267, −0.072]** | −0.040 [−0.132, +0.019] |
| Survival [s] | +2.24 [−0.57, +5.13] | **+3.59 [+1.77, +7.00]** | **+1.35 [+0.16, +3.42]** |

**Observations:**
- **Rank order.** The order is the same in every preset: D > C > A for turns, failure and survival.
- **C − A.** C − A is +0.14 to +0.16 turns in every preset, but every CI just includes 0. C seed 3 (pooled 0.773) is below A seeds 2 and 3 (0.793, 0.826).
- **D − A.** D − A is positive in every preset, and every CI excludes 0. Every D seed beats every A seed.
- **Share of the oracle gain.** In pooled turns, C recovers about 57% of the D − A gap (0.150 / 0.263).
- **Spread across seeds.** C has the largest spread: C seed 2 is the best policy in the study (1.095 pooled), while C seed 3 is at A's level. D is the most consistent.
- **Drops.** None of the 9 policies dropped the cube in any held-out evaluation. Every failure is a tilt or drift failure, with too-fast failures under 0.1%.
- **Presets.** light_grippy and high_delay cost every condition about 0.07–0.11 turns relative to nominal. heavy_slippery is slightly *easier* than nominal for all three. Three-fold observation noise has almost no effect (≤ 0.012 turns), including for C, whose estimator sees the noisier joints.
- **Test-seed noise reference.** Within a run, the two test seeds differ by 0.014 turns on average (maximum 0.039, over 54 run × preset pairs). That is far smaller than the across-training-seed s.d. (0.08–0.18), so the training seed dominates the uncertainty.
- **Episode reward ranks C > A ≈ D, even though D rotates most.** The evaluation reward uses the final (post-curriculum) penalty weights. D's higher action saturation and faster rotation may cost more torque, work and linear-velocity penalty. This is a hypothesis; the reward terms were not decomposed at test time.

### 3.4 Estimator B in closed loop (distribution shift)

Estimator B was run on every policy's own held-out test rollouts (`estimator_closed_loop` in `results/final/*.json`). The table below lists the mean over the 6 presets and both test seeds.

| Policy | Relation to B's data | Position RMSE [mm] | Symmetric rotation error [°] | Linear-velocity RMSE [m/s] | Yaw-rate RMSE [rad/s] |
|---|---|---|---|---|---|
| A seed 2 | produced training/val data | 5.6 | 6.8 | 0.018 | 0.26 |
| A seed 3 | produced training/val data | 5.6 | 7.7 | 0.021 | 0.26 |
| A seed 1 | produced offline *test* data | 8.8 | 15.7 | 0.020 | 0.33 |
| C seed 1 / 2 / 3 | own policy, never seen | 10.1 / 9.7 / 9.3 | 14.4 / 15.3 / 18.9 | 0.020–0.022 | 0.30–0.37 |
| D seed 1 / 2 / 3 | never seen | 9.2 / 11.1 / 16.4 | 12.3 / 18.7 / 20.3 | 0.020–0.022 | 0.30–0.34 |

- **Training versus unseen policies.** On the two policies it was trained on, B is accurate to about 5.6 mm and 7°. On any policy it has not seen, the errors roughly double: 9–16 mm and 12–20°.
- **C's own distribution.** The states C drives itself into are not harder for B than those of another unseen policy, such as A seed 1 or the D policies. C does not appear to push the estimator into a uniquely bad region.
- **Velocities.** Velocity errors barely change across policies. They are poor everywhere (Section 3.1), so the velocity features carry little information for any policy.
- **Not tested here.** Whether C would do better with an estimator trained on C's own rollouts (iterated data collection, as in DAgger) is a hypothesis this study did not test.

### 3.5 Feature ablation: do C and D use the cube-state features?

**Setup:**
- Every C and D policy was rerun with its 19 cube features replaced, the rest of the input untouched. The replacements were:
  - **zero:** all zeros;
  - **mean:** the run's own mean feature vector, recorded on the normal run;
  - **shuffle:** another environment's features, i.e. a plausible but wrong state (partner environment (i + n/2) mod n).
- Same sim seeds, presets and randomization as the normal run, so every comparison is paired per environment.
- Presets nominal and heavy_slippery × test seeds 5001 and 5002 × 1024 environments.
- The action change is the mean absolute change in the executed action ([−1, 1] scale) on the states the normal policy visits.

**Result** (averaged over the 2 presets × 2 test seeds; full per-cell table with paired 95% CIs in `results/tables.md`):

| Run | Normal turns | Zero turns | Mean turns | Shuffle turns | Turns kept (zero / mean / shuffle) | Failure, normal → shuffle | Mean abs. action change (zero / mean / shuffle) |
|---|---|---|---|---|---|---|---|
| C seed 1 | 0.884 | 0.294 | 0.316 | 0.168 | 33% / 36% / 19% | 0.27 → 0.78 | 0.39 / 0.34 / 0.44 |
| C seed 2 | 1.147 | 0.687 | 0.813 | 0.285 | 60% / 71% / 25% | 0.27 → 0.66 | 0.32 / 0.27 / 0.34 |
| C seed 3 | 0.825 | 0.281 | 0.526 | 0.297 | 34% / 64% / 36% | 0.38 → 0.68 | 0.36 / 0.29 / 0.39 |
| D seed 1 | 0.937 | 0.156 | 0.223 | 0.095 | 17% / 24% / 10% | 0.31 → 0.56 | 0.40 / 0.37 / 0.49 |
| D seed 2 | 1.109 | 0.196 | 0.234 | 0.133 | 18% / 21% / 12% | 0.27 → 0.76 | 0.36 / 0.31 / 0.42 |
| D seed 3 | 1.115 | 0.307 | 0.408 | 0.170 | 28% / 37% / 15% | 0.27 → 0.71 | 0.36 / 0.30 / 0.40 |

Every ablation effect is far outside its paired CI (typically ±0.03 turns). No drops occurred in any ablation; failures are tilt or drift.

**Observed:**
- **C relies on the estimated features.**
  - Replacing them changes 71–81% of executed action components by more than 0.05, with a mean change of 0.27–0.44.
  - Turns fall to 19–71% of normal.
  - With features zeroed, C seeds 1 and 3 (0.28–0.29 turns) fall far below every A policy, which was trained without features (A: 0.67–0.86 on nominal). C seed 2 (0.69) lands at the bottom of A's range. C did not learn "A plus an optional input": it built its control around the estimate.
- **Wrong state is worse than no state.** Shuffled (plausible but wrong) features are the most damaging for every run. The policies therefore use the features as state-dependent information, not as a constant bias.
- **D relies on its features even more** (10–37% of turns kept).
- **C keeps more performance than D** under zero and mean replacement (33–71% vs. 17–37%).
  - **Hypothesis:** because B's features are noisy, C learned to weight them less and to use the joint history more. Not tested.
- **The fix worked.** Before the fix (Section 1.8), these same ablations changed C's actions by exactly 0.

### 3.6 Action saturation and exploration diagnostics

These panels are in `figures/training_curves.png`. The values are the mean of the last 20 updates (`results/training_last20_summary.json`).

| Diagnostic | A | C | D |
|---|---|---|---|
| Executed saturation (|tanh(u)| > 0.99), end of training | 0.12–0.16 | 0.15–0.17 | 0.15–0.16 |
| Policy std (pre-tanh), from 0.70 at the start | 1.16–1.19 | 1.16–1.19 | 1.14 |
| Effective exploration share | 0.78–0.82 | 0.77–0.79 | 0.76–0.78 |
| Adaptive learning rate, end | 3e-4 | 3e-4–4e-4 | 3e-4–4e-4 |
| Approx. KL after update | 0.018 | 0.018–0.019 | 0.018–0.019 |
| Executed saturation at test (deterministic), nominal | 0.055 | 0.071 | 0.080 |

- **The Stage-0 corrections held for all conditions.** tanh squashing kept the executed saturation far below the original clipped action's, and the adaptive learning rate settled in a stable band without collapsing to its floor.
- **Shared drift in all conditions.** Policy std grew steadily, from 0.70 to about 1.15–1.19, and executed saturation rose with it. It is nearly identical across A, C and D, so it does not confound the comparison. It is a limitation of the shared PPO setup, though: the entropy bonus (0.003), acting on the pre-squash std, may be stronger than needed with tanh actions. This is a hypothesis, not tested.

## 4. Conclusions

These conclusions use n = 3 training seeds per condition. **Observed** marks direct measurements; **Hypothesis** marks interpretation not tested here.

1. **Can cube state be estimated from proprioception alone?**
   - **Observed:** partly yes. On a held-out policy, B estimates cube position to 10.2 mm and symmetric orientation to 16.8° (median 13.7°). That clearly beats a linear model (15.2 mm, 39°) and the prior (24.4 mm, 39°).
   - Contacts are estimated reasonably well (micro-F1 0.79).
   - Velocities are estimated poorly (R² 0.33–0.41, and ≤ 0 when the cube is stable).

2. **How accurate is it during difficult manipulation?**
   - **Observed:** errors grow in difficult phases: 10.4–10.6 mm and 17° during slips, 12.4 mm when drift exceeds 20 mm, and 13.2 mm and 23° within 0.5 s of failure.
   - The estimate still detects drift above 10 mm (AUROC 0.89) and imminent failure (0.81, comparable to the 0.79 from ground truth).
   - In closed loop on unseen policies, including C itself, errors are 9–16 mm and 12–20°, about twice those on the policies that produced the training data.

3. **Does estimator-augmented PPO learn faster than baseline PPO?**
   - **Observed:** yes, at this budget. All 3 C seeds reach 0.5 validation turns at updates 300–500, against 600–900 for A. All 3 C seeds have a higher mean-over-curve than all 3 A seeds (0.686 vs. 0.473), the strongest separation n = 3 allows (one-sided permutation p = 0.05).

4. **Does it achieve better final manipulation performance?**
   - **Observed:** C is better than A on average in every held-out preset: +0.15 turns pooled, failure −0.11, survival +2.2 s.
   - The 95% CIs over training seeds just include zero, and one C seed is no better than A.
   - **Verdict:** a consistent but not statistically established improvement. Final-performance evidence is preliminary.

5. **Does ground-truth cube-state information improve PPO?**
   - **Observed:** yes. D beats A in every preset: +0.26 turns pooled [+0.15, +0.38], failure −0.15 [−0.27, −0.07], survival +3.6 s [+1.8, +7.0].
   - D also learns faster, and its seeds are the most consistent.
   - D is a simulation-only upper reference and is not deployable without external object sensing. It is not an upper bound for every seed: C seed 2 (1.095 pooled turns) beat all three D seeds.

6. **How closely does estimator PPO approach oracle PPO?**
   - **Observed:** C recovers about 57% of the D − A gain in pooled turns (0.150 of 0.263) and about 73% of the failure-rate gain.
   - D − C is +0.11 turns [−0.08, +0.28], not resolved with n = 3. Survival is the exception: D − C is +1.35 s [+0.16, +3.42].
   - C learns about as fast as D.
   - C's across-seed standard deviation is about twice D's (nominal turns: 0.168 vs. 0.090).

7. **Does estimator PPO actually use the predicted information?**
   - **Observed:** yes, strongly. Replacing the estimated features changes most action components (mean absolute change 0.27–0.44) and cuts turns to 19–71% of normal.
   - Shuffled (wrong-state) features are the most harmful, which shows state-dependent use.
   - With features zeroed, C policies fall to or below the level of the feature-free A policies.
   - C depends on its features somewhat less than D depends on true state.

8. **Limitations and likely causes of the gap:**
   - **Seeds:** n = 3 per condition; the CIs are wide and the bootstrap over 3 values is coarse.
   - **Budget:** all runs were still improving at 131M steps, so the results describe sample efficiency at a fixed budget, not asymptotic performance.
   - **Estimator–policy mismatch:**
     - **Observed:** B was trained only on A rollouts and is about twice as inaccurate on unseen policies.
     - **Hypothesis:** this distribution shift and the poor velocity estimates explain part of the D − C gap and C's larger seed variance. Retraining B on C's own rollouts was not tested.
   - **Simulation only:** no hardware evaluation was run, and B's sim-to-real accuracy is unknown.
   - **Shared PPO drift:** the policy std grows in all conditions (Section 3.6). This affects A, C and D equally. **Hypothesis:** it may limit every condition's final performance.
   - **Bug:** the bug in the first C runs (Section 1.8) was found by the ablation and fixed. All reported C results come from the retrained runs.

## 5. Compute

Hardware: Slurm cluster, NVIDIA RTX 5000 Ada (32 GB) GPUs on node gpu-01, shared with other users' workloads.

| Work | GPU time | Basis |
|---|---|---|
| PPO training, valid runs (A, C, D × 3 seeds) | 34.1 run-hours wall-clock (A 13.1, C 7.9, D 13.1) | `Time elapsed` at update 1000 in `logs/train_*.log` |
| Same, as GPU-device hours | about 18 h | runs were co-located 2 per GPU (A_s and D_s on one GPU; C seeds 1 and 3 on one GPU), so per-run wall-clock overstates device use |
| Invalid C runs (Section 1.8), discarded | about 7.5 run-hours | archived logs |
| Pilot (Stage 0) | 3 runs × 200 updates | `logs/pilot_*.log` |
| Estimator data collection and training | about 1 h, approximate | collection logs end 06:28–06:38; estimator training about 5 min (06:37–06:42) |
| Evaluation: validation curves, held-out test, ablations | about 9 h, approximate | log timestamps (curves about 25 min per run, test about 15–25 min per run, ablations about 25–40 min per run) |

Training wall-clock varies (1.9–5.5 h per run) with GPU sharing. Results do not depend on it: every run used the same settings, budget and seeds, and none was resumed.

## 6. Reproduction

All commands run from `training_source/project/`, in the conda environment `leap-estimator` (`conda/setup_conda_env.sh`, exact versions in `conda/requirements-leap-estimator.lock.txt`).

**Stage 0 pilot:**
```bash
bash experiments/scratch_state_estimation/scripts/stage0_pilot.sh
```

**A and D, seeds 1–3** (4096 envs, `STUDY_ITERATIONS = 1001` in `leap_right_scratch_study/rl_cfg.py`, zero-initialized actor output):
```bash
GPU=<g> bash experiments/scratch_state_estimation/scripts/train_condition.sh A <seed>
GPU=<g> bash experiments/scratch_state_estimation/scripts/train_condition.sh D <seed>
```
The script resumes if partial checkpoints exist. This study never resumed: no `[resume]` line appears in any training log.

**B: data and estimator:**
```bash
MODE=trainval GPUS="2 3" A1=<A seed2 run> A2=<A seed3 run> bash experiments/scratch_state_estimation/scripts/stage2_estimator.sh
MODE=test     GPUS="1"   A3=<A seed1 run>                  bash experiments/scratch_state_estimation/scripts/stage2_estimator.sh
```
- The collection plan, including sim seeds and checkpoints, is logged in `data/collection_plan.txt`.
- Estimator hyperparameters are in `runs/estimator/*_estimator_B/args.json`.
- Estimator B was frozen to `models/estimator_B.pt` (sha256 `6e333d60…e64`).

**C, seeds 1–3** (identical settings; the estimator path comes from `LEAP_SCRATCH_ESTIMATOR`). The command trains, then runs the validation curve, held-out test and ablation for that seed:
```bash
CUDA_VISIBLE_DEVICES=<g> SEED=<seed> EXTRA_ABL=D<seed> bash experiments/scratch_state_estimation/scripts/sbatch_C_fresh.sh
```

**Evaluation and figures:**
```bash
GPU=<g> bash experiments/scratch_state_estimation/scripts/run_curves.sh A1 A2 A3 D1 D2 D3   # validation curves + checkpoint selection
GPU=<g> bash experiments/scratch_state_estimation/scripts/run_final.sh  A1 A2 A3 D1 D2 D3   # held-out test (seeds 5001/5002, 6 presets)
python experiments/scratch_state_estimation/scripts/study_eval.py ablation --policy D_seed1=D:<ckpt> --num-envs 1024 --out experiments/scratch_state_estimation/results/ablation
python experiments/scratch_state_estimation/scripts/make_figures.py   # figures/ and results/tables.md
pytest tests/test_scratch_study.py tests/test_estimation.py
```

**Seeds:**

| Purpose | Seeds |
|---|---|
| Training | 1, 2, 3 |
| Estimator data | 11000–11009 (train), 12000–12003 (val), 13000–13004 (test) |
| Validation | 7001 |
| Test | 5001, 5002 |

**Output layout** (`experiments/scratch_state_estimation/`):

| Path | Contents |
|---|---|
| `checkpoints/` | selected policies plus manifest |
| `models/` | estimator B |
| `results/curves/` | validation curves |
| `results/final/` | held-out test: JSON plus per-environment `.npz` |
| `results/ablation/` | ablation results |
| `results/estimator*/` | estimator evaluation |
| `results/tables.md` | generated tables |
| `figures/` | figures |
| `logs/` | all logs |
| `runs/estimator/` | estimator training run |
| `results/_INVALID_C_estimator_head_zeroed/` | archived invalid C results, not used |

Full RSL-RL run directories, with every checkpoint and TensorBoard logs, are in `logs/rsl_rl/scratch_{A,C,D}/`, which git ignores because of their size. The estimator dataset (9.8 GB) is also gitignored.

**Safety:** no hardware was used. Physical deployment remains disabled by default and requires the existing supervised safety workflow.
