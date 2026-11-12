# Sensorless cube-state estimation for LEAP Hand cube rotation: a from-scratch study

**Question.** Can explicit cube-state estimation from LEAP Hand proprioception alone improve continuous in-hand cube rotation, compared with proprioception-only reinforcement learning?

**Status: this document is being filled in as stages complete.** Sections marked *(pending)* have not been executed yet. Nothing in them is a result.

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
- C seeds 1–3 are retrained from scratch with the fix, using exactly the A/D settings and budget, uninterrupted (`scripts/sbatch_C_fresh.sh`).

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
