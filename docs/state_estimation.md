# Sensorless Object-State Estimation for Proprioceptive In-Hand Cube Manipulation

A research extension to the continuous positive-Z cube-rotation policy
(`Mjlab-Leap-Right-HandCube-MBZUAI-Rotate`, `checkpoint/model_400.pt`).

**Question.** The actor sees only 320 proprioceptive numbers (10 frames of joint
positions and 10 frames of its own previous commands). The critic sees the cube.
Does an explicit estimate of the cube's state, learned from privileged
simulation labels and computed from those same 320 numbers, help the policy
rotate the cube more reliably?

Everything here is additive. The original task, reward, PPO configuration,
`scripts/train.py` and `model_400.pt` are unchanged.

---

## 1. Installation (conda, Linux + NVIDIA)

```bash
bash training_source/project/conda/setup_conda_env.sh      # creates env "leap-estimator"
conda activate leap-estimator
cd training_source/project
python -m pytest tests/test_estimation.py -q                # CPU unit tests
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_estimation_env.py -q   # GPU env tests
```

The script installs:
- the project's exact `uv.lock` stack: MJLab v1.1.1, MuJoCo 3.8.1, MuJoCo-Warp 3.8.1, Warp 1.13.0, RSL-RL 4.0.1, NumPy 1.26.4,
- PyTorch **2.11.0+cu128**. The lockfile's torch 2.12.0 is a CUDA 13 build and needs a newer driver than the 570.x series,
- TensorBoard, Matplotlib, pytest, pyzmq and dynamixel-sdk,
- a **Warp compatibility hook** (`conda/leap_warp_compat.py`). MJLab v1.1.1 reads `warp.context.runtime.driver_version`, but Warp ≥ 1.12 no longer binds `warp.context`. Without the hook *every* CUDA run fails with `AttributeError`. CPU and macOS runs never reach that line, which is why the original setup didn't hit it.

`conda/requirements-leap-estimator.lock.txt` is a `pip freeze` of the tested environment.
The original repository tests (`tests/` at the repo root) also pass in this environment.

---

## 2. What was found in the existing system

| Item | Where | Finding |
|---|---|---|
| Actor observation | `ObservationManager`, group `actor` | `joint_pos` (160) then `prev_commanded_joint_pos` (160). Each block is 10 frames, oldest first. Joint positions are relative to the default pose with ±0.01 rad noise and 0–1 step delay. Commands are absolute. Verified in `tests/test_estimation_env.py`. |
| Actor normalisation | `rsl_rl.models.MLPModel.obs_normalizer` | `EmpiricalNormalization`: `(x − mean)/(std + 0.01)` |
| Critic (83) | `critic` group | joints (64) + `cube_pose_palm` (7) + linear/angular velocity in the palm frame (6) + size, mass, COM, friction (6) |
| Cube state in MuJoCo | `tasks/hand_cube/mdp/observations.py` | `cube_pose_in_palm_frame`, `cube_*_vel_in_palm_frame`. Palm frame: origin = centroid of `palm_collision_*` geoms, axes = palm body frame. |
| Fingertip contacts | sensor `fingertip_cube_contact` | `found` field, 4 slots: `if_tip, mf_tip, rf_tip, th_tip` |
| PPO / resume | `scripts/train.py --checkpoint-file` | `runner.load` loads actor, critic, Adam state and iteration. The actor class comes from `agent.actor.class_name`. |
| Parallel envs | `env.yaml`, `agent.yaml` | 1365 envs × 6 GPUs. Here: 4096 envs on one RTX 5000 Ada for each fine-tuning run. |

### 2.1 Reset-anchor bug in `cube_pose_deviation` (fixed here, opt-in)

`object_pose_rp_position_deviation_from_reset.reset()` stores the cube's
"episode-initial" pose. MJLab calls it *before* `sim.forward()`, so it reads
the **previous episode's final pose**. The grasp cache restores the cube in
many orientations, so on the first step the roll/pitch "deviation" usually
exceeds 0.8 rad and the episode ends again. This cascades until a stale pose
happens to match.

| 64 envs, 400 steps, `model_400` | pose-deviation terminations | timeouts |
|---|---|---|
| original task | 981 | 0 |
| corrected anchor | 66 | 31 |

The yaw reward in the same task already avoids this: it re-anchors on the
first post-reset step. `estimation/env_fixes.py` applies the same rule to the
termination, with identical thresholds. The fix is enabled in:
- the new task variants `…-FixedReset`, `…-Estimator` and `…-EstContact`,
- data collection (on by default),
- policy evaluation (`--env-variant fixed`, the default).

The original task is untouched, and the baseline is also reported on it
(`--env-variant original`). `model_400` was trained with the cascade present,
since MJLab is the same v1.1.1 tag as the course's lockfile.

Known limitation: with `init_at_random_ep_len` (PPO), environments that start
mid-episode keep the construction-time anchor until their first reset. This
produces a one-off burst of terminations in the first iteration, the same for
all runs. Evaluation resets every environment first, so it isn't affected.

### 2.2 Normalisation epsilon in the hardware runner (not changed; flagged)

`policy/yaw_policy.py` divides by `std + 1e-8`, while training used `std + 1e-2`.
Replaying the recorded seed-11 trajectory, 5.1% of clipped action entries
differ by more than 0.05, and some differ by up to 2.0 (full saturation flip).
The exported estimator bundles use the training normalisation. The test
`test_export_bundle_and_robot_runner` shows that setting the runner's
`_eps = 1e-2` makes it match `model_400` exactly. Fixing the existing runner is
left to the maintainers.

---

## 3. Architecture

```
            real-robot-available                           privileged (training only)
 q(t-9..t), cmd(t-9..t) ── obs_320 ─┬─ EmpiricalNorm ─┐
                                    │                 ├─ MLP 512-512-256 ─ action mean (16)
                                    └─ estimator ─────┘        (actor)
                                       (frozen)  19 features
                                                          critic(83) ─ MLP ─ V    (unchanged)
```

**Estimator** (`estimation/model.py`): MLP 320 → 512 → 512 → 256 with
LayerNorm and ELU, outputting 19 numbers:
- position (3, standardised),
- 6D rotation, made orthonormal with Gram–Schmidt (6),
- linear and angular velocity (3 + 3, standardised),
- 4 contact logits.

Input and target normalisation live inside the module.

**Loss** (`estimation/losses.py`):
`L = 1.0·Huber(pos) + 1.0·min_S‖R̂ − R·S‖²_F + 0.5·Huber(vel) + 0.5·BCE(contact)`.
`S` ranges over the cube's **24 rotational symmetries**. Proprioception
cannot tell a cube from its 90°-rotated copy; the raw orientation error stays
at about 130° while the symmetry-aware error is 8°. Contact labels on the
first step after a reset are masked out, because the sensor value predates
the reset.

**Augmented actor** (`estimation/policy.py`, `EstimatorAugmentedMLPModel`):
a subclass of RSL-RL's `MLPModel`. The 320 raw inputs keep their running
normaliser. The estimator features are appended *after* normalisation, using
the estimator's own fixed training statistics:
- position and velocities: standardised, clamped to ±5,
- rotation: unit 6D columns,
- contacts: probabilities.

That way the features can't drift while PPO updates its statistics. Feature
modes: `full` (19) and `contact` (4). The estimator is frozen: no gradients,
kept in eval mode, and it has no Adam state.

**Checkpoint migration** (`estimation/migration.py`): every tensor is mapped
explicitly.
- `mlp.0.weight` is widened from 320 to 320 + F columns; the new columns are zero.
- All other actor tensors, the normaliser, `std` and the critic are copied.
- Adam moments are re-indexed by parameter name, and the widened layer's moments are zero-padded.
- The iteration count (400) is kept.

The migrated policy is therefore *exactly* `model_400` at iteration 0 (unit
test). The unmodified `train.py --checkpoint-file` resumes from it.

**Deployment** (`estimation/deploy.py`, `policy/estimator_policy.py`,
`scripts/serve_estimator_policy.py`): a TorchScript bundle mapping
`obs_320 → (actions, state_17)`. It needs only `torch` and is a drop-in
subclass of the existing NumPy runner. The server speaks the same ZeroMQ
protocol as `scripts/serve_policy.py` and adds `estimated_cube_state` to each
reply. Nothing in this extension opens a serial port. Physical runs use the
existing supervised flow:
1. start the server,
2. run `hardware/deploy_direct.py --shadow` (torque off),
3. only then a short powered run, with the 350/450 mA and 60 °C limits unchanged (`TA_GUIDE.md`).

---

## 4. Data (Task 2)

`scripts/estimation/collect_estimator_data.py` runs `model_400` in the
training configuration: all domain randomisation, observation noise and
delays, grasp-cache resets, and random initial episode lengths. Half of the
environments sample from the policy's Gaussian (std ≈ 0.73), so the data
covers the PPO fine-tuning state distribution; the other half act
deterministically.

Each row holds the actor observation and labels read from the same physics
state. MJLab runs `sim.forward(); sim.sense()` immediately before computing
observations.

| split | seeds | envs × steps | rows |
|---|---|---|---|
| train | 1001–1006 | 2048 × 600 | 7,372,800 |
| val | 2001 | 1024 × 600 | 614,400 |
| test | 3001, 3002 | 1024 × 600 | 1,228,800 |

Splits are separated by seed, so they share no episode, environment instance
or startup-randomisation sample. The format is documented in
`estimation/dataset.py`. Each row also records the episode id, the step in the
episode, the end reason, the world-frame cube pose, and the randomisation
values in effect (cube mass, friction, COM offset, motor friction and damping,
effort limit, actuator delay).

---

## 5. Commands

All commands run from `training_source/project` with `leap-estimator` active.
`scripts/estimation/run_experiments.sh` runs all of them in order.

```bash
# Task 2: dataset (about 5 min on 3 GPUs)
GPUS="1 2 3" PY=python bash scripts/estimation/collect_all.sh

# Task 3: estimator (about 3 min on 1 GPU, plus data loading)
CUDA_VISIBLE_DEVICES=1 python scripts/estimation/train_estimator.py --run-name mlp_baseline
cp data/runs/estimator/<run>/best.pt data/estimator_models/cube_state_estimator.pt
tensorboard --logdir data/runs/estimator

# Task 4: independent estimator evaluation
CUDA_VISIBLE_DEVICES=1 python scripts/estimation/evaluate_estimator.py \
    --estimator data/estimator_models/cube_state_estimator.pt --output-dir data/eval/estimator

# Task 5: checkpoint migration
python scripts/estimation/migrate_checkpoint.py --estimator data/estimator_models/cube_state_estimator.pt \
    --feature-mode full    --output data/estimator_models/model_400_estimator_full.pt
python scripts/estimation/migrate_checkpoint.py --estimator data/estimator_models/cube_state_estimator.pt \
    --feature-mode contact --output data/estimator_models/model_400_estimator_contact.pt

# Task 6: fine-tuning (A' control, C full state, D contacts); 300 iterations x 4096 envs each
CUDA_VISIBLE_DEVICES=2 python scripts/train.py Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-FixedReset \
    --checkpoint-file $PWD/../../checkpoint/model_400.pt --env.scene.num-envs 4096 --agent.max-iterations 300
CUDA_VISIBLE_DEVICES=1 python scripts/train.py Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-Estimator \
    --checkpoint-file $PWD/data/estimator_models/model_400_estimator_full.pt --env.scene.num-envs 4096 --agent.max-iterations 300
CUDA_VISIBLE_DEVICES=3 python scripts/train.py Mjlab-Leap-Right-HandCube-MBZUAI-Rotate-EstContact \
    --checkpoint-file $PWD/data/estimator_models/model_400_estimator_contact.pt --env.scene.num-envs 4096 --agent.max-iterations 300

# Task 6: evaluation (identical held-out seeds 5001/5002, 2048 envs, 30 s, deterministic)
CUDA_VISIBLE_DEVICES=1 python scripts/estimation/evaluate_policies.py \
    --policy C_estimator=logs/rsl_rl/leap_right_mbzuai_rotate_estimator_full/<run>/model_700.pt \
    --seeds 5001 5002 --presets nominal heavy_slippery light_grippy high_delay weak_motors \
    --estimator data/estimator_models/cube_state_estimator.pt --output-dir data/eval/policies/C_estimator
python scripts/estimation/plot_results.py --eval data/eval/policies/*/results.json \
    --runs A_prime=<logdir> C_estimator=<logdir> D_contact=<logdir> --output-dir data/eval/report

# Task 7: deployment bundle
python scripts/estimation/export_estimator_policy.py --checkpoint <model_700.pt> \
    --output data/estimator_models/estimator_policy_full.pt
python ../../scripts/serve_estimator_policy.py --bundle data/estimator_models/estimator_policy_full.pt
```

---

## 6. Results

See section 8 (filled in from the executed runs).

---

## 7. Future real-world validation (camera / motion capture)

The deployed estimator gets no external measurement. To validate it on the
real hand *without* feeding ground truth into the policy:

1. **Measure independently.** Track the cube with the existing ArUco pipeline
   (`training_source/project/real_world/bridge_aruco_pose.py`, MBZUAI cube
   textures) or with a motion-capture rigid body (≥ 3 markers on the cube,
   ≥ 3 on the palm mount).
2. **Express in the palm frame.** Use the same convention as training: origin
   at the centroid of the `palm_collision_*` geoms, axes along the palm body.
   Calibrate camera→palm once (`real_world/camera_to_palm*.json`) and verify
   it by touching known palm points.
3. **Log, don't feed.** Run `serve_estimator_policy.py --state-log est.jsonl`
   while the camera process writes its own timestamped log. The policy and
   bridge never read the camera log.
4. **Align offline.** Interpolate the camera poses to the policy timestamps
   (20 Hz), compensating the measured camera latency. Then compute the same
   metrics as in simulation: position RMSE, *symmetry-aware* orientation error
   (the cube's printed faces only resolve the symmetry for the camera, not the
   policy), finite-difference yaw rate, and contact F1 against fingertip
   current spikes or video annotation.
5. **Report separately** for stable rotation, slips and drops (section 4
   definitions), and only from runs that followed the supervised shadow →
   powered procedure.

---

## 8. Results
