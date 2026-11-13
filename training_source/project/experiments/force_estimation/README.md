# Sensorless fingertip contact-force estimation (LEAP Hand)

**Goal:** estimate the compressive normal contact force between each of the four LEAP Hand fingertips and the cube, in newtons, using only the proprioceptive observation available on the real robot. MuJoCo contact forces are used **only as training labels**.

**Status:**

| Part | State |
|---|---|
| Implementation, label validation and tests | complete |
| Dataset collection (51 shards, about 3.9M rows), training (4 runs), selection and evaluation | complete (2026-11-13) |
| Results | `results/results.md`, `results/eval_report.json`, `figures/`; interpreted in Section 8 |
| Released model | `models/force_estimator.pt` (hurdle head, seed 0; sha256 in `models/force_estimator.sha256`) |
| Tests | 46 passed (12 force-estimation tests + 34 existing regression tests) |

**Headline (unseen policies):**
- MAE 0.19 N, RMSE 0.45 N, R² 0.78; ridge 0.30 N, best constant 0.41 N.
- Contact detection: AUROC 0.94–0.98.
- In contact, error is about 0.4 N, and forces above about 2 N are underestimated (Section 8).

This experiment is independent of the completed A/B/C/D cube-state study (`../scratch_state_estimation/`). Nothing in that study was modified (Section 9).

---

## 1. Interface

```python
from in_hand_rotation_mjlab.force_estimation.inference import ForceEstimatorRuntime
est = ForceEstimatorRuntime("experiments/force_estimation/models/force_estimator.pt", device="cpu")
forces = est.predict(obs320)   # (4,) or (B, 4): [thumb_force_N, index_force_N, middle_force_N, ring_force_N]
```

**Input:** the same 320-D actor observation as the policies and estimator B (`estimation/layout.py`), in the same order and with the same history semantics.

| Slice | Contents | Units |
|---|---|---|
| `obs[0:160]` | `joint_pos`: measured joint position minus the default pose. 10 frames × 16 joints, oldest to newest | rad |
| `obs[160:320]` | `prev_commanded_joint_pos`: absolute commanded joint target. 10 frames × 16 joints, oldest to newest | rad |

- Joint order is `estimation.layout.JOINT_NAMES` (`if_mcp, if_rot, if_pip, if_dip, mf_…, rf_…, th_cmc, th_axl, th_mcp, th_ipl`).
- MJLab's group history is term-major: all 10 frames of `joint_pos`, then all 10 frames of `prev_commanded_joint_pos`. The documentation sometimes describes this as "10 × (16 + 16)", but the actual order is term-major, and the estimator expects exactly that order.

**Normalization:** the per-input mean and std of the training split are stored inside the checkpoint, so callers pass raw observations.

**Output:**
- Order is thumb, index, middle, ring.
- Values are the compressive normal force between the whole distal link and the object, in N.
- Outputs are always finite and ≥ 0: softplus (direct head) or sigmoid × softplus (hurdle head).

**Checkpoint:**
- File: `models/force_estimator.pt` (format `leap_fingertip_force_estimator_v1`; config, weights, normalization), with its sha256 in `models/force_estimator.sha256`.
- The selection record is in `models/selection.json`.
- Load it with `force_estimation.model.load_force_estimator(path)`.

**Latency:** single-sample CPU inference takes about 0.12 ms (pilot measurement; the final value is in `results/results.md`).

**Example:** `scripts/inference_example.py`.

**Safety:** this is a standalone module. No hardware-control or deployment script was modified, and nothing is deployed.

## 2. Force labels (`src/in_hand_rotation_mjlab/force_estimation/labels.py`)

**Definitions:**

| Item | Definition |
|---|---|
| Fingertip | The whole distal body `<f>_ds` (`f` = th, if, mf, rf). It has four cube-collidable geoms: the `<f>_tip` mesh plus the three `<f>_ds_collision_*` boxes. The original `fingertip_cube_contact` sensor watches only the `_tip` meshes. |
| Per-contact force | Normal component of the contact-frame force (component 0). MuJoCo's elliptic cone (`opt.cone = 1`) makes it non-negative. |
| Per-finger force | Sum over all contacts between that body's geoms and `cube_geom`. Each contact is one (geom, geom) pair, so nothing is counted twice. |
| No contact | Exactly 0. |
| Units | N (MuJoCo SI). |

**Source:** a passive MJLab `ContactSensor`, added only in this experiment's environment factory (`make_force_env`):
- primary: the 4 distal bodies; secondary: the cube geom;
- `reduce="maxforce"`, 8 slots per finger;
- `history_length = decimation (10)`, which records the forces of every 5 ms physics substep.

**Labels stored per control step (20 Hz):**

| Label | Definition |
|---|---|
| `force_mean` (primary target) | Mean over the 10 substeps of the last control interval. This is the force actually applied by `mj_step`. |
| `force_now` | Force at the observation instant, after `env.step`'s final `forward()`. Read from the live `sensordata`, not the sensor's cache, which reward and metric terms fill before that `forward()`. |
| `force_max`, `contact_frac` | Maximum force, and fraction of substeps in contact, over the interval. |
| `found_now` | Contact count. |
| `tangential_now` | Friction-force magnitude. |
| `other_normal_now` | Normal force between the cube and non-fingertip links (palm, proximal and middle links). |

**Why the interval mean is the primary target:** contact forces chatter between substeps. In the pilot, `force_now` correlated only 0.47–0.81 with `force_mean`. The interval mean is the physically applied 50 ms average, matching the 20 Hz estimator.

**Resets:** the first row of each episode (`step_in_episode == 0`) has a zeroed substep history and a reset-state observation. These rows are excluded from training and evaluation, as in estimator B.

**Validation of the labels:**

| Check | Where | Result |
|---|---|---|
| Sensor sum vs. an independent pass over MuJoCo-Warp's raw contact list (`efc.force` at each contact's `efc_address`) | prototype (64 envs × 60 steps); every collector run audits this every 25 steps (`meta.json`) | max difference 1.9e-6 N (prototype), 0 N (pilot runs) |
| Slot overflow (more than 8 contacts per finger) | collector | 0 rows; at most 5 contacts per finger observed |
| Negative normal values clipped | collector | 0 |
| Force reported only when that finger's distal body touches the cube | `test_labels_correspond_to_actual_fingertip_cube_contacts` (GPU) | pass |
| Sensor equals `mj_contactForce` with multiple contacts over two geoms of one body; static box gives Σ normal force = m·g (N); no contact gives 0; adding the sensor leaves the dynamics bitwise identical | CPU MuJoCo tests | pass |

**Physics invariance:** on the GPU, two identical runs of the original environment already diverge. MuJoCo-Warp is not run-to-run deterministic, and the divergence is about 1e-3 at step 1. So passivity was established on CPU MuJoCo (bitwise identical). On the GPU, the with-sensor run differs from the baseline by the same amount as two baseline runs differ from each other.

## 3. Dataset (`scripts/collect_force_data.py`, plan in `scripts/run_pipeline.sh`, `data/collection_plan.txt`)

**Environment:** the completed study's environment, identical to A/C/D training:
- training DR, observation noise and delays, 20 s episodes;
- penalty curriculum at its final weights;
- grasp-cache resets, random initial episode phase.

**Policies:** the study's unchanged checkpoints. For each run, half of the environments sample from the policy Gaussian and half act on its mean. "std×2" runs double the sampling std, which produces more slips, drift and drops.

| Split | Policies | Checkpoints | Envs × steps × runs | Sim seeds |
|---|---|---|---|---|
| train | A seed 2, A seed 3, C seed 2, D seed 2 | 300, 600, 1000, 1000 std×2 | 192 × 600 × 16 | 21000–21015 |
| val (model selection) | same | 600, 1000 | 64 × 600 × 8 | 22000–22007 |
| **test** (held-out policies) | **A seed 1, C seed 3, D seed 1** | 300, 1000, 1000 std×2 | 160 × 600 × 9 | 23000–23008 |
| test_seen | A seed 2, C seed 2, D seed 2 | 1000 | 160 × 600 × 3 | 24000–24002 |
| robust_\<preset\> | A1, C3, D1 under heavy_slippery, light_grippy, high_delay, weak_motors, obs_noise_x3 | 1000 | 64 × 600 × 15 | 25000–25014 |

Notes:
- No simulation seed is shared between splits.
- The test policies never produced training or validation data.
- Using A, C and D policies covers different grasp styles; for example, C and D policies were trained with cube-state features.
- Final statistics, including contact rates, force quantiles, slot overflow and audit values, are in `results/eval_report.json` under `distribution`.

**Pilot distribution** (two 7.7k-row pilot runs, A seed 2 and C seed 2):

| Finger | Rows in contact (> 0.05 N) | Mean force in contact | p95 in contact |
|---|---|---|---|
| Thumb | 82–86% | 1.7–1.8 N | about 3.1–3.3 N |
| Index | 19–29% | 1.4–1.5 N | about 3.1 N |
| Middle | 41–44% | about 1.0 N | 2.2–2.5 N |
| Ring | 1–7% | about 1.3 N | 2.8 N |

- Rare impacts reach 25 N.
- The palm and other links carry about 2.5 N of normal force on average, so the fingertips are only part of the hand–cube interaction.

**Imbalance:** most index and ring rows have zero force. This is handled by:
- per-finger target scaling (RMS of the label);
- a Huber loss, which is robust to the heavy tail;
- a candidate hurdle head with a separate contact classifier (Section 4);
- evaluation that always reports in-contact and no-contact errors separately, plus errors by magnitude bin.

## 4. Model and training (`src/.../force_estimation/model.py`, `scripts/train_force_estimator.py`)

**Trunk:** MLP 320 → 512-512-256 (LayerNorm, ELU), the same trunk as estimator B for comparability. The weights are fully independent: no shared files or modules.

**Candidate heads:**
1. **direct:** softplus regression of force / scale.
2. **hurdle:** a contact logit and a softplus magnitude per finger. The prediction is the expected force p · m, trained with BCE (contact at > 0.05 N), a Huber loss on the magnitude in contact, and a Huber loss on the expected force.

**Training:**
- AdamW, learning rate 1e-3, weight decay 1e-4, cosine schedule, 40k steps at batch 4096, gradient clipping at 1.0.
- Input normalization and force scales come from the training split only.
- Non-finite losses or predictions abort the run.

**Selection:**
- Each head is trained with seeds 0 and 1, giving 4 runs.
- Each run's checkpoint is chosen by validation MAE; the final model is the run with the lowest validation MAE (`scripts/select_force_model.py`).
- Test splits are never read during training or selection.

The spec asked for a contact head only if it "meaningfully improves estimation". The two heads are compared on validation, and the comparison is reported (`models/selection.json`).

## 5. Evaluation (`scripts/evaluate_force_estimator.py`)

**Baselines** (fit on train; ridge λ chosen on validation):
- constant mean;
- constant median, the MAE-optimal constant;
- ridge regression on the standardized 320-D input, clipped at 0.

**Metrics:**
- MAE, RMSE, bias and R², overall and per finger.
- In-contact and no-contact errors.
- Contact transitions: within ±1 control step of a contact onset or offset.
- Slip-related rows: top 10% of cube speed.
- Force-magnitude bins.
- Contact classification: thresholds on predicted force chosen on validation; precision, recall, F1 and AUROC.
- Agreement with the instantaneous `force_now`.
- Breakdown by held-out policy and by DR quartile (friction, mass, actuator delay, effort limit).
- Robustness presets.

**Figures** (`figures/`):

| File | Contents |
|---|---|
| `force_distribution_train.png` | force distribution in the training split |
| `training_curves.png` | training curves |
| `pred_vs_true_{test,test_seen}.png` | predicted vs. true |
| `timeseries_{test,test_seen}.png` | predicted and true force over time |

## 6. Tests (`tests/test_force_estimation.py`)

**CPU:**
- multiple-contact aggregation, empty slots, no-contact zero, negative clipping;
- MuJoCo contact-sensor semantics: agreement with `mj_contactForce`, Σ = m·g in newtons, passivity;
- non-negative, finite outputs under extreme inputs, and the dimension check;
- checkpoint round-trip, reproducible inference and format check;
- independence from B.

**GPU:**
- privileged-information isolation: actor terms are exactly `joint_pos` and `prev_commanded_joint_pos`, 320-D;
- labels correspond to actual fingertip–cube contacts, sensor = raw contacts;
- dataset synchronization of a collected shard: shapes, row ↔ (t, env) alignment, reset masking, audit values.

**Results** (after training, `logs/pytest_final.log`):
- All 12 force-estimation tests pass, including the released-checkpoint test.
- The existing regression tests (`test_scratch_study.py`, `test_estimation.py`) all pass (34).
- Total: 46 passed.

## 7. Reproduction

From `training_source/project/`, in the conda environment `leap-estimator`:

```bash
sbatch experiments/force_estimation/scripts/run_pipeline.sh       # or: CUDA_VISIBLE_DEVICES=<g> bash .../run_pipeline.sh
```

`run_pipeline.sh` runs, in order:
1. collection, for all splits;
2. training: `direct` and `hurdle` heads × seeds 0 and 1;
3. selection;
4. evaluation;
5. the inference example;
6. the preservation check;
7. pytest.

It resumes where it left off: finished shards (`meta.json`) and finished runs (`summary.json`) are skipped. Logs are in `logs/`.

Individual steps:

```bash
python experiments/force_estimation/scripts/collect_force_data.py --condition A --checkpoint <pt> --policy-name A_seed2_it1000 --split train --seed 21002 --num-envs 192 --num-steps 600
python experiments/force_estimation/scripts/train_force_estimator.py --head direct --seed 0 --run-name direct_s0
python experiments/force_estimation/scripts/select_force_model.py
python experiments/force_estimation/scripts/evaluate_force_estimator.py
python experiments/force_estimation/scripts/inference_example.py
bash   experiments/force_estimation/scripts/check_preservation.sh
pytest tests/test_force_estimation.py
```

## 8. Results

All numbers come from `results/eval_report.json` and `results/results.md`, produced by `scripts/evaluate_force_estimator.py`. Unless stated otherwise:
- the split is `test`: held-out policies A seed 1, C seed 3 and D seed 1, none of which were used for training, validation or selection;
- the label is the interval-mean normal force;
- rows at `step_in_episode == 0` are excluded;
- "in contact" means a label above 0.05 N.

### 8.1 Dataset

51 shards. Sensor vs. raw-contact audit difference ≤ 1e-6 N in every shard; 0 rows with slot overflow; 0 negative normal values clipped.

| Split | Rows | Policies | Thumb: contact rate / mean in contact / p95 [N] | Index | Middle | Ring |
|---|---|---|---|---|---|---|
| train | 1,836,483 | A2, A3, C2, D2 × it 300/600/1000/1000-std×2 (16) | 0.80 / 1.69 / 3.20 | 0.19 / 1.49 / 3.38 | 0.40 / 1.01 / 2.23 | 0.03 / 1.37 / 3.19 |
| val | 306,198 | same policies, new sim seeds (8) | 0.80 / 1.64 / 3.07 | 0.18 / 1.46 / 3.31 | 0.42 / 0.96 / 2.07 | 0.03 / 1.29 / 2.86 |
| test (unseen policies) | 860,545 | A1, C3, D1 (9) | 0.84 / 1.74 / 3.29 | 0.26 / 1.35 / 3.12 | 0.30 / 0.98 / 2.24 | 0.10 / 1.68 / 3.52 |
| test_seen | 287,119 | A2, C2, D2 at it 1000, new sim seeds (3) | 0.79 / 1.61 / 3.02 | 0.17 / 1.51 / 3.46 | 0.45 / 0.97 / 2.10 | 0.04 / 1.33 / 3.04 |
| robust_* (5 presets) | about 114,800 each | A1, C3, D1 at it 1000 | see `results/results.md` | | | |

**Force imbalance:**
- The thumb is in contact about 80% of the time.
- The ring finger is in contact only about 3% of the time in training. Its share depends strongly on the policy: 0.5–12% in the training policies, but **27.6% for the held-out C seed 3**.
- Maximum forces reach 23–43 N, in brief impacts.
- The other hand links (palm, proximal and middle phalanges) press on the cube with a mean of about 2.4 N. This force is not part of the four targets.

### 8.2 Model selection

| Run | Best val MAE [N] | Test MAE [N] (reported only, not used) |
|---|---|---|
| **hurdle_s0 (selected)** | **0.1176** | 0.1870 |
| hurdle_s1 | 0.1177 | 0.1857 |
| direct_s1 | 0.1191 | 0.1936 |
| direct_s0 | 0.1202 | 0.1934 |

**Hurdle vs. direct head:**
- The hurdle head (contact probability × magnitude) is better than the direct head for both seeds: about 1.5% better on validation and about 3.5% on unseen-policy test.
- The gain is small but consistent.
- It also provides a contact probability (`contact_probability()`).
- It was therefore kept under the pre-declared rule: lowest validation MAE.
- It is not a large improvement. A direct head would be an acceptable, simpler alternative.

**Training behavior (`figures/training_curves.png`):**
- Validation MAE was still decreasing slowly at 40k steps, and the best checkpoint was the last or nearly the last.
- There was no instability: no non-finite loss or prediction, and no negative output.

### 8.3 Accuracy on unseen policies (`test`) vs. baselines

| Model | MAE [N] | RMSE [N] | R² | In-contact MAE | Transition MAE |
|---|---|---|---|---|---|
| **estimator** | **0.187** | **0.450** | **0.78** | **0.433** | **0.297** |
| ridge (λ = 10) | 0.301 | 0.570 | 0.64 | 0.554 | 0.415 |
| constant median | 0.413 | 0.832 | 0.23 | 0.958 | 0.496 |
| constant mean | 0.504 | 0.801 | 0.29 | 0.847 | 0.559 |

The estimator reduces MAE by 38% vs. ridge and by 55% vs. the best constant.

**Per finger** (estimator; thresholds for contact classification chosen on validation):

| Finger | MAE [N] | RMSE [N] | R² | In-contact MAE / bias [N] | No-contact MAE [N] | Contact F1 / AUROC | Ridge MAE |
|---|---|---|---|---|---|---|---|
| thumb | 0.338 | 0.604 | 0.69 | 0.377 / −0.09 | 0.129 | 0.96 / 0.97 | 0.525 |
| index | 0.155 | 0.404 | 0.74 | 0.486 / −0.19 | 0.036 | 0.84 / 0.96 | 0.236 |
| middle | 0.159 | 0.369 | 0.63 | 0.394 / −0.19 | 0.060 | 0.78 / 0.94 | 0.266 |
| ring | 0.096 | 0.381 | 0.59 | 0.901 / −0.73 | 0.011 | 0.76 / 0.98 | 0.179 |

The ring finger's low overall MAE is mostly due to its 90% no-contact rows. In contact it is the worst finger (see 8.5).

**By force magnitude** (estimator; n / MAE / bias in N):

| Label bin [N] | Thumb | Index | Middle | Ring |
|---|---|---|---|---|
| < 0.05 (no contact) | 136k / 0.13 / +0.13 | 634k / 0.04 / +0.04 | 606k / 0.06 / +0.06 | 778k / 0.01 / +0.01 |
| 0.05–0.5 | 59k / 0.28 / +0.17 | 45k / 0.27 / +0.03 | 69k / 0.25 / +0.05 | 9k / 0.32 / −0.07 |
| 0.5–1 | 92k / 0.32 / +0.10 | 53k / 0.39 / −0.10 | 81k / 0.33 / −0.13 | 14k / 0.62 / −0.40 |
| 1–2 | 324k / 0.30 / −0.00 | 81k / 0.48 / −0.14 | 86k / 0.43 / −0.27 | 35k / 0.88 / −0.72 |
| 2–4 | 237k / 0.44 / −0.25 | 45k / 0.67 / −0.43 | 17k / 0.84 / −0.72 | 23k / 1.12 / −0.98 |
| ≥ 4 | 11.5k / 2.15 / −2.10 | 3.7k / 2.36 / −2.27 | 1.4k / 3.25 / −3.24 | 2.5k / 2.65 / −2.64 |

Forces in the typical 0.5–2 N grasp range are estimated to about 0.3–0.5 N. Forces above about 2 N are increasingly **underestimated** (regression to the mean), and short impacts above 4 N are essentially missed.

**Contact transitions** (±1 control step of an onset or offset): MAE 0.297 N, positive bias +0.09 N. The estimate lags or blurs the edges by about one control step (`figures/timeseries_test.png`).

**Fast cube motion** (top 10% of cube speed, a proxy for slip): there is no degradation. Per-finger MAE is 0.33 / 0.12 / 0.11 / 0.07 N.

### 8.4 Seen vs. unseen policies, and robustness

| Split | Estimator MAE | RMSE | R² | In-contact MAE | Ridge MAE |
|---|---|---|---|---|---|
| test_seen (training policies, new seeds) | 0.116 | 0.332 | 0.86 | 0.283 | 0.242 |
| **test (unseen policies)** | **0.187** | **0.450** | **0.78** | **0.433** | 0.301 |
| robust_heavy_slippery | 0.186 | 0.443 | 0.74 | 0.431 | 0.297 |
| robust_light_grippy | 0.201 | 0.462 | 0.76 | 0.461 | 0.309 |
| robust_high_delay | 0.196 | 0.448 | 0.75 | 0.454 | 0.310 |
| robust_weak_motors | 0.193 | 0.448 | 0.75 | 0.445 | 0.304 |
| robust_obs_noise_x3 | 0.209 | 0.480 | 0.72 | 0.481 | 0.332 |

**The main generalization gap is the policy, not the physics.**
- Moving to unseen policies raises MAE by 61% (0.116 → 0.187 N).
- The five robustness presets add only 0–12% on top of the unseen-policy MAE.
- MAE is flat across DR quartiles of friction (0.18–0.19), cube mass, action delay and effort limit.

**Per held-out policy (MAE by finger: thumb / index / middle / ring):**

| Held-out policy | Thumb | Index | Middle | Ring |
|---|---|---|---|---|
| A seed 1 | 0.30 | 0.16 | 0.18 | 0.01 |
| C seed 3 | 0.38 | 0.20 | 0.13 | **0.24** |
| D seed 1 | 0.33 | 0.11 | 0.16 | 0.03 |

C seed 3 uses its ring finger in 27.6% of steps, more than twice any training policy (≤ 12%). The ring in-contact error and bias (−0.73 N) on `test` come mostly from this out-of-distribution gait.

### 8.5 Observability: what can and cannot be estimated

**What the estimator cannot see:**
- The only inputs are joint positions and commanded positions. Under position (PD) control, the tracking error (command − position) is the only proxy for actuator torque, and torque is the only path by which fingertip force is visible to the estimator.
- Several things are therefore not separable from the observation:
  - **gravity and inertia loads on the finger;**
  - **forces carried by non-fingertip links** (palm and phalanges, about 2.4 N on average);
  - **the direction of the contact normal** (only the torque projection is visible);
  - **the true effort limit and gains** (DR).

**Consequences seen in the data:**
1. **Interval mean vs. instantaneous force.** The interval-mean force over each 50 ms control step is estimable (R² 0.78). The instantaneous force at the observation time is not: against `force_now` the same predictions give R² only 0.23–0.45 per finger, because MuJoCo contact forces chatter within a control step.
2. **Contact presence** is well observable (AUROC 0.94–0.98), so contact and no-contact decisions are reliable. On frames with no contact the estimate stays near 0: mean 0.01–0.13 N.
3. **Magnitude in contact** is moderately observable. The in-contact MAE of 0.38–0.49 N for thumb, index and middle is 22–36% of the typical in-contact force. Large forces and impacts are biased low.
4. **The ring finger in contact** is poorly estimated for unseen gaits (R² < 0 in contact): there are too few ring-contact examples in training (3%).

### 8.6 Latency

- About 0.12 ms per single-sample call on CPU (`ForceEstimatorRuntime.predict`, including numpy↔torch conversion; `logs/inference_example.log`).
- Well within the 50 ms control period.
- The checkpoint is 2.3 MB.

### 8.7 Limitations and future integration

**Limitations:**
- **Simulation only.** No real-robot validation: real fingertip loads, cable and backlash effects, and Dynamixel current behavior are not modelled.
- **Joint positions only.** Motor currents are not used. On hardware, adding measured current (or a current-based torque estimate) to the input is the most promising route to better force observability. That would require new data and a new model, since the 320-D input order is fixed here.
- **Policy dependence.** Accuracy degrades for gaits outside the training policies, especially fingers used more than in training. Training on more, and more diverse, policies (including real-robot rollouts) is the most effective improvement.
- **Normal force only.** Only total normal force per distal link is estimated; there is no tangential or friction component, no contact location, and no palm or phalanx forces.
- **Regression to the mean.** Forces above about 2 N and impacts are underestimated, so the estimator should not be used to detect excessive grip force.
- **Sim-specific labels.** Labels depend on MuJoCo's soft-contact model (solref/solimp). Absolute newton values transfer only as well as that model matches the real hand.

**Integration (not done here):**
- The estimator is a pure function of the observation vector the deployed policy already builds: same 320-D order and units, no extra sensors.
- It can run alongside the policy at 20 Hz and be used for monitoring and logging, or as a slip or grasp-loss indicator via the contact probability.
- No deployment or hardware script was modified.

## 9. Preservation of the completed study

Before any force-estimation work started, at commit 310ce80, the following baseline fingerprints were recorded in `preservation/`:
- sha256 of every git-tracked file (753);
- sha256 of every RSL-RL checkpoint under `logs/rsl_rl/` (303, including all A/C/D runs and the pilots);
- path, size and mtime of both existing datasets.

`scripts/check_preservation.sh` re-verifies them. Before collection started, the result was:
- all 752 tracked files unchanged, excluding `.gitignore`, which only gained ignore lines for this experiment's dataset;
- all 303 checkpoints unchanged;
- both datasets unchanged.

The pipeline repeated this check after collection, training and evaluation (`logs/preservation_check.log`). Everything was still unchanged: 752 tracked files, 303 checkpoints and both datasets; the only modified tracked file is `.gitignore`.

**What changed in the repository:**
- New files only:
  - `src/in_hand_rotation_mjlab/force_estimation/`;
  - `experiments/force_estimation/`;
  - `tests/test_force_estimation.py`.
- Lines appended to `.gitignore`.

No existing source file, environment, reward, termination, action, observation, PPO configuration, estimator, checkpoint, dataset, figure, report or hardware script was edited. The force sensor exists only in `make_force_env`, which is used by this experiment alone; registered tasks are deep-copied, never mutated.
