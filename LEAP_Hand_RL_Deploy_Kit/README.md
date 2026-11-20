# LEAP Hand RL Deploy Kit

This kit runs a reinforcement-learning policy trained in MJLab with rsl_rl on a
physical **right LEAP Hand** (16 Dynamixel motors). Copy your checkpoint into
one folder and run four commands.

You don't need ROS, MJLab, MuJoCo, PyTorch, a GPU, or internet access on the
robot computer. The kit only needs Python 3.10 or newer with numpy. The
Dynamixel SDK and pyserial come bundled with it.

---

## 1. One-time setup (fresh Linux computer)

```bash
unzip LEAP_Hand_RL_Deploy_Kit.zip && cd LEAP_Hand_RL_Deploy_Kit
# optional: activate the Python env you use for RL first, or set LEAP_PYTHON=/path/to/python
./install.sh
```

`install.sh` picks a Python (3.10 or newer), unpacks the bundled Dynamixel SDK,
gives your user access to USB-serial ports (`dialout` group), sets the FTDI
latency timer to 1 ms, and runs an offline self-test. **If it adds you to
`dialout`, log out and back in once.**

Check the whole kit without any hardware:

```bash
./leap.sh test     # policy check + the full control loop on a simulated motor bus
```

## 2. Put your RL files in

Copy these files from your training run, for example
`logs/rsl_rl/<experiment>/<run>/`, into **`PUT_YOUR_RL_FILES_HERE/`**:

| file | needed? | why |
|---|---|---|
| `model_XXXX.pt` | **yes** | the trained policy (the highest iteration is used if there are several) |
| `params/env.yaml` | strongly recommended | observation terms, history length, action scale, default pose, control rate |
| `params/agent.yaml` | recommended | network activation, action clipping |

If you leave out `env.yaml`, the kit assumes the reference task
(`Mjlab-Leap-Right-HandCube-MBZUAI-Rotate`): joint position plus previous
command, history 10, ±1/24 rad per step, 20 Hz.

## 3. Deploy

```bash
./leap.sh inspect   # does the policy load and give sane commands? (no hardware)
./leap.sh check     # power the hand: are all 16 motors (IDs 0-15, 4 Mbps) there? torque stays OFF
./leap.sh shadow    # 30 s: real encoders -> policy -> logged commands, torque stays OFF
./leap.sh run       # REAL MOTION: 6 s, you must type YES
```

* `run` refuses to start until that exact checkpoint has finished a shadow run.
* During the 5 s countdown, place the object in the hand. The hand then ramps to
  the policy's start pose over 5 s, runs the policy, and returns to the flat pose.
* **Ctrl+C** at any time returns the hand home and turns torque off. Keep the
  12 V power cutoff within reach anyway.
* Each run writes a log to `logs/`. Run `./leap.sh report` to summarise the newest one.

Make runs longer only after short runs look good:
`./leap.sh run -- --max-steps 600` (30 s), or change `RUN_STEPS` in `configs/hand.env`.

## 4. Station settings: `configs/hand.env`

| setting | default | notes |
|---|---|---|
| `LEAP_PORT` | `auto` | picks the FTDI (U2D2) adapter; set it to `/dev/serial/by-id/...` if you have several |
| `LEAP_BAUD` | `4000000` | LEAP factory setting |
| `LEAP_GAIN_PROFILE` | `H1` | `H1` = P600/D150, `H2` = P800/D150; use the one your sim was identified with |
| `CURRENT_LIMIT_MA` / `CURRENT_ABORT_MA` | 350 / 450 | motor current limit and abort threshold |
| `TEMPERATURE_ABORT_C` | 60 | |
| `TRACKING_ERROR_ABORT_RAD` | 0.65 | aborts if a joint is stuck far from its command |
| `MAX_COMMAND_STEP_RAD` | 0.05 | aborts if the policy jumps more than this in one step |

Every other bridge flag can be passed after `--`. To list them all:
`PYTHONPATH=kit/_deps $(cat .python) kit/deploy_direct.py --help`.

## Additions in this repository (Team 3, 17 Nov 2026)

* `./leap.sh functest [sim]`: basic hardware function test with no policy (`kit/function_test.py`).
* `squash: tanh` in `env.yaml` is honoured, and `clip_actions: null` in `agent.yaml` now means "no clip" (it was treated as 1.0).
* Estimator-augmented policies: a `cube_estimator.npz` next to the checkpoint adds the 19 cube-state features (`kit/cube_estimator.py`, checked against the training features on every load).
* `policies/{A,C}_seed{1,2,3}/`: the from-scratch study policies, ready to run.
* Session report: `../docs/hardware_session_2026-11-17.md`.

## Live view: estimated cube + fingertip forces (added 18 Nov 2026)

Every `shadow` / `run` now also logs, for **any** policy (A, C, model_400), two
estimates computed from the same 320 joint numbers the policy reads: the cube
state (estimator B) and the normal force at each fingertip (force estimator).
They are display/log only and never reach the policy or the motors.
`live` adds a 3D window next to the hand:

```bash
./leap.sh inspect policies/hwlim_C_seed1                 # new C (trained inside the kit limits)
./leap.sh live shadow policies/hwlim_C_seed1             # torque off: move the cube/fingers by hand
./leap.sh live run    policies/hwlim_C_seed1             # powered, 6 s; same safety flow as `run`
./leap.sh live run    policies/hwlim_A_seed1 -- --max-steps 600   # baseline A, 30 s
./leap.sh view                                           # replay the newest log in 3D
./leap.sh analyze                                        # forces vs motor current, newest log
```

| In the window | Meaning |
|---|---|
| hand | measured encoder angles (mirrors the real hand) |
| orange see-through cube | estimated palm-frame cube pose, kept continuous over the cube's 24 symmetries (face identity is not observable, so the cube is plain) |
| arrows into the cube | estimated fingertip normal force, 3 cm per N, labelled `T/I/M/R x.xxN` |
| spheres on fingertips | force estimator's contact probability (size and opacity) |
| top-left text | policy, SHADOW/POWERED, time, **est. turns** (+ = the direction the policy was trained to turn), est. cube position (mm) |
| top-right table | per finger: F_est [N], contact p (force est.), contact p (estimator B), finger motor current [mA], F(I) [N] |
| bottom plots | last 10 s of estimated forces and of forces implied by motor current |

`F(I)` is an independent physics cross-check: joint torque = Kt × current
(`--kt`, default 0.5 N m/A, approximate), fingertip force from J^T F = τ
(damped least squares). It also contains gravity, friction and tracking effort
and saturates at the current limit, so use it for trends (correlation), not
absolute newtons. `analyze` reports r(F_est, current), r(F_est, F(I)) and the
current with vs without estimated contact, and writes `<log>_analysis.json/.png`.

* Viewer and analysis need `mujoco` (the repo's `.venv`; `uv sync` at the repo
  root). The control loop does not; it streams UDP packets to
  `LEAP_LIVE_UDP` (configs/hand.env) and never waits for the viewer.
* Older logs work too: `view` / `analyze` re-run the estimators on the logged
  joints and commands.
* `policies/hwlim_{A,C}_seed{1,2,3}` link to the study's deploy folders
  (`training_source/project/experiments/hwlimit_ac/deploy/policies/`).

## Policy comparison trials: A3, C5, R1, P0 (added 20 Nov 2026)

Protocol, questions and observer instructions are in `../docs/hardware_trials_A3_C5_R1_P0.md`.

```bash
./leap.sh trial-setup      # one shadow run per policy (unlocks them)
./leap.sh trial next       # next policy in the counterbalanced schedule; 60 s live run; then asks what you saw
./leap.sh trial-plan       # progress
./leap.sh compare          # Q1-Q4 -> logs/trial_comparison.md
```

`TRIAL_STEPS` (default 1200 = 60 s) and `TRIAL_BLOCKS` (default 5) are in `configs/hand.env`.

## What policies work

The kit supports any rsl_rl MLP actor (any depth, width, or activation, with or
without observation normalization; rsl_rl 1.x, 2.x, or 3.x checkpoint layout)
whose actor observations use only hand proprioception, in MJLab's right-hand
joint order:

`joint_pos` (relative to the default pose), `joint_vel`, `prev_commanded_joint_pos`,
`joint_pos_error`, `last_action`, with any history length.

Actions can be **delta** joint-position (`delta_min`/`delta_max` in env.yaml,
like the reference task) or **absolute** joint-position (`scale`, with an
optional default offset).

`./leap.sh inspect` **rejects policies that need cube pose, camera, goal, or
touch inputs.** The bare hand can't provide those, and feeding them zeros would
be unsafe.

## How it maps sim to hardware

* Joint order: hardware motors 0..15 are
  `[if_rot, if_mcp, if_pip, if_dip, mf_rot, mf_mcp, ..., th_cmc, th_axl, th_mcp, th_ipl]`.
  MJLab swaps rot/mcp for each finger. The bridge does this remapping automatically.
* Angle zero: raw encoder 2048 = 0 rad, which is the flat, open hand (the LEAP
  convention). Run `./leap.sh check` with the hand lying flat: every joint
  should read close to 0.
* Inference copies rsl_rl exactly: `(obs - mean) / (std + 0.01)`, then the MLP,
  then the mean action (no sampling noise), clipped to [-1, 1] and mapped to
  joint targets. It was checked against rsl_rl's own modules (max difference
  1.7e-4).
* Each step, the command actually sent to the motors (after safety clipping) is
  fed back as the policy's "previous command", so the observation history
  matches what the hand really did.
* Hardware limits from `leap_hand_controller` are enforced on top of the
  policy's own limits.

## Folder map

```
install.sh                   one-time setup
leap.sh                      inspect / check / shadow / run / report / test
PUT_YOUR_RL_FILES_HERE/      your model_XXXX.pt (+ env.yaml, agent.yaml)
configs/hand.env             port, gains, run length, safety limits
configs/leap_right_gain_profiles/H1.json, H2.json
examples/reference_policy/   known-good continuous cube-yaw policy (model_400.pt + yaml)
kit/policy_runtime.py        torch-free checkpoint reader + rsl_rl-equivalent actor
kit/deploy_direct.py         guarded Dynamixel control loop (current/temp/velocity/tracking guards)
kit/check_hand.py            read-only motor/port diagnostic
kit/inspect_policy.py        offline policy check (+ optional --export-npz)
kit/report.py                log summary
kit/fake_dynamixel/          simulated motor bus used only by ./leap.sh test
vendor/wheels/               dynamixel-sdk 3.8.4 + pyserial 3.5 (offline install)
logs/                        run logs (.npz)
```

## Troubleshooting

| symptom | fix |
|---|---|
| `Cannot open /dev/ttyUSB0` / permission denied | log out and back in after `install.sh`; or `sudo usermod -aG dialout $USER` |
| `/dev/ttyUSB0` never appears (Ubuntu 22.04+) | `sudo apt remove brltty` |
| `Motor ping failed for IDs [...]` | check power and the daisy-chain cable; `./leap.sh check` scans other baud rates and IDs |
| a joint reads about ±1.57 or ±3.14 rad with the hand flat | that motor's horn is mounted one or more 90° steps off; fix it mechanically |
| `Command-step guard tripped` | the policy jumped too far in one step; look at the shadow log before raising `MAX_COMMAND_STEP_RAD` |
| `Tracking-error guard tripped` | a finger is blocked or the gains are too low; check the gain profile and that nothing jams |
| loop slower than 50 ms | the FTDI latency timer is not 1 ms: `cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer` |
| `Policy was trained at X Hz` | pass `-- --policy-hz X` (the kit reads the training rate from env.yaml) |
