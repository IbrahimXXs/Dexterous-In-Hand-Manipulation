# LEAP Hand: continuous cube yaw rotation

This folder holds the right LEAP Hand policy that keeps a 58 mm cube spinning about the vertical axis (positive Z / yaw). It includes the trained policy (`model_400.pt`), the MJLab training source, the reward functions, a portable MuJoCo viewer, and the guarded hardware bridge.

The policy only sees the hand's joint positions and its own previous commands. It never sees the cube's pose, so it is not a "turn the cube to a target" policy. That separate goal-conditioned task (Class 07, `model_4900.pt`) and the other course material are kept in a local `etc/` folder, which is not part of this repository.

---

## Folder map

```text
Leap Hand Manipulation/
├── README.md  STUDENT_GUIDE.md  TA_GUIDE.md      (TA_GUIDE = hardware safety procedure)
├── pyproject.toml  uv.lock                       lightweight env: viewer, policy, hardware bridge (Python 3.11–3.12)
│
├── checkpoint/          model_400.pt (trained policy) + policy_weights.npz (same actor for NumPy)
├── training_source/     TRAINING: the original MJLab task, rewards, curriculum, train/eval/export scripts
│   ├── project/             own pyproject + uv.lock (Python 3.10, MJLab, MuJoCo-Warp, PyTorch)
│   └── training_params/     the exact env.yaml + agent.yaml that produced model_400.pt
├── policy/              actor inference (NumPy or PyTorch)
├── scripts/             launchers (setup, replay, live policy, training) + serve_policy.py (ZeroMQ server)
├── simulation/          replay_exact_mjlab_yaw.py (faithful) · run_yaw_rotation.py (live approximation)
├── models/              portable MuJoCo LEAP hand + 58 mm MBZUAI cube scene
├── reference/           exact seed-11 trajectory + 30 s preview MP4
├── hardware/  config/   guarded 16-motor Dynamixel bridge · gain profiles H1/H2
├── tests/               package, model and policy checks
├── outputs/             written by the simulation scripts (git-ignored)
├── docs/                Class 06 lesson.md + practice.md, with figures/ and recordings/
├── evidence/            recorded runs: videos, hardware and sim frames, telemetry, photos
├── LICENSES/            MIT licence for the LEAP hand assets
│
└── etc/                 local only, not in this repository: Class 07 packages and lessons, project worksheets
```

---

## Quick start (works on macOS)

Prerequisite: [`uv`](https://docs.astral.sh/uv/).

```bash
scripts/setup.sh                                   # uv sync + self-tests
scripts/replay.sh                                  # watch the exact 30 s trajectory in MuJoCo
scripts/replay.sh --start-time 5                   # skip the 5 s grasp-settling phase
scripts/replay.sh --start-time 5 --playback-speed 0.5 --loop
scripts/live_policy.sh                             # run the actor live (lighter approximation)
```

`replay.sh` plays back states recorded from the full MJLab run with `model_400.pt`, seed 11 and 600 steps. `live_policy.sh` runs the NumPy actor in plain MuJoCo, without MJLab's delays and randomisation, so it behaves differently (see *Status* below).

To run the trained policy live in the exact environment it was trained in, matching `reference/continuous_z_model400_preview_30s.mp4`:

```bash
scripts/setup_training.sh                          # once: the MJLab stack
scripts/play_mjlab.sh                              # model_400.pt live in MuJoCo's viewer (CUDA if present, else CPU)
scripts/play_mjlab.sh --checkpoint-file /abs/path/model_XXXX.pt     # a policy you trained
PLAY_DEBUG_VIS=1 scripts/play_mjlab.sh             # also show the world/palm/cube debug axes
```

---

## Training

| What | Where |
|---|---|
| Task registration (`Mjlab-Leap-Right-HandCube-MBZUAI-Rotate`) | `training_source/project/src/in_hand_rotation_mjlab/tasks/hand_cube/config/leap_right_mbzuai_rotate/` |
| Reward terms and weights | `…/tasks/hand_cube/hand_cube_env_cfg.py` (`rewards = {…}`) |
| Penalty curriculum (sets the four penalty weights during training) | same file, `curriculum = {…}` |
| Reward function code | `…/tasks/hand_cube/mdp/rewards.py` |
| PPO settings | `…/config/leap_right_mbzuai_rotate/rl_cfg.py` (inherits `leap_right_upstream_baseline/rl_cfg.py`) |
| Settings used for `model_400.pt` | `training_source/training_params/env.yaml`, `agent.yaml` |

```bash
scripts/setup_training.sh                          # Python 3.10 + MJLab + MuJoCo-Warp + PyTorch
scripts/train.sh --help                            # list every option you can override
scripts/train.sh --checkpoint-file "$PWD/checkpoint/model_400.pt"   # fine-tune the shipped policy
```

Runs are saved to `training_source/project/logs/rsl_rl/leap_right_hand_cube_mbzuai_rotate/<timestamp>/`.

Real training needs Linux with an NVIDIA GPU. `model_400.pt` was trained with 8192 parallel simulations on 6 GPUs. On a Mac, training runs on the CPU and is only useful for checking that a change doesn't crash. See [`training_source/README.md`](training_source/README.md) for the exact command that recorded the reference trajectory.

> **Dependency fix.** The course pinned a MuJoCo nightly build that has since been deleted upstream. `training_source/project/` now pins the stable `mujoco==3.8.1`, `mujoco-warp==3.8.1` and `warp-lang==1.13.0`. The untouched originals are kept as `*.course-original`.

---

## Physical LEAP Hand (supervised only)

None of the `scripts/*.sh` launchers touch hardware. The hardware path is: start `scripts/serve_policy.py` (policy server), then run `hardware/deploy_direct.py --shadow` (a pass with motor torque off), and only then do a short powered run. Install the extras with `uv sync --extra hardware`. The exact commands and the 350 mA / 450 mA / 60 °C limits are in `docs/practice.md` (Route B) and `TA_GUIDE.md`.

Safety rules from the course:
- Use the right LEAP Hand with motor IDs 0–15 at 4 Mbps, on its persistent serial path (`/dev/cu.usbserial-*` on macOS).
- Always do a pass with motor torque off first.
- Keep an independent power cutoff within reach.
- Never weaken a current or temperature limit.

---

## Status on this Mac (macOS, Apple Silicon)

| Step | Result |
|---|---|
| `setup.sh` | ✅ installed, 2/2 tests pass |
| `replay.sh` (no viewer) | ✅ 30 s, 601 frames, +8.22 rad (1.31 turns), cube stays in hand and touches the palm and fingers |
| `live_policy.sh` (no viewer, 30 s) | ⚠️ keeps the cube and turns it the right way, but only 0.09 turns (MJLab: 1.31). This simplified run leaves out MJLab's actuator and delay model |
| `play_mjlab.sh` / `record_policy_video.py --seed 11 --mbzuai-cube` (CPU) | ✅ full MJLab run of `model_400`; frames match the reference video |
| Training | not run (meant for a Linux NVIDIA machine) |

**Fixed 2026-11-11:** `models/scene_right_mbzuai_cube.xml` placed the hand 10 cm lower than MJLab does. MJLab mounts the palm on a base at z = 0.1 m, so the palm sits at z = 0.2 m. As a result, the replay drew the cube floating about 25 mm above the fingers, and the live simulation dropped the cube onto the hand, which caused the earlier "−2.4 turns" result. The scene now attaches the hand to a `mocap_base` body at z = 0.1 m.
| Physical LEAP Hand | not touched |
