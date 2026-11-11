---
description: >-
  Class 06 student practice · run continuous cube yaw in simulation and
  optionally on a LEAP Hand
---

# Practice: continuous in-hand cube yaw

Run the supplied learned policy in a visible simulation, inspect its measured cube-yaw trace, and—only at a supervised teaching station—optionally compare it with the physical right LEAP Hand. You may complete the simulation route, observe the hardware route, or do both.

This is the continuous **positive-Z/yaw rotation** task. It is not a one-shot orientation-goal or arbitrary cube-reorientation task. The actor uses joint and previous-command history; it does not receive cube pose, RGB images, or a requested final orientation.

## Practice files

The [Class 06 continuous-yaw package](../README.md) is the root of this folder.

It is already extracted at the folder root (originally `Class06_Continuous_Yaw_Standalone`), containing the policy, complete MuJoCo model and assets, hardware bridge, safety configuration, checkpoint, tests, and original MJLab task/training source. No course-repository checkout is needed.

## Prepare the environment

{% tabs %}
{% tab title="Linux" %}
```bash
sudo apt update
sudo apt install -y curl libgl1 libglfw3
uv --version || (curl -LsSf https://astral.sh/uv/install.sh | sh)
source "$HOME/.local/bin/env"
cd "Leap Hand Manipulation"   # the folder root
uv sync
uv run pytest
```
{% endtab %}

{% tab title="macOS" %}
Install [Homebrew](https://brew.sh/) first if `brew --version` is unavailable.

```bash
brew install uv glfw
cd "Leap Hand Manipulation"   # the folder root
uv sync
uv run pytest
```
{% endtab %}
{% endtabs %}

Keep the terminal in the standalone folder.

## Choose a route

### Route A — run the checkpoint in the exact simulation

On Linux, launch a fresh `model_400.pt` rollout in the supplied full MJLab task:

```bash
uv run python simulation/run_yaw_rotation.py
```

This is the real policy/environment loop, not a recording. It opens MJLab's interactive MuJoCo viewer and runs task `Mjlab-Leap-Right-HandCube-MBZUAI-Rotate` with seed 11. The first run creates the separate Python 3.10 training environment and therefore downloads a substantially larger simulation stack. Use `--device cuda:0` on a compatible NVIDIA workstation.

The original MuJoCo-Warp training stack is Linux-oriented. For a lightweight Linux/macOS visual check without installing that stack, replay the exact known-good trajectory:

{% tabs %}
{% tab title="Linux" %}
```bash
uv run python simulation/replay_exact_mjlab_yaw.py
```
{% endtab %}

{% tab title="macOS" %}
```bash
uv run mjpython simulation/replay_exact_mjlab_yaw.py
```
{% endtab %}
{% endtabs %}

The portable viewer must show the cube held by the fingers and palm throughout the stable 30-second trajectory. It was generated with `model_400.pt`, seed 11 and 600 policy steps. The terminal prints duration, unwrapped yaw change, turns and minimum cube height. The included original preview is `reference/continuous_z_model400_preview_30s.mp4`.

This portable viewer is a smooth state replay, not a new policy rollout. `simulation/run_yaw_rotation.py` is a lightweight live native-MuJoCo evaluation that runs the exported actor directly on CPU without needing the heavy GPU/Warp training stack.

During the first \~5 seconds ($t = 0 \to 5.0$s), the history buffer warms up and the fingers adjust from the static pre-grasp pose into the palm pocket, causing rapid initial reorientation. Once settled ($t \ge 5.0$s), the fingers maintain a smooth, rhythmic positive-Z rotation at \~16.5°–20°/s for the remaining 25 seconds. To skip the initial settling transient and view only the steady continuous rotation:

{% tabs %}
{% tab title="Linux" %}
```bash
uv run python simulation/replay_exact_mjlab_yaw.py --start-time 5.0
```
{% endtab %}

{% tab title="macOS" %}
```bash
uv run mjpython simulation/replay_exact_mjlab_yaw.py --start-time 5.0
```
{% endtab %}
{% endtabs %}

For a slower visual replay without the settling transient, use:

{% tabs %}
{% tab title="Linux" %}
```bash
uv run python simulation/replay_exact_mjlab_yaw.py --start-time 5.0 --playback-speed 0.5
```
{% endtab %}

{% tab title="macOS" %}
```bash
uv run mjpython simulation/replay_exact_mjlab_yaw.py --start-time 5.0 --playback-speed 0.5
```
{% endtab %}
{% endtabs %}

### Route B — optional supervised hardware

Students do not operate the powered hand independently. A TA must inspect the station, keep the physical cutoff reachable, verify the right LEAP Hand's IDs 0–15 at 4 Mbps, use the actual persistent adapter path, and complete the mandatory torque-disabled shadow pass. Read `TA_GUIDE.md` before connecting power.

Install the optional serial dependency:

{% tabs %}
{% tab title="Linux" %}
```bash
uv sync --extra hardware
ls -l /dev/serial/by-id/
```
{% endtab %}

{% tab title="macOS" %}
```bash
uv sync --extra hardware
ls -l /dev/cu.usbserial-*
```
{% endtab %}
{% endtabs %}

Set the verified adapter path in each terminal:

{% tabs %}
{% tab title="Linux" %}
```bash
export LEAP_PORT="/dev/serial/by-id/REPLACE_WITH_THE_ACTUAL_ADAPTER"
```
{% endtab %}

{% tab title="macOS" %}
```bash
export LEAP_PORT="/dev/cu.usbserial-REPLACE_WITH_THE_ACTUAL_ADAPTER"
```
{% endtab %}
{% endtabs %}

In Terminal A, start the exact final Class 01 continuous-yaw server with `model_400.pt`:

{% tabs %}
{% tab title="Linux" %}
```bash
uv run python scripts/serve_policy.py \
  --task Mjlab-Leap-Right-HandCube-MBZUAI-Rotate \
  --checkpoint-file checkpoint/model_400.pt \
  --device cpu \
  --bind tcp://127.0.0.1:5555 \
  --keep-alive
```
{% endtab %}

{% tab title="macOS" %}
```bash
uv run python scripts/serve_policy.py \
  --task Mjlab-Leap-Right-HandCube-MBZUAI-Rotate \
  --checkpoint-file checkpoint/model_400.pt \
  --device cpu \
  --bind tcp://127.0.0.1:5555 \
  --keep-alive
```
{% endtab %}
{% endtabs %}

In Terminal B, the TA runs the full non-actuating shadow pass. Replace only the example adapter path:

{% tabs %}
{% tab title="Linux" %}
```bash
uv run python hardware/deploy_direct.py \
  --server-endpoint tcp://127.0.0.1:5555 \
  --port "$LEAP_PORT" \
  --baudrate 4000000 \
  --policy-hz 20 \
  --shadow \
  --max-steps 600 \
  --gain-profile H1 \
  --current-limit-ma 350 \
  --current-abort-ma 450 \
  --temperature-abort-c 60 \
  --log-file "outputs/shadow_$(date +%Y%m%d_%H%M%S).npz"
```
{% endtab %}

{% tab title="macOS" %}
```bash
uv run python hardware/deploy_direct.py \
  --server-endpoint tcp://127.0.0.1:5555 \
  --port "$LEAP_PORT" \
  --baudrate 4000000 \
  --policy-hz 20 \
  --shadow \
  --max-steps 600 \
  --gain-profile H1 \
  --current-limit-ma 350 \
  --current-abort-ma 450 \
  --temperature-abort-c 60 \
  --log-file "outputs/shadow_$(date +%Y%m%d_%H%M%S).npz"
```
{% endtab %}
{% endtabs %}

Only after the shadow log passes inspection, run the final six-second physical sequence:

{% tabs %}
{% tab title="Linux" %}
```bash
uv run python hardware/deploy_direct.py \
  --server-endpoint tcp://127.0.0.1:5555 \
  --port "$LEAP_PORT" \
  --baudrate 4000000 \
  --policy-hz 20 \
  --max-steps 120 \
  --gain-profile H1 \
  --start-delay-s 5 \
  --init-duration-s 5 \
  --return-home-duration-s 3 \
  --current-limit-ma 350 \
  --current-abort-ma 450 \
  --temperature-abort-c 60 \
  --log-file "outputs/physical_$(date +%Y%m%d_%H%M%S).npz"
```
{% endtab %}

{% tab title="macOS" %}
```bash
uv run python hardware/deploy_direct.py \
  --server-endpoint tcp://127.0.0.1:5555 \
  --port "$LEAP_PORT" \
  --baudrate 4000000 \
  --policy-hz 20 \
  --max-steps 120 \
  --gain-profile H1 \
  --start-delay-s 5 \
  --init-duration-s 5 \
  --return-home-duration-s 3 \
  --current-limit-ma 350 \
  --current-abort-ma 450 \
  --temperature-abort-c 60 \
  --log-file "outputs/physical_$(date +%Y%m%d_%H%M%S).npz"
```
{% endtab %}
{% endtabs %}

These deployment stages enforce safe execution boundaries. The three explicit current and temperature parameters implement the 350 mA configured operating limit, 450 mA abort threshold, and 60 °C thermal cutoff to safeguard the actuators during in-hand manipulation. Never weaken a guard during physical testing.

### Route C — both

Complete Route A first. Then observe Route B and compare the object-level simulation trace with the physical video and bridge telemetry. Joint tracking and current alone do not measure physical cube yaw; a calibrated, time-aligned object-pose record is required for that comparison.

## Inspect and extend the source

The archive includes every source needed by the portable run and the original training project:

* `policy/`, `simulation/`, `scripts/`, `hardware/`, and `models/` contain the runnable deployment path.
* `training_source/project/` contains the original MJLab task, reward/event definitions, robot configuration, training, playback, evaluation, export scripts, and its locked environment files.
* `training_source/training_params/` contains the exported environment and agent configurations.
* `checkpoint/model_400.pt` and `checkpoint/policy_weights.npz` preserve the original and portable actor representations.
* `reference/` contains the original stable preview and the exact seed-11 state trajectory used by the cross-platform viewer.

The portable viewer runs on Linux and macOS. The original accelerated training stack uses Python 3.10, MJLab/MuJoCo Warp, PyTorch, and GPU-oriented dependencies; use a compatible Linux/CUDA workstation for reproduction. Read `training_source/README.md` before changing it.

On a compatible Linux MJLab workstation, reproduce the original stable preview from scratch from `training_source/` with the bundle command documented in `training_source/README.md`. That full route is not presented as a macOS command because its original MuJoCo-Warp/CUDA environment is Linux-oriented.

## Submit one bounded evidence record

Record:

1. the exact task ID and observation/action contract;
2. the printed simulated duration, unwrapped yaw change, turns, and minimum cube height;
3. one interval in the visible trajectory where the cube advances and one where it stalls or reverses;
4. one finger-contact or command pattern visible in the viewer that could explain each interval;
5. whether you completed simulation, observed hardware, or both;
6. one code or training change you would test, with the measurement needed to decide whether it improved continuous yaw without increasing drops or unsafe actuation.

Clearly distinguish simulation evidence from physical hardware observation. In the hardware record, document operator interactions, reset events, and telemetry trends alongside visual observations.

## Return to the lesson

The [Class 06 lesson](lesson.md) explains the joint-order contract, history observation, contact mechanics, training evidence, and limits of a physical deployment claim.
