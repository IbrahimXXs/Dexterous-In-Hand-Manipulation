# Class 06 TA guide — final continuous-yaw hardware route

Use only a right LEAP Hand whose sixteen Dynamixel motors are deliberately configured as IDs 0–15 at 4 Mbps. Keep the 12 V cutoff reachable, clear the hand workspace, inspect the cube and fingers, and do not operate alone.

## Preflight

1. Identify the persistent adapter path. Linux should use `/dev/serial/by-id/...`; macOS should use `/dev/cu.usbserial-*`, not a transient Linux `/dev/ttyUSB0` path.
2. Confirm no other process owns the device.
3. Install the hardware extra with `uv sync --extra hardware`.
4. Start the exact final Class 01 policy server with `model_400.pt`.
5. Run the complete torque-disabled shadow trial and inspect its log.
6. Only then run a six-second (`120` step at `20 Hz`) supervised actuated trial.

On the teaching Linux station, set the persistent adapter once in each terminal:

```bash
export LEAP_PORT="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBIHRNS-if00-port0"
```

Terminal A:

```bash
uv run python scripts/serve_policy.py \
  --task Mjlab-Leap-Right-HandCube-MBZUAI-Rotate \
  --checkpoint-file checkpoint/model_400.pt \
  --device cpu \
  --bind tcp://127.0.0.1:5555 \
  --keep-alive
```

Terminal B, mandatory shadow pass:

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

Short actuated trial, only after the shadow pass succeeds:

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

This is the final Class 01 hardware sequence and checkpoint. The source TA guide declares a 450 mA current abort and 60 °C thermal abort, while the bridge's bare CLI defaults are 700 mA and 65 °C and its current-limit default is 450 mA. The explicit `350/450/60` arguments above make the documented safety settings effective and satisfy the requirement that abort current exceed configured current limit.

Do not weaken a guard to force completion. Stop on unexpected motion, communication errors, overheating, repeated stalls, excessive current or tracking error. The bridge ramps from measured state, applies the declared limits, records NPZ telemetry and returns toward the initial pose after completion or interruption when communication remains available.

