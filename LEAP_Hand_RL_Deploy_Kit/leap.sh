#!/usr/bin/env bash
# LEAP Hand RL Deploy Kit - single entry point.
#
#   ./leap.sh inspect [policy_dir]   check your policy (no hardware)
#   ./leap.sh check                  read all 16 motors (torque stays OFF)
#   ./leap.sh shadow  [policy_dir]   policy on live encoders, torque stays OFF
#   ./leap.sh run     [policy_dir]   REAL MOTION (needs a passed shadow run)
#   ./leap.sh report  [log.npz]      summarise the newest (or given) run log
#   ./leap.sh test                   offline self-test with the reference policy
#   ./leap.sh functest [sim]         basic hardware function test, no policy (MOVES the hand)
#   ./leap.sh live shadow|run [policy_dir]  same as shadow / run + live 3D viewer
#                                    (estimated cube, fingertip forces, motor current)
#   ./leap.sh live sim [policy_dir]  try the viewer with NO hand (simulated motor bus)
#   ./leap.sh view    [log.npz]      replay a run log in the 3D viewer
#   ./leap.sh analyze [log.npz]      estimated forces vs motor current for a run
#   ./leap.sh trial-setup            shadow runs (torque off) for A3, C5, R1, P0
#   ./leap.sh trial-plan             the counterbalanced schedule (and progress)
#   ./leap.sh trial [next|A3|C5|R1|P0]  one powered live trial + your observations
#   ./leap.sh compare                answer Q1-Q4 from the recorded trials
#   ./leap.sh trial-record NAME LOG  save observations for a run that was not recorded
#
# policy_dir defaults to PUT_YOUR_RL_FILES_HERE/. Extra flags for the hardware
# bridge go after "--", e.g.  ./leap.sh run -- --max-steps 200 --gain-profile H2
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

[[ -f "$KIT/.python" ]] || die "Run ./install.sh first."
PY="$(cat "$KIT/.python")"
export PYTHONPATH="$KIT/kit/_deps${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$KIT/configs/hand.env"

CMD="${1:-help}"; shift || true
POS=(); EXTRA=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then shift; EXTRA=("$@"); break; fi
  POS+=("$1"); shift
done

policy_dir() {
  local d="${POS[0]:-$KIT/PUT_YOUR_RL_FILES_HERE}"
  [[ -e "$d" ]] || die "Policy path $d does not exist."
  if [[ -d "$d" ]] && ! find -L "$d" -name '*.pt' | grep -q .; then
    die "No .pt checkpoint in $d - copy your model_XXXX.pt (and params/env.yaml, params/agent.yaml) there."
  fi
  echo "$d"
}

resolve_port() {
  if [[ "$LEAP_PORT" != "auto" ]]; then echo "$LEAP_PORT"; return; fi
  # The LEAP hand's U2D2 adapter is an FTDI chip; prefer it over other serial devices.
  local p n
  n="$(ls /dev/serial/by-id/*FTDI* 2>/dev/null | wc -l)"
  if [[ "$n" -gt 1 ]]; then
    die "Several FTDI adapters found - set LEAP_PORT in configs/hand.env (see ./leap.sh check -- --list)."
  fi
  p="$(ls /dev/serial/by-id/*FTDI* 2>/dev/null | head -n1 || true)"
  if [[ -z "$p" ]]; then
    n="$(ls /dev/ttyUSB* 2>/dev/null | wc -l)"
    [[ "$n" -eq 1 ]] && p="$(ls /dev/ttyUSB*)"
    [[ "$n" -gt 1 ]] && die "Several USB serial devices and none is FTDI - set LEAP_PORT in configs/hand.env."
  fi
  [[ -n "$p" ]] || die "No USB serial device found. Plug in the hand's USB adapter (and run ./leap.sh check)."
  echo "$p"
}

policy_hash() {
  "$PY" - "$1" <<'EOF'
import hashlib, sys
from pathlib import Path
p = Path(sys.argv[1])
files = [p] if p.is_file() else sorted(p.rglob("*.pt")) + sorted(p.rglob("*.yaml"))
h = hashlib.sha256()
for f in files:
    h.update(f.read_bytes())
print(h.hexdigest()[:16])
EOF
}

MONITOR_ARGS=()
if [[ -f "$KIT/monitors/cube_estimator.npz" && -f "$KIT/monitors/force_estimator.npz" ]]; then
  # cube-state + fingertip-force estimates are logged for every policy (never fed to it)
  MONITOR_ARGS=(--monitor-dir "$KIT/monitors")
fi
LIVE_ARGS=()

viewer_python() {
  if [[ "${LEAP_VIEWER_PYTHON:-auto}" != "auto" ]]; then echo "$LEAP_VIEWER_PYTHON"; return; fi
  if [[ -x "$KIT/../.venv/bin/python" ]] && "$KIT/../.venv/bin/python" -c "import mujoco" 2>/dev/null; then
    echo "$KIT/../.venv/bin/python"; return
  fi
  command -v uv >/dev/null || die "The viewer needs mujoco: run 'uv sync' at the repo root or set LEAP_VIEWER_PYTHON."
  echo "uv run --project $KIT/.. python"
}

newest_log() {
  ls -t "$KIT"/logs/physical_*.npz "$KIT"/logs/shadow_*.npz 2>/dev/null | grep -v _part | head -n1
}

bridge() {
  local port; port="$(resolve_port)"
  echo "[leap] port=$port baud=$LEAP_BAUD gains=$LEAP_GAIN_PROFILE"
  # FTDI latency timer: 16 ms (Linux default after every replug) adds ~16 ms to each
  # motor read and makes the 20 Hz loop jitter (18 Nov: 20% of steps > 60 ms).
  local lt; lt="/sys/bus/usb-serial/devices/$(basename "$(readlink -f "$port")")/latency_timer"
  if [[ -r "$lt" && "$(cat "$lt")" != "1" ]]; then
    printf '\033[1;33m[warn] USB latency timer is %s ms, not 1 ms: the control loop will jitter.\n' "$(cat "$lt")" >&2
    printf '       Fix: echo 1 | sudo tee %s   (or ./install.sh once)\033[0m\n' "$lt" >&2
  fi
  "$PY" "$KIT/kit/deploy_direct.py" "${MONITOR_ARGS[@]}" "${LIVE_ARGS[@]}" \
    --policy "$1" --server-endpoint local \
    --port "$port" --baudrate "$LEAP_BAUD" \
    --gain-profile "$LEAP_GAIN_PROFILE" \
    --current-limit-ma "$CURRENT_LIMIT_MA" --current-abort-ma "$CURRENT_ABORT_MA" \
    --temperature-abort-c "$TEMPERATURE_ABORT_C" \
    --velocity-abort-rad-s "$VELOCITY_ABORT_RAD_S" \
    --tracking-error-abort-rad "$TRACKING_ERROR_ABORT_RAD" \
    --max-command-step-rad "$MAX_COMMAND_STEP_RAD" \
    --log-chunk-steps 0 \
    "${@:2}"
}

if [[ "$CMD" == "live" ]]; then
  SUB="${POS[0]:-}"
  [[ "$SUB" == "shadow" || "$SUB" == "run" || "$SUB" == "sim" ]] \
    || die "usage: ./leap.sh live shadow|run|sim [policy_dir] [-- bridge flags]"
  [[ ${#MONITOR_ARGS[@]} -gt 0 ]] || die "monitors/ is missing cube_estimator.npz or force_estimator.npz"
  POS=("${POS[@]:1}")
  # check everything that can fail BEFORE opening the window
  policy_dir >/dev/null
  if [[ "$SUB" != "sim" ]]; then
    LIVE_PORT="$(resolve_port)"
    if [[ ! -r "$LIVE_PORT" || ! -w "$LIVE_PORT" ]]; then
      die "No permission to open $LIVE_PORT. Run: sudo chmod 666 $LIVE_PORT  (or ./install.sh once, then log out/in)"
    fi
  fi
  VPY="$(viewer_python)"
  echo "[leap] starting live viewer ($VPY) on udp $LEAP_LIVE_UDP"
  # shellcheck disable=SC2086
  $VPY "$KIT/kit/live_viewer.py" --udp "$LEAP_LIVE_UDP" --monitor-dir "$KIT/monitors" &
  VIEWER_PID=$!
  trap 'kill $VIEWER_PID 2>/dev/null || true' EXIT
  LIVE_ARGS=(--live-udp "$LEAP_LIVE_UDP")
  sleep 2
  CMD="$SUB"
fi

if [[ "$CMD" == "sim" ]]; then
  # Live viewer on the SIMULATED motor bus (kit/fake_dynamixel): no hand, no cube.
  P="$(policy_dir)"
  echo "[leap] SIMULATED motor bus: no hardware is used; the estimates are not meaningful without a real cube"
  PYTHONPATH="$KIT/kit/fake_dynamixel:$PYTHONPATH" "$PY" "$KIT/kit/deploy_direct.py" \
    "${MONITOR_ARGS[@]}" "${LIVE_ARGS[@]}" --policy "$P" --server-endpoint local --port SIMULATED \
    --gain-profile "$LEAP_GAIN_PROFILE" --no-shadow --max-steps 600 \
    --start-delay-s 0 --init-duration-s 1 --return-home-duration-s 0.5 \
    --current-limit-ma "$CURRENT_LIMIT_MA" --current-abort-ma "$CURRENT_ABORT_MA" \
    --max-command-step-rad "$MAX_COMMAND_STEP_RAD" --log-chunk-steps 0 \
    --log-file "$KIT/logs/sim_$(date +%Y%m%d_%H%M%S).npz" "${EXTRA[@]}"
  echo "[leap] simulated run finished; close the viewer window to exit"
  wait "$VIEWER_PID" 2>/dev/null || true
  exit 0
fi

case "$CMD" in
  inspect)
    "$PY" "$KIT/kit/inspect_policy.py" "$(policy_dir)" "${EXTRA[@]}"
    ;;
  test)
    # No hardware needed: policy check + full bridge run on a SIMULATED motor bus.
    P="${POS[0]:-$KIT/examples/reference_policy}"
    "$PY" "$KIT/kit/inspect_policy.py" "$P" --steps 200
    echo; echo "[leap] bridge dry-run on a simulated 16-motor bus (shadow, then actuated)"
    TMP="$(mktemp -d)"
    for mode in --shadow --no-shadow; do
      PYTHONPATH="$KIT/kit/fake_dynamixel:$PYTHONPATH" "$PY" "$KIT/kit/deploy_direct.py" \
        --policy "$P" --server-endpoint local --port SIMULATED \
        --gain-profile "$LEAP_GAIN_PROFILE" "$mode" --max-steps 60 \
        --start-delay-s 0 --init-duration-s 1 --return-home-duration-s 0.5 \
        --current-limit-ma "$CURRENT_LIMIT_MA" --current-abort-ma "$CURRENT_ABORT_MA" \
        --max-command-step-rad "$MAX_COMMAND_STEP_RAD" --log-chunk-steps 0 \
        --log-file "$TMP/sim${mode}.npz" | grep -E "policy|network|completed|saved|error|Error" || true
      [[ -f "$TMP/sim${mode}.npz" ]] || die "simulated bridge run ($mode) failed"
    done
    rm -rf "$TMP"
    echo "[leap] self-test passed"
    ;;
  check)
    PORT_ARG=()
    if [[ " ${EXTRA[*]:-} " != *" --list "* ]]; then PORT_ARG=(--port "$(resolve_port)"); fi
    "$PY" "$KIT/kit/check_hand.py" "${PORT_ARG[@]}" --baudrate "$LEAP_BAUD" "${EXTRA[@]}"
    ;;
  shadow)
    P="$(policy_dir)"
    "$PY" "$KIT/kit/inspect_policy.py" "$P" --steps 50 >/dev/null || die "Policy failed inspection: run ./leap.sh inspect"
    STAMP="$(date +%Y%m%d_%H%M%S)"
    bridge "$P" --shadow --max-steps "$SHADOW_STEPS" \
      --log-file "$KIT/logs/shadow_${STAMP}.npz" "${EXTRA[@]}"
    LOG="$KIT/logs/shadow_${STAMP}.npz"
    "$PY" "$KIT/kit/report.py" "$LOG" || true
    if "$PY" -c 'import sys, numpy as np; sys.exit(0 if bool(np.load(sys.argv[1])["completed"]) else 1)' "$LOG"; then
      date > "$KIT/logs/.shadow_passed_$(policy_hash "$P")"
    else
      die "Shadow run did not complete - it does not unlock ./leap.sh run."
    fi
    echo
    echo "[leap] Shadow run finished. Check that cmd ranges look sane and the q columns"
    echo "[leap] match how the hand actually sits. Then: ./leap.sh run"
    ;;
  run)
    P="$(policy_dir)"
    H="$(policy_hash "$P")"
    if [[ ! -f "$KIT/logs/.shadow_passed_$H" && "${LEAP_SKIP_SHADOW_CHECK:-0}" != "1" ]]; then
      die "This exact policy has not passed a shadow run yet. Run ./leap.sh shadow first."
    fi
    cat <<EOF

  *** REAL MOTION ***  policy: $P
  - clear the workspace, keep hands and cables away from the fingers
  - keep the hand's power cutoff within reach (Ctrl+C returns the hand home)
  - $RUN_STEPS steps ($((RUN_STEPS / 20)) s at 20 Hz; 0 = until Ctrl+C); place the object during the ${START_DELAY_S}s countdown

EOF
    read -r -p "Type YES to start: " ans
    [[ "$ans" == "YES" ]] || die "Aborted."
    STAMP="$(date +%Y%m%d_%H%M%S)"
    bridge "$P" --no-shadow --max-steps "$RUN_STEPS" \
      --start-delay-s "$START_DELAY_S" --init-duration-s "$INIT_DURATION_S" \
      --return-home-duration-s "$RETURN_HOME_S" \
      --log-file "$KIT/logs/physical_${STAMP}.npz" "${EXTRA[@]}"
    "$PY" "$KIT/kit/report.py" "$KIT/logs/physical_${STAMP}.npz" || true
    ;;
  report)
    "$PY" "$KIT/kit/report.py" "${POS[@]}"
    ;;
  view)
    LOG="${POS[0]:-$(newest_log)}"
    [[ -n "$LOG" && -f "$LOG" ]] || die "no log to view"
    VPY="$(viewer_python)"
    # shellcheck disable=SC2086
    $VPY "$KIT/kit/live_viewer.py" --replay "$LOG" --monitor-dir "$KIT/monitors" "${EXTRA[@]}"
    ;;
  analyze)
    [[ ${#POS[@]} -gt 0 ]] || POS=("$(newest_log)")
    VPY="$(viewer_python)"
    # matplotlib (for the plot) is optional; uv can add it on the fly
    # shellcheck disable=SC2086
    if ! $VPY -c "import matplotlib" 2>/dev/null && command -v uv >/dev/null; then
      VPY="uv run --project $KIT/.. --with matplotlib python"
    fi
    # shellcheck disable=SC2086
    $VPY "$KIT/kit/analyze_run.py" "${POS[@]}" --monitor-dir "$KIT/monitors" "${EXTRA[@]}"
    ;;
  trial-setup)
    for n in A3 C5 R1 P0; do
      echo; echo "[leap] ===== shadow run for $n (torque off) ====="
      "$KIT/leap.sh" shadow "$KIT/policies/$n" -- "${EXTRA[@]}"
    done
    echo; echo "[leap] all four policies unlocked. Next: ./leap.sh trial next"
    ;;
  trial-plan)
    "$PY" "$KIT/kit/trials.py" plan --blocks "$TRIAL_BLOCKS"
    ;;
  trial)
    NAME="${POS[0]:-next}"
    if [[ "$NAME" == "next" ]]; then NAME="$("$PY" "$KIT/kit/trials.py" next --blocks "$TRIAL_BLOCKS")"; fi
    if [[ "$NAME" == "DONE" ]]; then echo "[leap] all scheduled trials are done. Run: ./leap.sh compare"; exit 0; fi
    [[ "$NAME" =~ ^(A3|C5|R1|P0|M400)$ ]] || die "unknown trial policy '$NAME' (A3, C5, R1, P0, M400)"
    echo; echo "[leap] ===== TRIAL: $NAME ($((TRIAL_STEPS / 20)) s) ====="
    echo "[leap] place the cube the same way every time; count the turns you see"
    MARK="$(mktemp)"
    RUN_STEPS="$TRIAL_STEPS" "$KIT/leap.sh" live run "$KIT/policies/$NAME" -- "${EXTRA[@]}" || true
    LOG="$(find "$KIT/logs" -maxdepth 1 -name 'physical_*.npz' -newer "$MARK" | grep -v _part | sort | tail -n1 || true)"
    rm -f "$MARK"
    [[ -n "$LOG" ]] || die "no new run log (run not started?) - nothing recorded; repeat: ./leap.sh trial $NAME"
    "$PY" "$KIT/kit/trials.py" record --policy "$NAME" --log "$LOG"
    echo "[leap] next scheduled: $("$PY" "$KIT/kit/trials.py" next --blocks "$TRIAL_BLOCKS")   (./leap.sh trial next)"
    ;;
  compare)
    "$PY" "$KIT/kit/trials.py" compare
    ;;
  trial-record)
    # record observations for a run that already happened: ./leap.sh trial-record C5 logs/physical_X.npz
    [[ ${#POS[@]} -eq 2 ]] || die "usage: ./leap.sh trial-record A3|C5|R1|P0 logs/physical_X.npz"
    "$PY" "$KIT/kit/trials.py" record --policy "${POS[0]}" --log "${POS[1]}"
    echo "[leap] next scheduled: $("$PY" "$KIT/kit/trials.py" next --blocks "$TRIAL_BLOCKS")   (./leap.sh trial next)"
    ;;
  functest)
    # Basic hardware function test, no policy. Stage 1 is read-only; later stages MOVE the hand.
    if [[ "${POS[0]:-}" == "sim" ]]; then
      echo "[leap] function test on the SIMULATED motor bus (no hardware)"
      PYTHONPATH="$KIT/kit/fake_dynamixel:$PYTHONPATH" "$PY" "$KIT/kit/function_test.py" \
        --port SIMULATED --no-confirm --ramp-s 1 --hold-s 1 --track-s 4 \
        --log-file "$KIT/logs/functest_sim.npz" "${EXTRA[@]}"
      exit $?
    fi
    cat <<EOF

  *** HARDWARE FUNCTION TEST ***  stage 1 is read-only; stages 2-7 MOVE the hand
  - take the cube out, clear the workspace, keep the 12 V cutoff within reach
  - Ctrl+C at any time returns the hand to flat and turns torque off

EOF
    read -r -p "Type YES to start: " ans
    [[ "$ans" == "YES" ]] || die "Aborted."
    "$PY" "$KIT/kit/function_test.py" --port "$(resolve_port)" --baudrate "$LEAP_BAUD" \
      --gain-profile "$LEAP_GAIN_PROFILE" \
      --current-limit-ma "$CURRENT_LIMIT_MA" --current-abort-ma "$CURRENT_ABORT_MA" \
      --temperature-abort-c "$TEMPERATURE_ABORT_C" --velocity-abort-rad-s "$VELOCITY_ABORT_RAD_S" \
      --tracking-error-abort-rad "$TRACKING_ERROR_ABORT_RAD" \
      --max-command-step-rad "$MAX_COMMAND_STEP_RAD" "${EXTRA[@]}"
    ;;
  *)
    sed -n '2,/^set -euo/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'
    ;;
esac
