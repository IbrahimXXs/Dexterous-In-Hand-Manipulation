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
#
# policy_dir defaults to PUT_YOUR_RL_FILES_HERE/. Extra flags for the hardware
# bridge go after "--", e.g.  ./leap.sh run -- --max-steps 200 --gain-profile H2
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { printf '\033[1;31m[error] %s\033[0m\n' "$*"; exit 1; }

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
  if [[ -d "$d" ]] && ! find "$d" -name '*.pt' | grep -q .; then
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

bridge() {
  local port; port="$(resolve_port)"
  echo "[leap] port=$port baud=$LEAP_BAUD gains=$LEAP_GAIN_PROFILE"
  "$PY" "$KIT/kit/deploy_direct.py" \
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
  - $RUN_STEPS steps; place the object during the ${START_DELAY_S}s countdown

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
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
