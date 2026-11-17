#!/usr/bin/env bash
# One-time setup of the LEAP Hand RL Deploy Kit on a fresh Linux computer.
#
#   ./install.sh                 use the python3 on PATH (activate your RL env first)
#   LEAP_PYTHON=/path/python ./install.sh   use a specific interpreter
#   ./install.sh --no-sudo       skip serial-port permission setup
#
# What it does:
#   1. picks a Python >= 3.10 that has numpy (creates .venv with numpy if not)
#   2. unpacks the bundled dynamixel_sdk + pyserial (no internet needed)
#   3. gives your user access to USB serial ports (dialout group) and sets the
#      FTDI latency timer to 1 ms (needed for a smooth 20 Hz control loop)
#   4. runs an offline self-test with the bundled reference policy
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NO_SUDO=0
[[ "${1:-}" == "--no-sudo" ]] && NO_SUDO=1

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m[error] %s\033[0m\n' "$*"; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "This kit targets Linux (Ubuntu 20.04+ recommended)."

# ---------------------------------------------------------------- 1. Python
say "Selecting Python"
PY="${LEAP_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
  done
fi
[[ -n "$PY" ]] || die "No python3 found. Install it: sudo apt install python3 python3-venv"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || die "$PY is $("$PY" -V 2>&1); Python >= 3.10 is required. Set LEAP_PYTHON=/path/to/python3.10+"
echo "Python: $PY ($("$PY" -V 2>&1))"

if ! "$PY" -c 'import numpy' >/dev/null 2>&1; then
  warn "numpy is not available in $PY - creating a private environment in $KIT/.venv"
  if ! "$PY" -m venv "$KIT/.venv" >/dev/null 2>&1; then
    if [[ $NO_SUDO -eq 0 ]] && command -v apt-get >/dev/null; then
      PYVER="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
      sudo apt-get update && sudo apt-get install -y "python${PYVER}-venv" || sudo apt-get install -y python3-venv
      "$PY" -m venv "$KIT/.venv"
    else
      die "python venv module missing: sudo apt install python3-venv"
    fi
  fi
  "$KIT/.venv/bin/python" -m pip install --upgrade pip >/dev/null
  "$KIT/.venv/bin/python" -m pip install "numpy>=1.22" || die "Could not install numpy (internet needed for this step)."
  PY="$KIT/.venv/bin/python"
fi
"$PY" -c 'import numpy; print("numpy", numpy.__version__)'
echo "$PY" > "$KIT/.python"

# ---------------------------------------------------- 2. bundled hardware deps
say "Unpacking bundled dynamixel_sdk + pyserial into kit/_deps"
rm -rf "$KIT/kit/_deps"
"$PY" - "$KIT" <<'EOF'
import sys, zipfile, glob, os
kit = sys.argv[1]
dest = os.path.join(kit, "kit", "_deps")
os.makedirs(dest, exist_ok=True)
for whl in sorted(glob.glob(os.path.join(kit, "vendor", "wheels", "*.whl"))):
    with zipfile.ZipFile(whl) as z:
        z.extractall(dest)
    print("  ", os.path.basename(whl))
EOF
PYTHONPATH="$KIT/kit/_deps" "$PY" -c 'import dynamixel_sdk, serial; print("dynamixel_sdk OK, pyserial", serial.__version__)'

# ------------------------------------------------- 3. serial-port permissions
say "Serial port access"
if [[ $NO_SUDO -eq 1 ]]; then
  warn "--no-sudo given: skipping. You need read/write access to /dev/ttyUSB* yourself."
else
  if id -nG "$USER" | tr ' ' '\n' | grep -qx dialout; then
    echo "user $USER is already in group 'dialout'"
  else
    echo "adding $USER to group 'dialout' (sudo password may be asked)"
    sudo usermod -aG dialout "$USER" && NEED_RELOGIN=1
  fi
  RULE=/etc/udev/rules.d/99-leap-hand-ftdi.rules
  if [[ ! -f $RULE ]]; then
    echo "installing $RULE (FTDI latency_timer=1 ms, world-readable tty)"
    sudo tee "$RULE" >/dev/null <<'EOF'
# LEAP hand U2D2 / FTDI USB-serial: 1 ms latency for fast Dynamixel sync reads
ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", MODE="0666"
EOF
    sudo udevadm control --reload-rules && sudo udevadm trigger || true
  fi
  for f in /sys/bus/usb-serial/devices/ttyUSB*/latency_timer; do
    [[ -e $f ]] && echo 1 | sudo tee "$f" >/dev/null && echo "set $(dirname "$f" | xargs basename) latency_timer=1"
  done
  # brltty grabs FTDI/CH340 adapters on some Ubuntu releases
  if systemctl list-unit-files 2>/dev/null | grep -q '^brltty'; then
    warn "brltty is installed; it can steal USB-serial adapters. If /dev/ttyUSB0 never appears run:"
    warn "    sudo apt remove brltty"
  fi
fi

chmod +x "$KIT/leap.sh" "$KIT"/kit/*.py

# ------------------------------------------------------------- 4. self-test
say "Offline self-test with the bundled reference policy"
PYTHONPATH="$KIT/kit/_deps" "$PY" "$KIT/kit/inspect_policy.py" "$KIT/examples/reference_policy" --steps 200 | tail -n 4

say "Done"
if [[ "${NEED_RELOGIN:-0}" == "1" ]]; then
  warn "You were added to 'dialout'. LOG OUT AND BACK IN (or reboot) before talking to the hand."
fi
cat <<EOF

Next steps:
  1. copy your RL files into  $KIT/PUT_YOUR_RL_FILES_HERE/
  2. ./leap.sh inspect      (checks your policy, no hardware)
  3. ./leap.sh check        (reads all 16 motors, torque stays off)
  4. ./leap.sh shadow       (runs the policy on live encoders, torque stays off)
  5. ./leap.sh run          (real motion - supervised, power cutoff within reach)
EOF
