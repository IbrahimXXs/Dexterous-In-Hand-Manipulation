# Shared helpers for the continuous-yaw launchers in scripts/ (sourced, not executed).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING="$ROOT/training_source/project"

if ! command -v uv >/dev/null 2>&1; then
  echo "❌ 'uv' is not installed. macOS: brew install uv   Linux: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

# MuJoCo's interactive viewer must run on the main thread on macOS -> use mjpython there.
if [ "$(uname -s)" = "Darwin" ]; then VIEWER_PY="mjpython"; else VIEWER_PY="python"; fi

banner() { printf '\n==================================================================\n%s\n==================================================================\n' "$*"; }
