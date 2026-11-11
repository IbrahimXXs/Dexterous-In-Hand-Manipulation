#!/usr/bin/env bash
# Install the lightweight continuous-yaw environment (viewer, policy, hardware bridge) and run its self-tests.
source "$(dirname "$0")/_common.sh"
banner "Continuous-yaw setup  ->  $ROOT"
cd "$ROOT"
uv sync
uv run pytest
echo "✅ Ready. Next: scripts/replay.sh"
