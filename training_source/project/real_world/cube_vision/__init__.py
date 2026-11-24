"""Read-only vision tracking + digital-twin dashboard for the MBZUAI six-face ArUco cube.

Nothing in this package talks to the motors: it reads camera frames, depth frames,
recordings and the kit's one-way UDP / .npz telemetry, and writes JSONL/CSV/video
files.  See docs/live_vision_digital_twin.md.
"""
