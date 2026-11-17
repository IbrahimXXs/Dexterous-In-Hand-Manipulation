#!/usr/bin/env python3
"""Export from-scratch study checkpoints (A / C) to robot-side bundles.

Needs only torch (no MJLab / RSL-RL).  By default exports every selected A and
C checkpoint in experiments/scratch_state_estimation/checkpoints/ to
<repo>/checkpoint/scratch_study/, with estimator B and the selected fingertip
force estimator attached:

    python scripts/estimation/export_scratch_policy.py
    python scripts/estimation/export_scratch_policy.py --checkpoint .../C_seed2_model_1000.pt --output out.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from in_hand_rotation_mjlab.scratch_study.deploy import export_bundle  # noqa: E402

STUDY = PROJECT / "experiments" / "scratch_state_estimation"
DEFAULT_ESTIMATOR = STUDY / "models" / "estimator_B.pt"
DEFAULT_FORCE = PROJECT / "experiments" / "force_estimation" / "models" / "force_estimator.pt"


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", type=Path, default=None, help="one study checkpoint (default: all A and C)")
  p.add_argument("--output", type=Path, default=None)
  p.add_argument("--out-dir", type=Path, default=REPO / "checkpoint" / "scratch_study")
  p.add_argument("--estimator", type=Path, default=DEFAULT_ESTIMATOR)
  p.add_argument("--force-estimator", type=Path, default=DEFAULT_FORCE)
  p.add_argument("--no-force-estimator", action="store_true")
  a = p.parse_args()
  force = None if a.no_force_estimator else a.force_estimator

  if a.checkpoint is not None:
    jobs = [(a.checkpoint, a.output or a.out_dir / f"{a.checkpoint.stem}.pt")]
  else:
    ckpts = sorted((STUDY / "checkpoints").glob("[AC]_seed*_model_*.pt"))
    jobs = [(c, a.out_dir / f"{c.name.split('_model_')[0]}.pt") for c in ckpts]
  manifest = {}
  for ckpt, out in jobs:
    meta = export_bundle(ckpt, out, a.estimator, force, label=ckpt.stem)
    manifest[out.name] = meta
    print(f"[export] {ckpt.name} -> {out}  (condition {meta['condition']}, {meta['estimator_role']})")
  if a.checkpoint is None:
    (a.out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
  main()
