#!/usr/bin/env python3
"""Build an augmented-actor checkpoint from model_400.pt + a trained estimator.

The result is a regular RSL-RL checkpoint (actor, critic, migrated Adam state,
iteration) that the *unmodified* ``scripts/train.py`` resumes with
``--checkpoint-file``.  See ``estimation/migration.py`` for the tensor map.

    python scripts/estimation/migrate_checkpoint.py \
        --estimator data/estimator_models/cube_state_estimator.pt \
        --feature-mode full --output data/estimator_models/model_400_estimator_full.pt
"""

import argparse
from pathlib import Path

from in_hand_rotation_mjlab.estimation.migration import migrate_checkpoint

ROOT = Path(__file__).resolve().parents[4]


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--source", default=str(ROOT / "checkpoint" / "model_400.pt"))
  p.add_argument("--estimator", required=True)
  p.add_argument("--feature-mode", choices=["full", "contact"], default="full")
  p.add_argument("--output", required=True)
  a = p.parse_args()
  if Path(a.output).resolve() == Path(a.source).resolve():
    raise SystemExit("Refusing to overwrite the source checkpoint.")
  out = migrate_checkpoint(a.source, a.estimator, a.feature_mode, a.output)
  for line in out["estimation"]["migration_report"]:
    print("  " + line)
  print(f"[migrate] wrote {a.output} (iteration {out['iter']})")


if __name__ == "__main__":
  main()
