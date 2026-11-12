#!/usr/bin/env python3
"""Export an estimator-augmented checkpoint to a deployable TorchScript bundle.

    python scripts/estimation/export_estimator_policy.py \
        --checkpoint logs/rsl_rl/leap_right_mbzuai_rotate_estimator_full/<run>/model_700.pt \
        --output ../../checkpoint/estimator/estimator_policy_full.pt

For the unchanged model_400 actor, ``--estimator`` attaches the estimator as a
state monitor only (actions identical to model_400).
"""

import argparse
import json

from in_hand_rotation_mjlab.estimation.deploy import export_bundle


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--output", required=True)
  p.add_argument("--estimator", default=None)
  a = p.parse_args()
  meta = export_bundle(a.checkpoint, a.output, a.estimator)
  print(json.dumps(meta, indent=2))
  print(f"[export] wrote {a.output}")


if __name__ == "__main__":
  main()
