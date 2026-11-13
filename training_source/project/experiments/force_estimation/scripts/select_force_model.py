#!/usr/bin/env python3
"""Pick the final model among training runs by VALIDATION MAE only; copy it to models/ with sha256."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--runs", default="experiments/force_estimation/runs")
  p.add_argument("--out", default="experiments/force_estimation/models/force_estimator.pt")
  a = p.parse_args()
  rows = []
  for s in sorted(Path(a.runs).glob("*/summary.json")):
    r = json.loads(s.read_text())
    rows.append((r["best_val_mae_N"], s.parent.name, r))
  if not rows:
    raise SystemExit("no finished runs")
  rows.sort()
  best = rows[0]
  out = Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(Path(a.runs) / best[1] / "best.pt", out)
  h = hashlib.sha256(out.read_bytes()).hexdigest()
  (out.with_suffix(".sha256")).write_text(f"{h}  {out.name}\n")
  sel = {"selected_run": best[1], "rule": "lowest validation MAE (N, mean over fingers)",
         "candidates": [{"run": n, "val_mae_N": v, "head": r["head"], "seed": r["seed"], "best_step": r["best_step"]} for v, n, r in rows],
         "sha256": h}
  (out.parent / "selection.json").write_text(json.dumps(sel, indent=2))
  print(json.dumps(sel, indent=2))


if __name__ == "__main__":
  main()
