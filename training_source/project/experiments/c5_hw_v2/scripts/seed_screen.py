#!/usr/bin/env python3
"""C5-HW-V2: the 350 mA jam screen (c5_hw_v1/scripts/seed_screen.py procedure, evaluation only, no training).

Frozen checkpoints, grasp 49, seed 8301, 256 environments, 30 s, with the simulated effort limit capped at the
deployed 350 mA (0.167-0.204 Nm) and with the training default.  Recorder and analysis are the audit's
(c5_hw_v1 sim_actuator_reference.record + sim_effort_diagnostic.analyse, imported unchanged).  Screens C5HW2_1 and
C5-EFFORT (EFFORT_1); the C5 seeds 1-3 at the same seed are read from c5_hw_v1/results/c5_seed_screen.json
(same procedure and seed), not re-run.  Writes data/seed_screen/*.npz (git-ignored) and results/seed_screen.json.

    python experiments/c5_hw_v2/scripts/seed_screen.py
"""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]
V1 = E.parent / "c5_hw_v1"


def _load(name: str):
  spec = importlib.util.spec_from_file_location(name, V1 / "scripts" / f"{name}.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


REF = _load("sim_actuator_reference")
DIAG = _load("sim_effort_diagnostic")
OUT = E / "data" / "seed_screen"
CAP350 = (0.167, 0.204)  # as c5_hw_v1/scripts/seed_screen.py


def _ck(pattern: str) -> str:
  c = sorted(glob.glob(str(pattern)))
  assert len(c) == 1, c
  return str(Path(c[0]).relative_to(E.parents[1]))


def main() -> None:
  new = {"EFFORT_1": _ck(V1 / "checkpoints" / "EFFORT_1_selected_model_*.pt"),
         "C5HW2_1": _ck(E / "checkpoints" / "C5HW2_1_selected_model_*.pt")}
  for k, rel in new.items():
    REF.POLICIES[k] = (rel, hashlib.sha256((E.parents[1] / rel).read_bytes()).hexdigest()[:16])
  OUT.mkdir(parents=True, exist_ok=True)
  res = {"procedure": "c5_hw_v1/scripts/seed_screen.py (grasp 49, seed 8301, 256 envs, 30 s)", "checkpoints": new, "rows": {}}
  for pol in new:
    for tag, el in (("_cap350", CAP350), ("", None)):
      f = OUT / f"sim_{pol}{tag}_grasp49_seed8301_n256.npz"
      if not f.exists():
        np.savez_compressed(f, **REF.record(pol, 8301, 256, 600, "cuda:0", el))
      r = DIAG.analyse(f)
      res["rows"][f.stem] = r
      print(f"{f.stem:38s} turns30 {r['turns30_mean']:.3f} survived {r['survived_frac']:.3f} "
            f"index-sat {r['sat_frac_index_mcp_pip']:.4f} envs index>=1s {r['envs_index_sustained_ge_1s']:.3f} "
            f"most {r['joint_most_often_sustained']}", flush=True)
  res["c5_seeds_from_c5_hw_v1"] = json.loads((V1 / "results" / "c5_seed_screen.json").read_text())
  (E / "results" / "seed_screen.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
  main()
