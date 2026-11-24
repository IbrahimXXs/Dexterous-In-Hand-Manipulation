#!/usr/bin/env python3
"""Screen the three C5 seeds for the 20 Nov hardware failure signature (evaluation only, no training).

Same procedure as the audit (sim_actuator_reference.py + sim_effort_diagnostic.py, reused unchanged):
frozen checkpoints, grasp 49, seed 8301, 256 environments, 30 s, with the simulated effort limit
capped at the deployed 350 mA (0.167-0.204 Nm) and with the training default.  Adds the C5 seeds 2
and 3 (same recipe as C5 seed 1) and writes to seed_screen/ and results/c5_seed_screen.json, so the
audit's own outputs are not touched.

    python experiments/c5_hw_v1/scripts/seed_screen.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]


def _load(name: str):
  spec = importlib.util.spec_from_file_location(name, E / "scripts" / f"{name}.py")
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


REF = _load("sim_actuator_reference")
DIAG = _load("sim_effort_diagnostic")
REF.POLICIES.update({
  "C5s2": ("experiments/grasp_rotation_v2/checkpoints/C5_2_selected_model_1500.pt", "49579bdf1f598d8c"),
  "C5s3": ("experiments/grasp_rotation_v2/checkpoints/C5_3_selected_model_3000.pt", "c221af89199cdeab"),
})
OUT = E / "seed_screen"
CAP350 = (0.167, 0.204)


def main() -> None:
  OUT.mkdir(exist_ok=True)
  res = {}
  for pol in ("C5", "C5s2", "C5s3"):
    for tag, el in (("_cap350", CAP350), ("", None)):
      f = OUT / f"sim_{pol}{tag}_grasp49_seed8301_n256.npz"
      if not f.exists():
        np.savez_compressed(f, **REF.record(pol, 8301, 256, 600, "cuda:0", el))
      r = DIAG.analyse(f)
      res[f.stem] = r
      print(f"{f.stem:34s} turns30 {r['turns30_mean']:.3f} survived {r['survived_frac']:.3f} "
            f"index-sat {r['sat_frac_index_mcp_pip']:.4f} index-blocked {r['blocked_frac_index_mcp_pip']:.4f} "
            f"envs index>=1s {r['envs_index_sustained_ge_1s']:.3f} most {r['joint_most_often_sustained']}", flush=True)
  (E / "results" / "c5_seed_screen.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
  main()
