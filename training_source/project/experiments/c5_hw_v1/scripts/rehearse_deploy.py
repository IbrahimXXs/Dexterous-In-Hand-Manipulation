#!/usr/bin/env python3
"""Simulation-only closed-loop rehearsal of the deploy kit runtime with the C5-HW-EFFORT-V1
candidate (no hardware); a copy of grasp_rotation_v2/scripts/rehearse_deploy.py run in the candidate's own
task Mjlab-Leap-C5HW-EFFORT (350 mA-equivalent effort limits).  Reuses ``experiments/hwlimit_ac/scripts/rehearse_deploy_hwlim.py``
(``run_kit``, the hardware session's ``KitRunner``) unchanged; environments
Same actor interface and limits as HwLim-A / Retention-V1 / GRV2.
Modes: kit reset "measured" (as on the hand), "target" (as in training), and the
diagnostic "target + training limits in the kit runtime" (in memory only; PIP gap).

    python experiments/c5_hw_v1/scripts/rehearse_deploy.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "c5_hw_v1"
_spec = importlib.util.spec_from_file_location("hw_rehearse", PROJECT / "experiments" / "hwlimit_ac" / "scripts" / "rehearse_deploy_hwlim.py")
RH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RH)
H = RH.H


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--seed", type=int, default=5001)
  p.add_argument("--preset", default="nominal")
  p.add_argument("--num-envs", type=int, default=128)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  out = E / "results" / "deploy_rehearsal.json"
  cfg = H.EvalConfig(num_envs=a.num_envs, num_steps=600, device=a.device)
  allres = json.loads(out.read_text()) if out.exists() else {}
  for d in sorted((E / "deploy" / "policies").iterdir()):
    key = f"{d.name}/{a.preset}/seed{a.seed}/n{a.num_envs}"
    if key in allres:
      continue
    ckpt = next(d.glob("model_*.pt"))
    TASK = "Mjlab-Leap-C5HW-EFFORT"
    ref, _ = H.evaluate(TASK, str(ckpt), a.seed, a.preset, cfg)
    res = {"checkpoint": ckpt.name, "reference_training_actor": {k: float(np.mean(ref[k])) for k in RH.KEYS}}
    for name, mode, tl in (("kit_reset_measured", "measured", False), ("kit_reset_target", "target", False),
                           ("diagnostic_kit_reset_target_training_limits", "target", True)):
      pe, diag = RH.run_kit(d, ckpt, TASK, a.seed, a.preset, cfg, mode, 0.01, training_limits=tl)
      dt = pe["turns"] - ref["turns"]
      res[name] = {**{k: float(np.mean(pe[k])) for k in RH.KEYS}, "turns_minus_reference": float(dt.mean()),
                   "turns_diff_ci95": float(1.96 * dt.std(ddof=1) / np.sqrt(len(dt))), **diag}
      print(f"[{key} {name}] ref {np.mean(ref['turns']):.3f} kit {np.mean(pe['turns']):.3f} ({dt.mean():+.3f}±{res[name]['turns_diff_ci95']:.3f}) "
            f"clip {diag['bridge_clip_fraction_of_steps']:.4f} obs err {diag['obs_max_abs_err_after_10_steps']} "
            f"max step {res[name]['max_cmd_step_rad']:.4f}", flush=True)
    allres[key] = res
    out.write_text(json.dumps(allres, indent=1) + "\n")


if __name__ == "__main__":
  main()
