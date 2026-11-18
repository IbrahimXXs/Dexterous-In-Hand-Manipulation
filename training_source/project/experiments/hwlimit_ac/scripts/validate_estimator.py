#!/usr/bin/env python3
"""Phase 2: is the frozen estimator B still adequate under the approved joint limits?

Trajectories: the six selected A and C policies of the completed study
(read-only copies in ``experiments/scratch_state_estimation/checkpoints/``),
each rolled out (deterministic, 30 s, 512 environments, nominal preset,
simulation seed 8101, used nowhere else) in

* the original study environment (``Mjlab-Leap-Scratch-<c>``: B's training
  distribution, reference), and
* the hardware-limit environment (``Mjlab-Leap-HwLim-<c>``: approved command
  limits, new reset cache).

B is scored closed-loop on the states each policy visits (simulator ground
truth used only as labels): position, orientation (symmetry-aware), linear
and angular velocity, fingertip-contact classification, the error of the 19
actor features C consumes, and the input shift relative to B's normaliser.

Decision rule (fixed before running; README Section 3):
  B is reused unchanged if, averaged over the six policies, each of
    position RMSE, symmetric rotation error (mean), linear-velocity RMSE,
    angular-velocity RMSE, and the RMSE of the 19 actor features
  is at most 1.25x its value in the original environment, and the mean
  fingertip-contact F1 (4 fingers) drops by at most 0.05.
  Otherwise B is reported as degraded and training of C is not started.

    python experiments/hwlimit_ac/scripts/validate_estimator.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

E = Path("experiments/hwlimit_ac")
S = Path("experiments/scratch_state_estimation")
POLICIES = {"A_seed1": "A_seed1_model_1000.pt", "A_seed2": "A_seed2_model_1000.pt", "A_seed3": "A_seed3_model_900.pt",
            "C_seed1": "C_seed1_model_1000.pt", "C_seed2": "C_seed2_model_1000.pt", "C_seed3": "C_seed3_model_1000.pt"}
RATIO_MAX, F1_DROP_MAX = 1.25, 0.05
KEYS = ["pos_rmse_mm", "rot_err_sym_mean_deg", "lin_vel_rmse_mps", "ang_vel_rmse_radps", "feature_rmse_all"]


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--num-envs", type=int, default=512)
  p.add_argument("--seed", type=int, default=8101)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out", default=str(E / "results" / "estimator_validation.json"))
  a = p.parse_args()

  import torch

  from in_hand_rotation_mjlab.estimation.model import load_estimator
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import EvalConfig, evaluate, summarize

  est = load_estimator(str(S / "models" / "estimator_B.pt"), map_location=a.device).to(a.device).eval()
  out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
  out.setdefault("runs", {})
  cfg = EvalConfig(num_envs=a.num_envs, num_steps=600, device=a.device)
  for name, ck in POLICIES.items():
    c = name[0]
    for envname, task in (("original", f"Mjlab-Leap-Scratch-{c}"), ("hwlimit", f"Mjlab-Leap-HwLim-{c}")):
      key = f"{name}/{envname}"
      if key in out["runs"]:
        continue
      t0 = time.time()
      pe, extra = evaluate(task, str(S / "checkpoints" / ck), a.seed, "nominal", cfg, estimator=est)
      sm = summarize(pe)
      el = extra["estimator_closed_loop"]
      fr = np.array(list(extra["feature_rmse"].values()))
      el["feature_rmse_all"] = float(np.sqrt((fr ** 2).mean()))
      el["contact_f1_mean"] = float(np.mean([el[f"contact_{f}_f1"] for f in ("if", "mf", "rf", "th")]))
      out["runs"][key] = {"task": task, "estimator": el, "feature_rmse": extra["feature_rmse"], "input_shift": extra["input_shift"],
                          "policy": {k: sm[k] for k in ("turns", "survival_s", "failure", "cmd_at_limit_frac", "any_joint_at_limit_frac",
                                                        "kit_clip_frac", "state_outside_frac", "max_cmd_step_rad")},
                          "seconds": round(time.time() - t0, 1)}
      print(f"[{key}] pos {el['pos_rmse_mm']:.2f} mm rot {el['rot_err_sym_mean_deg']:.2f} deg lin {el['lin_vel_rmse_mps']:.4f} "
            f"ang {el['ang_vel_rmse_radps']:.3f} featRMSE {el['feature_rmse_all']:.3f} F1 {el['contact_f1_mean']:.3f} "
            f"|z| {extra['input_shift']['mean_abs_z']:.3f} turns {sm['turns']['mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)
      Path(a.out).write_text(json.dumps(out, indent=2))

  agg = {}
  for envname in ("original", "hwlimit"):
    rows = [out["runs"][f"{n}/{envname}"] for n in POLICIES]
    agg[envname] = {k: float(np.mean([r["estimator"][k] for r in rows])) for k in KEYS + ["contact_f1_mean"]}
    agg[envname]["input_mean_abs_z"] = float(np.mean([r["input_shift"]["mean_abs_z"] for r in rows]))
    agg[envname]["input_frac_abs_z_gt4"] = float(np.mean([r["input_shift"]["frac_abs_z_gt4"] for r in rows]))
  ratios = {k: agg["hwlimit"][k] / agg["original"][k] for k in KEYS}
  f1_drop = agg["original"]["contact_f1_mean"] - agg["hwlimit"]["contact_f1_mean"]
  passed = all(r <= RATIO_MAX for r in ratios.values()) and f1_drop <= F1_DROP_MAX
  out["aggregate"] = agg
  out["decision"] = {"ratios_hwlimit_over_original": ratios, "contact_f1_drop": f1_drop,
                     "rule": f"all ratios <= {RATIO_MAX} and contact F1 drop <= {F1_DROP_MAX}",
                     "estimator_B_reusable": bool(passed)}
  Path(a.out).write_text(json.dumps(out, indent=2))
  print(json.dumps(out["decision"], indent=2))


if __name__ == "__main__":
  main()
