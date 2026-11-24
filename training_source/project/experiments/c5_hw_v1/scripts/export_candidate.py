#!/usr/bin/env python3
"""C5-HW-EFFORT-V1: NumPy export of the deployment candidate + numerical-equivalence check (offline only).

Runs only after build_deploy.py (i.e. only after a single-seed pass).  Uses the kit's own, unchanged
``policy_runtime.export_npz`` to write deploy/exports/C5HW_EFFORT_V1_seed1_actor.npz (outside the policy folder,
which stays exactly as built), then checks:
  * every exported weight / bias / normaliser array equals the checkpoint tensor bit-for-bit;
  * a NumPy forward pass rebuilt from the .npz alone equals the kit runtime actor on 4096 observations
    drawn around the training normaliser (mean + std * N(0, 1));
  * the spec stored in the export equals the kit's describe() of the policy folder.
Writes results/export_verification.json and adds the export hash to deploy/MANIFEST.txt.

    python experiments/c5_hw_v1/scripts/export_candidate.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "c5_hw_v1"
KIT = PROJECT.parents[1] / "LEAP_Hand_RL_Deploy_Kit" / "kit"
sys.path.insert(0, str(KIT))
from policy_runtime import RSL_RL_NORMALIZER_EPS, LeapPolicy, export_npz  # noqa: E402

NAME = "C5HW_EFFORT_V1_seed1"


def sha(p: Path) -> str:
  return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
  cmp = json.loads((E / "results" / "effort_v1_comparison.json").read_text())
  if not cmp["decision"].get("verdict", "").startswith("SINGLE-SEED PASS"):
    sys.exit("readiness gate not met; no export")
  d = E / "deploy" / "policies" / NAME
  out = E / "deploy" / "exports" / f"{NAME}_actor.npz"
  if out.exists():
    sys.exit(f"{out} exists; not overwritten")
  out.parent.mkdir(parents=True, exist_ok=True)
  pol = LeapPolicy(d)
  export_npz(pol, out)
  z = np.load(out)

  # 1. arrays vs checkpoint tensors (bit-exact)
  ck = next(d.glob("model_*.pt"))
  sd = torch.load(ck, map_location="cpu", weights_only=False)
  sd = sd.get("actor_state_dict", sd.get("model_state_dict"))
  pre = "mlp." if any(k.startswith("mlp.") for k in sd) else "actor."  # rsl_rl >= 3 / older layouts
  lin = sorted({k.rsplit(".", 1)[0] for k in sd if k.endswith(".weight") and k.startswith(pre)},
               key=lambda s: int(s.split(".")[-1]))
  n_layers = sum(1 for k in z.files if k.startswith("w"))
  arrays_equal = n_layers == len(lin) and all(
    np.array_equal(z[f"w{i}"], sd[f"{l}.weight"].numpy()) and np.array_equal(z[f"b{i}"], sd[f"{l}.bias"].numpy())
    for i, l in enumerate(lin))
  nm = next((p for p in ("actor_obs_normalizer", "obs_normalizer") if f"{p}._mean" in sd), None)
  norm_equal = nm is not None and np.array_equal(z["obs_mean"], sd[f"{nm}._mean"].numpy().reshape(-1)) and \
    np.array_equal(z["obs_std"], (sd[f"{nm}._std"] if f"{nm}._std" in sd else torch.sqrt(sd[f"{nm}._var"])).numpy().reshape(-1))

  # 2. forward from the npz alone vs the kit actor
  rng = np.random.default_rng(20261124)
  obs = (z["obs_mean"] + z["obs_std"] * rng.standard_normal((4096, z["obs_mean"].size))).astype(np.float32)

  def fwd(x):
    x = (x - z["obs_mean"]) / (z["obs_std"] + RSL_RL_NORMALIZER_EPS)
    for i in range(n_layers - 1):
      h = x @ z[f"w{i}"].T + z[f"b{i}"]
      x = np.where(h > 0, h, np.expm1(np.minimum(h, 20.0)))  # elu, as the kit
    return x @ z[f"w{n_layers - 1}"].T + z[f"b{n_layers - 1}"]

  a_kit = np.stack([pol.actor(o) for o in obs])
  a_npz = np.stack([fwd(o) for o in obs])  # one observation at a time, as the runtime does
  raw = float(np.abs(a_npz - a_kit).max())
  th = float(np.abs(np.tanh(a_npz) - np.tanh(a_kit)).max())
  raw_batched = float(np.abs(fwd(obs) - a_kit).max())  # batched BLAS: float32 summation-order differences only
  spec_equal = json.loads(str(z["spec_json"])) == json.loads(json.dumps(pol.describe()))

  res = {"export": str(out.relative_to(PROJECT)), "export_sha256": sha(out), "checkpoint": ck.name, "checkpoint_sha256": sha(ck),
         "arrays_equal_checkpoint_bit_exact": bool(arrays_equal), "normaliser_equal_checkpoint_bit_exact": bool(norm_equal),
         "npz_forward_vs_kit_actor_max_abs_raw": raw, "npz_forward_vs_kit_actor_max_abs_tanh": th,
         "npz_forward_batched_vs_kit_actor_max_abs_raw": raw_batched, "n_obs": int(len(obs)),
         "spec_equal_kit_describe": bool(spec_equal), "layers": n_layers,
         "passed": bool(arrays_equal and norm_equal and raw < 1e-5 and th < 1e-4 and spec_equal)}
  (E / "results" / "export_verification.json").write_text(json.dumps(res, indent=1) + "\n")
  man = E / "deploy" / "MANIFEST.txt"
  line = f"exports | {out.name} | {res['export_sha256']}\n"
  if line not in man.read_text():
    with open(man, "a") as f:
      f.write(line)
  print(json.dumps(res, indent=1))
  if not res["passed"]:
    sys.exit("export verification FAILED")


if __name__ == "__main__":
  main()
