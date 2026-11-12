#!/usr/bin/env python3
"""Is the estimator informative in the situations that matter?

Uses the held-out *test* split (unseen seeds / episodes).  Difficult
situations are defined from ground truth (palm frame, control rate 20 Hz):

  stable             speed < 1 cm/s, |w_xy| < 0.3 rad/s, drift < 5 mm,
                     not within 0.5 s of a failure
  slip_translation   cube speed > 3 cm/s relative to the palm
  slip_tumble        off-axis angular speed |w_xy| > 1 rad/s
  drift_10_20mm      10 mm <= distance from the episode's initial position < 20 mm
  drift_gt_20mm      distance >= 20 mm
  near_failure       last 10 control steps (0.5 s) before any failure
                     termination (drop, too fast, pose deviation)
  near_drop          last 10 control steps before a cube_fell termination

For each subset: errors, and R^2 *within the subset* (1 - MSE / variance of
the truth inside the subset -- i.e. against an oracle that knows the subset
mean), for position, drift, tilt, velocities and contacts, next to a ridge
(linear) estimator trained on the same inputs.  Drift and tilt are computed
estimator-internally (current estimate vs. the estimator's own estimate at the
episode's first step), exactly as a policy could use them.

Detection: AUROC of estimator-derived scores for each event, compared with the
ridge estimator, a proprioceptive heuristic (newest joint motion |q_t -
q_t-1| from the observation history) and the ground-truth value (upper bound).
Lead time: mean true / estimated drift and tilt over the 2 s before failures.

    CUDA_VISIBLE_DEVICES=3 python scripts/estimation/diagnose_estimator_events.py \
        --estimator data/estimator_models/cube_state_estimator.pt \
        --output-dir data/eval/diagnostics/estimator_events
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from in_hand_rotation_mjlab.estimation.dataset import load_split_tensors, run_dirs  # noqa: E402
from in_hand_rotation_mjlab.estimation.model import load_estimator  # noqa: E402
from in_hand_rotation_mjlab.estimation.rotations import (  # noqa: E402
  cube_symmetry_matrices,
  quat_to_matrix,
  symmetric_geodesic_angle,
  tilt_angle,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_estimator import ridge_baseline  # noqa: E402

FAILURE_CODES = (2, 3, 4, 5)
DROP_CODE = 2
WINDOW = 10  # control steps = 0.5 s
LEAD = 40  # 2 s


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--estimator", required=True)
  p.add_argument("--data-root", default="data/estimator")
  p.add_argument("--ridge-train-rows", type=int, default=1_000_000)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--output-dir", required=True)
  return p.parse_args()


def auroc(score: torch.Tensor, label: torch.Tensor) -> float | None:
  label = label.bool()
  npos, nneg = int(label.sum()), int((~label).sum())
  if npos == 0 or nneg == 0:
    return None
  order = torch.argsort(score.float())
  ranks = torch.empty_like(order, dtype=torch.float64)
  ranks[order] = torch.arange(1, len(score) + 1, device=score.device, dtype=torch.float64)
  return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def r2(pred: torch.Tensor, true: torch.Tensor) -> float:
  if true.shape[0] < 2:
    return float("nan")
  mse = ((pred - true) ** 2).sum(-1).mean() if true.ndim > 1 else ((pred - true) ** 2).mean()
  var = ((true - true.mean(0)) ** 2).sum(-1).mean() if true.ndim > 1 else true.var()
  return float(1 - mse / var.clamp_min(1e-12))


def carry_episode_start(x: torch.Tensor, ep: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
  """[T, n, ...] -> value at the episode's first valid row, carried forward."""
  out = x.clone()
  started = valid[0].clone()
  for t in range(1, x.shape[0]):
    same = ep[t] == ep[t - 1]
    keep = same & started
    out[t][keep] = out[t - 1][keep]
    started = torch.where(same, started | valid[t], valid[t])
  return out


def countdown_to_end(ep: torch.Tensor, end: torch.Tensor, codes: tuple[int, ...]) -> torch.Tensor:
  """Steps until the end of an episode that ends with one of ``codes`` (else -1)."""
  T, n = ep.shape
  out = torch.full((T, n), -1, dtype=torch.int64, device=ep.device)
  hit = torch.zeros(n, dtype=torch.bool, device=ep.device)
  cnt = torch.zeros(n, dtype=torch.int64, device=ep.device)
  for t in range(T - 1, -1, -1):
    boundary = torch.ones(n, dtype=torch.bool, device=ep.device) if t == T - 1 else ep[t] != ep[t + 1]
    is_end = torch.zeros(n, dtype=torch.bool, device=ep.device)
    for c in codes:
      is_end |= end[t] == c
    hit = torch.where(boundary, is_end, hit)
    cnt = torch.where(boundary, torch.zeros_like(cnt), cnt + 1)
    out[t] = torch.where(hit, cnt, out[t])
  return out


def derived(pred: dict, run: dict, sym: torch.Tensor) -> dict[str, torch.Tensor]:
  """Per-row quantities for truth / estimate, all [T, n]."""
  ep, valid = run["episode_id"], run["contact_valid"].bool()
  R = quat_to_matrix(run["quat"])
  d = {
    "valid": valid,
    "pos_err_mm": (pred["pos"] - run["pos"]).norm(dim=-1) * 1000,
    "rot_err_deg": torch.rad2deg(symmetric_geodesic_angle(pred["rotmat"].flatten(0, 1), R.flatten(0, 1), sym)).view(valid.shape),
  }
  for who, src in (("true", {"pos": run["pos"], "R": R, "v": run["lin_vel"], "w": run["ang_vel"]}), ("est", {"pos": pred["pos"], "R": pred["rotmat"], "v": pred["lin_vel"], "w": pred["ang_vel"]})):
    p0 = carry_episode_start(src["pos"], ep, valid)
    tilt = tilt_angle(src["R"])
    d[f"drift_mm_{who}"] = (src["pos"] - p0).norm(dim=-1) * 1000
    d[f"tilt_deg_{who}"] = torch.rad2deg(tilt)
    d[f"tilt_change_deg_{who}"] = torch.rad2deg((tilt - carry_episode_start(tilt, ep, valid)).abs())
    d[f"speed_{who}"] = src["v"].norm(dim=-1)
    d[f"wxy_{who}"] = src["w"][..., :2].norm(dim=-1)
    d[f"wz_{who}"] = src["w"][..., 2]
  d["lin_vel_true"], d["lin_vel_est"] = run["lin_vel"], pred["lin_vel"]
  d["ang_vel_true"], d["ang_vel_est"] = run["ang_vel"], pred["ang_vel"]
  d["pos_true"], d["pos_est"] = run["pos"], pred["pos"]
  d["contact_true"] = run["contact"].bool()
  d["contact_est"] = pred["contact_prob"] >= 0.5
  d["to_failure"] = countdown_to_end(ep, run["end_reason"], FAILURE_CODES)
  d["to_drop"] = countdown_to_end(ep, run["end_reason"], (DROP_CODE,))
  obs = run["obs"]
  d["joint_motion"] = (obs[..., 144:160] - obs[..., 128:144]).norm(dim=-1)
  return d


@torch.no_grad()
def predict(model, obs: torch.Tensor, batch: int = 65536) -> dict[str, torch.Tensor]:
  shp = obs.shape[:-1]
  flat = obs.reshape(-1, obs.shape[-1]).float()
  outs: dict[str, list] = {}
  for s in range(0, flat.shape[0], batch):
    p = model(flat[s : s + batch])
    for k in ("pos", "rotmat", "lin_vel", "ang_vel", "contact_prob"):
      outs.setdefault(k, []).append(p[k])
  return {k: torch.cat(v).view(*shp, *torch.cat(v).shape[1:]) for k, v in outs.items()}


def subset_metrics(D: dict, m: torch.Tensor, tag: str) -> dict:
  n = int(m.sum())
  if n < 50:
    return {"n": n}
  g = lambda k: D[k][m]  # noqa: E731
  ct, ce = g("contact_true"), g("contact_est")
  tp, fp, fn = (ct & ce).sum(), (~ct & ce).sum(), (ct & ~ce).sum()
  f1 = float(2 * tp / (2 * tp + fp + fn).clamp_min(1))
  return {
    "n": n,
    f"{tag}pos_rmse_mm": float(torch.sqrt((g("pos_est") - g("pos_true")).pow(2).sum(-1).mean()) * 1000),
    f"{tag}pos_r2": r2(g("pos_est"), g("pos_true")),
    f"{tag}drift_true_mean_mm": float(g("drift_mm_true").mean()),
    f"{tag}drift_abs_err_mm": float((g("drift_mm_est") - g("drift_mm_true")).abs().mean()),
    f"{tag}drift_r2": r2(g("drift_mm_est"), g("drift_mm_true")),
    f"{tag}tilt_abs_err_deg": float((g("tilt_deg_est") - g("tilt_deg_true")).abs().mean()),
    f"{tag}tilt_r2": r2(g("tilt_deg_est"), g("tilt_deg_true")),
    f"{tag}rot_err_mean_deg": float(g("rot_err_deg").mean()) if "rot_err_deg" in D else None,
    f"{tag}lin_vel_rmse_cmps": float(torch.sqrt((g("lin_vel_est") - g("lin_vel_true")).pow(2).sum(-1).mean()) * 100),
    f"{tag}lin_vel_r2": r2(g("lin_vel_est"), g("lin_vel_true")),
    f"{tag}true_speed_mean_cmps": float(g("speed_true").mean() * 100),
    f"{tag}w_xy_r2": r2(g("ang_vel_est")[:, :2], g("ang_vel_true")[:, :2]),
    f"{tag}yaw_rate_rmse": float(torch.sqrt((g("wz_est") - g("wz_true")).pow(2).mean())),
    f"{tag}yaw_rate_r2": r2(g("wz_est"), g("wz_true")),
    f"{tag}contact_micro_f1": f1,
  }


def main():
  args = parse_args()
  out = Path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)
  dev = args.device
  model = load_estimator(args.estimator, map_location=dev).eval()
  sym = cube_symmetry_matrices(device=dev) if "device" in cube_symmetry_matrices.__code__.co_varnames else cube_symmetry_matrices().to(dev)

  train = load_split_tensors(args.data_root, "train", dev, max_rows=args.ridge_train_rows)
  names = ["obs", "pos", "quat", "lin_vel", "ang_vel", "contact", "contact_valid", "episode_id", "step_in_episode", "end_reason"]
  per_run = {"mlp": [], "ridge": []}
  for run_dir in run_dirs(args.data_root, "test"):
    run = {k: torch.from_numpy(np.load(run_dir / f"{k}.npy")).to(dev) for k in names}
    run["episode_id"] = run["episode_id"].long()
    T, n = run["obs"].shape[:2]
    per_run["mlp"].append(derived(predict(model, run["obs"]), run, sym))
    ridge = ridge_baseline(train, {"obs": run["obs"].reshape(T * n, -1).float()})
    ridge = {k: v.view(T, n, *v.shape[1:]) for k, v in ridge.items()}
    per_run["ridge"].append(derived(ridge, run, sym))
    del run
  del train

  def flat(lst: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.cat([d[k].flatten(0, 1) for d in lst]) for k in lst[0]}

  D, Dr = flat(per_run["mlp"]), flat(per_run["ridge"])
  v = D["valid"]
  near_fail = (D["to_failure"] >= 0) & (D["to_failure"] < WINDOW)
  near_drop = (D["to_drop"] >= 0) & (D["to_drop"] < WINDOW)
  events = {
    "all": v,
    "stable": v & (D["speed_true"] < 0.01) & (D["wxy_true"] < 0.3) & (D["drift_mm_true"] < 5) & ~near_fail,
    "slip_translation": v & (D["speed_true"] > 0.03),
    "slip_tumble": v & (D["wxy_true"] > 1.0),
    "drift_10_20mm": v & (D["drift_mm_true"] >= 10) & (D["drift_mm_true"] < 20),
    "drift_gt_20mm": v & (D["drift_mm_true"] >= 20),
    "near_failure": v & near_fail,
    "near_drop": v & near_drop,
  }
  report: dict = {"estimator": str(Path(args.estimator).resolve()), "window_steps": WINDOW, "subsets": {}}
  for name, m in events.items():
    report["subsets"][name] = {
      "fraction_of_valid_rows": float(m.sum() / v.sum()),
      "mlp": subset_metrics(D, m, ""),
      "ridge": subset_metrics(Dr, m, ""),
    }

  # Detection (all valid rows).
  det_targets = {
    "drift_gt_10mm": D["drift_mm_true"] >= 10,
    "slip_translation": D["speed_true"] > 0.03,
    "slip_tumble": D["wxy_true"] > 1.0,
    "near_failure_0.5s": near_fail,
    "near_drop_0.5s": near_drop,
  }
  scores = {
    "est_drift": D["drift_mm_est"], "est_tilt_change": D["tilt_change_deg_est"], "est_speed": D["speed_est"], "est_w_xy": D["wxy_est"],
    "ridge_drift": Dr["drift_mm_est"], "ridge_tilt_change": Dr["tilt_change_deg_est"], "ridge_speed": Dr["speed_est"], "ridge_w_xy": Dr["wxy_est"],
    "proprio_joint_motion": D["joint_motion"],
    "TRUE_drift": D["drift_mm_true"], "TRUE_tilt_change": D["tilt_change_deg_true"], "TRUE_speed": D["speed_true"], "TRUE_w_xy": D["wxy_true"],
  }
  relevant = {
    "drift_gt_10mm": ["est_drift", "ridge_drift", "proprio_joint_motion"],
    "slip_translation": ["est_speed", "ridge_speed", "proprio_joint_motion"],
    "slip_tumble": ["est_w_xy", "ridge_w_xy", "proprio_joint_motion"],
    "near_failure_0.5s": ["est_drift", "est_tilt_change", "est_speed", "est_w_xy", "ridge_drift", "ridge_tilt_change", "proprio_joint_motion", "TRUE_drift", "TRUE_tilt_change", "TRUE_speed", "TRUE_w_xy"],
    "near_drop_0.5s": ["est_drift", "est_tilt_change", "est_speed", "est_w_xy", "ridge_drift", "proprio_joint_motion", "TRUE_drift", "TRUE_tilt_change", "TRUE_speed", "TRUE_w_xy"],
  }
  report["detection_auroc"] = {
    t: {"positives": int((lab & v).sum()), **{s: auroc(scores[s][v], lab[v]) for s in relevant[t]}} for t, lab in det_targets.items()
  }

  # Lead time before failures.
  lead = {}
  for key in ("drift_mm_true", "drift_mm_est", "tilt_change_deg_true", "tilt_change_deg_est", "pos_err_mm", "rot_err_deg"):
    lead[key] = [float(D[key][v & (D["to_failure"] == k)].mean()) if bool((v & (D["to_failure"] == k)).any()) else None for k in range(LEAD)]
  stable_ref = {k: float(D[k][events["stable"]].mean()) for k in ("pos_err_mm", "rot_err_deg")}
  report["lead_time_before_failure"] = {"steps_before_failure": list(range(LEAD)), **lead, "stable_reference": stable_ref}
  (out / "estimator_events.json").write_text(json.dumps(report, indent=2))

  fig, axes = plt.subplots(1, 3, figsize=(16, 4))
  x = -np.arange(LEAD) * 0.05
  axes[0].plot(x, lead["drift_mm_true"], label="true drift", color="k")
  axes[0].plot(x, lead["drift_mm_est"], label="estimated drift", color="C3")
  axes[0].set_ylabel("mm")
  axes[1].plot(x, lead["tilt_change_deg_true"], label="true tilt change", color="k")
  axes[1].plot(x, lead["tilt_change_deg_est"], label="estimated tilt change", color="C3")
  axes[1].set_ylabel("deg")
  axes[2].plot(x, lead["pos_err_mm"], label="position error [mm]")
  axes[2].plot(x, lead["rot_err_deg"], label="rotation error [deg]")
  axes[2].axhline(stable_ref["pos_err_mm"], ls="--", color="C0", lw=0.8)
  axes[2].axhline(stable_ref["rot_err_deg"], ls="--", color="C1", lw=0.8)
  for ax in axes:
    ax.set_xlabel("time before failure termination [s]")
    ax.legend(fontsize=8)
  fig.suptitle("Estimator before failure terminations (held-out test split; dashed = stable-phase error)")
  fig.tight_layout()
  fig.savefig(out / "lead_time_before_failure.png", dpi=110)
  plt.close(fig)
  print(json.dumps({"subsets": report["subsets"], "detection_auroc": report["detection_auroc"]}, indent=1))


if __name__ == "__main__":
  main()
