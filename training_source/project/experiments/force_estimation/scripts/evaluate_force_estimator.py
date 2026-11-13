#!/usr/bin/env python3
"""Evaluate the fingertip force estimator and baselines on held-out splits; write tables and figures.

Splits (see run_pipeline.sh): ``test`` (held-out policies), ``test_seen`` (training policies,
new sim seeds), ``robust_*`` (held-out policies under robustness presets).
Baselines (fit on the training split only): constant mean, constant median (= 0 where most
rows are contact-free; the MAE-optimal constant), ridge regression on the standardised
320-D input (lambda chosen on validation), clipped at 0.

    python evaluate_force_estimator.py --model experiments/force_estimation/models/force_estimator.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from in_hand_rotation_mjlab.force_estimation.constants import CONTACT_THRESHOLD_N, FORCE_FINGER_NAMES
from in_hand_rotation_mjlab.force_estimation.dataset import load_split, shard_dirs
from in_hand_rotation_mjlab.force_estimation.model import load_force_estimator

F4 = list(FORCE_FINGER_NAMES)
MAG_BINS = [0.0, CONTACT_THRESHOLD_N, 0.5, 1.0, 2.0, 4.0, np.inf]


def fit_ridge(root, dev, max_rows=1_500_000, seed=0):
  tr = load_split(root, "train")
  va = load_split(root, "val")
  rng = np.random.default_rng(seed)
  idx = np.arange(len(tr["force_mean"]))
  if len(idx) > max_rows:
    idx = np.sort(rng.choice(idx, max_rows, replace=False))
  X = torch.from_numpy(np.ascontiguousarray(tr["obs"][idx])).to(dev).double()
  Y = torch.from_numpy(tr["force_mean"][idx]).to(dev).double()
  mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
  Xs = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, device=dev, dtype=X.dtype)], 1)
  A, B = Xs.T @ Xs, Xs.T @ Y
  vX = torch.from_numpy(np.ascontiguousarray(va["obs"][: 400_000])).to(dev).double()
  vY = torch.from_numpy(va["force_mean"][: 400_000]).to(dev).double()
  vXs = torch.cat([(vX - mu) / sd, torch.ones(len(vX), 1, device=dev, dtype=vX.dtype)], 1)
  best = None
  for lam in (1e-2, 1e0, 1e1, 1e2, 1e3, 1e4):
    reg = lam * torch.eye(A.shape[0], device=dev, dtype=A.dtype)
    reg[-1, -1] = 0
    W = torch.linalg.solve(A + reg, B)
    mae = (vXs @ W).clamp_min(0).sub(vY).abs().mean().item()
    if best is None or mae < best[0]:
      best = (mae, lam, W)
  const_mean = Y.mean(0).float().cpu().numpy()
  const_median = Y.median(0).values.float().cpu().numpy()
  info = {"ridge_lambda": best[1], "ridge_val_mae_N": best[0], "const_mean_N": const_mean.tolist(),
          "const_median_N": const_median.tolist(), "train_rows_used": int(len(idx))}
  return (mu.float(), sd.float(), best[2].float()), const_mean, const_median, info


def predict_all(fn, X, bs=131072):
  return np.concatenate([fn(X[i : i + bs]) for i in range(0, len(X), bs)])


def metrics(pred, y, mask=None):
  if mask is not None:
    pred, y = pred[mask], y[mask]
  if len(y) == 0:
    return None
  e = pred - y
  out = {"n": int(len(y)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
         "bias": float(e.mean())}
  vy = y.var()
  out["r2"] = float(1 - (e ** 2).mean() / vy) if vy > 0 else float("nan")
  return out


def auroc(score, label):
  label = label.astype(bool)
  npos, nneg = label.sum(), (~label).sum()
  if npos == 0 or nneg == 0:
    return float("nan")
  order = np.argsort(score, kind="mergesort")
  ranks = np.empty(len(score))
  ranks[order] = np.arange(1, len(score) + 1)
  # average ranks for ties
  s = score[order]
  i = 0
  while i < len(s):
    j = i
    while j + 1 < len(s) and s[j + 1] == s[i]:
      j += 1
    if j > i:
      ranks[order[i : j + 1]] = (i + j + 2) / 2
    i = j + 1
  return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def classification(score, ytrue, thr):
  c = ytrue > CONTACT_THRESHOLD_N
  p = score > thr
  tp, fp, fn = (p & c).sum(), (p & ~c).sum(), (~p & c).sum()
  prec = tp / max(tp + fp, 1)
  rec = tp / max(tp + fn, 1)
  return {"threshold_N": float(thr), "precision": float(prec), "recall": float(rec),
          "f1": float(2 * prec * rec / max(prec + rec, 1e-12)), "accuracy": float((p == c).mean()),
          "auroc": auroc(score, c), "positive_rate": float(c.mean())}


def transition_mask(sp):
  """Rows within +-1 control step of a contact onset/offset (same episode), per finger: [rows, 4]."""
  c = sp["force_mean"] > CONTACT_THRESHOLD_N
  key = sp["shard_id"].astype(np.int64) * 10**12 + sp["episode_id"] * 10**3
  # rows are time-major per shard; build lookup (shard, env, t) -> row
  order = np.lexsort((sp["t"], sp["env"], sp["shard_id"]))
  out = np.zeros_like(c)
  cs, ks, ts = c[order], key[order], sp["t"][order]
  same_prev = np.r_[False, (ks[1:] == ks[:-1]) & (ts[1:] == ts[:-1] + 1)]
  change = np.zeros_like(cs)
  change[1:] = (cs[1:] != cs[:-1]) & same_prev[1:, None]
  near = change.copy()
  near[:-1] |= change[1:] & same_prev[1:, None]  # the row before a change
  out[order] = near
  return out


def evaluate_split(name, sp, preds, cls_thr):
  y = sp["force_mean"]
  res = {"rows": int(len(y))}
  inc = y > CONTACT_THRESHOLD_N
  trans = transition_mask(sp)
  speed = np.linalg.norm(sp["cube_lin_vel"], axis=1)
  slip = speed > np.quantile(speed, 0.9)
  for mname, P in preds.items():
    r = {"overall": metrics(P, y), "per_finger": {}, "in_contact": {}, "no_contact": {}, "transition": {},
         "slip_top10pct_speed": {}, "by_magnitude": {}, "vs_force_now": {}}
    for k, f in enumerate(F4):
      r["per_finger"][f] = metrics(P[:, k], y[:, k])
      r["in_contact"][f] = metrics(P[:, k], y[:, k], inc[:, k])
      r["no_contact"][f] = metrics(P[:, k], y[:, k], ~inc[:, k])
      r["transition"][f] = metrics(P[:, k], y[:, k], trans[:, k])
      r["slip_top10pct_speed"][f] = metrics(P[:, k], y[:, k], slip)
      r["vs_force_now"][f] = metrics(P[:, k], sp["force_now"][:, k])
      r["by_magnitude"][f] = {f"[{lo},{hi})": metrics(P[:, k], y[:, k], (y[:, k] >= lo) & (y[:, k] < hi))
                              for lo, hi in zip(MAG_BINS[:-1], MAG_BINS[1:])}
    r["in_contact_all"] = metrics(P[inc], y[inc])
    r["transition_all"] = metrics(P[trans], y[trans])
    r["classification"] = {f: classification(P[:, k], y[:, k], cls_thr[mname][k]) for k, f in enumerate(F4)}
    res[mname] = r
  return res


def plots(figdir, sp, P, name, hist_runs):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  y = sp["force_mean"]
  # predicted vs actual (hexbin)
  fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
  for k, ax in enumerate(axes):
    hi = np.quantile(y[:, k], 0.999) + 1e-3
    ax.hexbin(y[:, k], P[:, k], gridsize=60, extent=(0, hi, 0, hi), bins="log", mincnt=1, cmap="viridis")
    ax.plot([0, hi], [0, hi], "r--", lw=1)
    ax.set_title(f"{F4[k]}  (MAE {np.abs(P[:, k] - y[:, k]).mean():.3f} N)")
    ax.set_xlabel("true normal force [N]")
    ax.set_ylabel("predicted [N]")
  fig.suptitle(f"Predicted vs. true fingertip normal force ({name}; log-count hexbin)")
  fig.tight_layout()
  fig.savefig(figdir / f"pred_vs_true_{name}.png", dpi=110)
  plt.close(fig)
  # time series: three envs from the first shard with the most contact activity
  s0 = sp["shard_id"] == 0
  envs = np.unique(sp["env"][s0])
  act = [(y[s0 & (sp["env"] == e)].std(0).sum(), e) for e in envs[:64]]
  pick = [e for _, e in sorted(act, reverse=True)[:3]]
  fig, axes = plt.subplots(len(pick), 4, figsize=(22, 3.2 * len(pick)), squeeze=False)
  for r, e in enumerate(pick):
    m = s0 & (sp["env"] == e)
    t = sp["t"][m] * 0.05
    for k in range(4):
      ax = axes[r, k]
      ax.plot(t, y[m, k], "k", lw=1, label="true (interval mean)")
      ax.plot(t, P[m, k], "C1", lw=1, label="estimate")
      ax.set_title(f"env {e} - {F4[k]}", fontsize=9)
      ax.set_xlabel("time [s]")
  axes[0, 0].legend(fontsize=8)
  fig.suptitle(f"Fingertip normal force over time ({name}, {Path(sp['shards'][0]).name})")
  fig.tight_layout()
  fig.savefig(figdir / f"timeseries_{name}.png", dpi=110)
  plt.close(fig)
  # training curves
  if hist_runs:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for rn, h in hist_runs.items():
      st = [x["step"] for x in h]
      axes[0].plot(st, [x["train_loss"] for x in h], label=rn)
      axes[1].plot(st, [x["val_mae_N"] for x in h], label=rn)
    axes[0].set_title("training loss")
    axes[1].set_title("validation MAE [N]")
    for ax in axes:
      ax.set_xlabel("optimizer step")
      ax.legend(fontsize=8)
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(figdir / "training_curves.png", dpi=110)
    plt.close(fig)


def distribution_report(root, figdir):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  rep = {}
  fig, axes = plt.subplots(1, 4, figsize=(20, 4))
  for split in ("train", "val", "test", "test_seen"):
    if not shard_dirs(root, split):
      continue
    sp = load_split(root, split, with_obs=False)
    y = sp["force_mean"]
    r = {"rows": int(len(y)), "shards": len(sp["shards"]), "per_finger": {}}
    for k, f in enumerate(F4):
      v = y[:, k]
      pos = v[v > CONTACT_THRESHOLD_N]
      r["per_finger"][f] = {"contact_rate": float((v > CONTACT_THRESHOLD_N).mean()), "exact_zero_rate": float((v == 0).mean()),
                            "mean_N": float(v.mean()), "mean_in_contact_N": float(pos.mean()) if len(pos) else 0.0,
                            "p50_in_contact_N": float(np.median(pos)) if len(pos) else 0.0,
                            "p95_in_contact_N": float(np.quantile(pos, 0.95)) if len(pos) else 0.0,
                            "max_N": float(v.max())}
      if split == "train":
        axes[k].hist(pos, bins=100, log=True)
        axes[k].set_title(f"{f}: in-contact force (train), contact rate {r['per_finger'][f]['contact_rate']:.2f}")
        axes[k].set_xlabel("normal force [N]")
    r["other_hand_links_normal_mean_N"] = float(sp["other_normal_now"].mean())
    r["any_slot_overflow_rows"] = int(sum(m.get("rows_with_slot_overflow", 0) for m in sp["metas"]))
    r["audit_sensor_vs_raw_max_abs_diff_N"] = float(max(m.get("audit_sensor_vs_raw_max_abs_diff_N", 0) for m in sp["metas"]))
    r["policies"] = sorted({m["policy_name"] for m in sp["metas"]})
    rep[split] = r
  fig.tight_layout()
  fig.savefig(figdir / "force_distribution_train.png", dpi=110)
  plt.close(fig)
  return rep


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", default="experiments/force_estimation/data")
  p.add_argument("--model", default="experiments/force_estimation/models/force_estimator.pt")
  p.add_argument("--runs", default="experiments/force_estimation/runs")
  p.add_argument("--out", default="experiments/force_estimation/results")
  p.add_argument("--figs", default="experiments/force_estimation/figures")
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()
  dev = torch.device(a.device)
  out, figdir = Path(a.out), Path(a.figs)
  out.mkdir(parents=True, exist_ok=True)
  figdir.mkdir(parents=True, exist_ok=True)

  report = {"distribution": distribution_report(a.data, figdir)}
  model = load_force_estimator(a.model, map_location=dev).to(dev)
  (rmu, rsd, rW), cmean, cmed, binfo = fit_ridge(a.data, dev)
  report["baselines"] = binfo

  def f_model(x):
    with torch.no_grad():
      return model(torch.from_numpy(np.ascontiguousarray(x)).to(dev)).cpu().numpy()

  def f_ridge(x):
    xx = (torch.from_numpy(np.ascontiguousarray(x)).to(dev) - rmu) / rsd
    xx = torch.cat([xx, torch.ones(len(xx), 1, device=dev)], 1)
    return (xx @ rW).clamp_min(0).cpu().numpy()

  fns = {"estimator": f_model, "ridge": f_ridge,
         "const_mean": lambda x: np.broadcast_to(cmean, (len(x), 4)).copy(),
         "const_median": lambda x: np.broadcast_to(cmed, (len(x), 4)).copy()}
  # contact-classification thresholds chosen on validation (max F1 of predicted force)
  va = load_split(a.data, "val")
  vi = np.arange(len(va["force_mean"]))[:400_000]
  thr = {}
  for mname, fn in fns.items():
    P = predict_all(fn, va["obs"][vi])
    yv = va["force_mean"][vi]
    ts = []
    for k in range(4):
      cands = np.unique(np.quantile(P[:, k], np.linspace(0.0, 0.995, 200)))
      best = max(cands, key=lambda c: classification(P[:, k], yv[:, k], c)["f1"]) if len(cands) > 1 else cands[0]
      ts.append(float(best))
    thr[mname] = ts
  report["contact_thresholds_from_val"] = thr
  del va

  hist_runs = {}
  for h in sorted(Path(a.runs).glob("*/history.json")):
    hist_runs[h.parent.name] = json.loads(h.read_text())
  splits = [s for s in ["test", "test_seen"] + sorted(d.name for d in Path(a.data).glob("robust_*")) if shard_dirs(a.data, s)]
  report["splits"] = {}
  for s in splits:
    sp = load_split(a.data, s, dr=(s == "test"))
    preds = {m: predict_all(fn, sp["obs"]) for m, fn in fns.items()}
    for m, P in preds.items():
      if not np.isfinite(P).all() or (P < 0).any():
        raise FloatingPointError(f"{m} produced invalid predictions on {s}")
    report["splits"][s] = evaluate_split(s, sp, preds, thr)
    # per held-out policy breakdown
    if s == "test":
      pol = np.array([sp["metas"][i]["policy_name"].split("_it")[0] for i in sp["shard_id"]])
      report["splits"][s]["estimator_by_policy"] = {
        q: {f: metrics(preds["estimator"][pol == q, k], sp["force_mean"][pol == q, k]) for k, f in enumerate(F4)}
        for q in sorted(set(pol))}
      # robustness to dynamics randomisation (quartiles of friction / mass)
      rob = {}
      for fld in ("dr_friction", "dr_cube_mass", "dr_action_delay_steps", "dr_effort_limit"):
        if fld in sp:
          qs = np.quantile(sp[fld], [0, 0.25, 0.5, 0.75, 1.0])
          rob[fld] = [metrics(preds["estimator"], sp["force_mean"], (sp[fld] >= lo) & (sp[fld] <= hi))
                      for lo, hi in zip(qs[:-1], qs[1:])]
      report["splits"][s]["estimator_by_dr_quartile"] = rob
      plots(figdir, sp, preds["estimator"], s, hist_runs)
    elif s == "test_seen":
      plots(figdir, sp, preds["estimator"], s, None)
    print(f"[eval] {s}: estimator MAE {report['splits'][s]['estimator']['overall']['mae']:.4f} N  "
          f"ridge {report['splits'][s]['ridge']['overall']['mae']:.4f}  const_median {report['splits'][s]['const_median']['overall']['mae']:.4f}",
          flush=True)
  # latency
  x1 = torch.zeros(1, 320)
  m_cpu = load_force_estimator(a.model, map_location="cpu")
  import time as _t
  with torch.no_grad():
    for _ in range(50):
      m_cpu(x1)
    t0 = _t.perf_counter()
    for _ in range(1000):
      m_cpu(x1)
    report["latency_cpu_single_ms"] = (_t.perf_counter() - t0)
  (out / "eval_report.json").write_text(json.dumps(report, indent=1))
  (out / "results.md").write_text(render_md(report))
  print((out / "results.md").read_text())


def render_md(rep):
  L = ["# Fingertip force estimator: evaluation (generated)\n"]
  L.append("## Dataset (primary label = interval-mean normal force; rows with step_in_episode > 0)\n")
  L.append("| split | rows | policies | " + " | ".join(f"{f} contact rate / mean in contact [N] / p95 [N]" for f in F4) + " |")
  L.append("|---|---|---|" + "---|" * 4)
  for s, r in rep["distribution"].items():
    L.append(f"| {s} | {r['rows']:,} | {', '.join(r['policies'])} | " + " | ".join(
      f"{r['per_finger'][f]['contact_rate']:.2f} / {r['per_finger'][f]['mean_in_contact_N']:.2f} / {r['per_finger'][f]['p95_in_contact_N']:.2f}" for f in F4) + " |")
  L.append(f"\nBaselines: {json.dumps(rep['baselines'])}\n")
  for s, R in rep["splits"].items():
    L.append(f"\n## Split `{s}` ({R['rows']:,} rows)\n")
    L.append("| model | MAE [N] | RMSE [N] | R2 | in-contact MAE | transition MAE | " + " | ".join(f"{f} MAE / RMSE / in-contact MAE / F1 / AUROC" for f in F4) + " |")
    L.append("|---|---|---|---|---|---|" + "---|" * 4)
    for m in ("estimator", "ridge", "const_mean", "const_median"):
      r = R[m]
      cells = []
      for f in F4:
        ic = r["in_contact"][f]
        cl = r["classification"][f]
        cells.append(f"{r['per_finger'][f]['mae']:.3f} / {r['per_finger'][f]['rmse']:.3f} / "
                     f"{ic['mae'] if ic else float('nan'):.3f} / {cl['f1']:.2f} / {cl['auroc']:.2f}")
      L.append(f"| {m} | {r['overall']['mae']:.4f} | {r['overall']['rmse']:.4f} | {r['overall']['r2']:.3f} | "
               f"{r['in_contact_all']['mae']:.3f} | {r['transition_all']['mae'] if r['transition_all'] else float('nan'):.3f} | " + " | ".join(cells) + " |")
  L.append(f"\nCPU single-sample latency: {rep['latency_cpu_single_ms']:.3f} ms\n")
  return "\n".join(L) + "\n"


if __name__ == "__main__":
  main()
