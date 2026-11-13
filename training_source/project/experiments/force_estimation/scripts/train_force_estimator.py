#!/usr/bin/env python3
"""Train the fingertip normal-force estimator (320-D obs -> 4 forces, N).

Primary label: ``force_mean`` (mean normal force over the last control interval).
Normalisation: per-input mean/std and per-finger force scale (RMS of the label)
from the TRAINING split only, stored in the checkpoint.  Selection: lowest
validation MAE (N, mean over fingers).  The test splits are never read.

    python train_force_estimator.py --head direct --seed 0 --run-name direct_s0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from in_hand_rotation_mjlab.force_estimation.constants import CONTACT_THRESHOLD_N, FORCE_FINGER_NAMES
from in_hand_rotation_mjlab.force_estimation.dataset import load_split
from in_hand_rotation_mjlab.force_estimation.model import FingertipForceEstimator, ForceEstimatorConfig


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--data", default="experiments/force_estimation/data")
  p.add_argument("--runs", default="experiments/force_estimation/runs")
  p.add_argument("--run-name", required=True)
  p.add_argument("--head", choices=["direct", "hurdle"], default="direct")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--steps", type=int, default=40000)
  p.add_argument("--batch", type=int, default=4096)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--wd", type=float, default=1e-4)
  p.add_argument("--huber", type=float, default=1.0)
  p.add_argument("--eval-every", type=int, default=2000)
  p.add_argument("--max-val-rows", type=int, default=600_000)
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()

  torch.manual_seed(a.seed)
  np.random.seed(a.seed)
  run = Path(a.runs) / a.run_name
  run.mkdir(parents=True, exist_ok=True)
  (run / "args.json").write_text(json.dumps(vars(a), indent=2))
  dev = torch.device(a.device)

  t0 = time.time()
  tr = load_split(a.data, "train")
  va = load_split(a.data, "val")
  Xtr = torch.from_numpy(np.ascontiguousarray(tr["obs"])).to(dev)
  Ytr = torch.from_numpy(tr["force_mean"].astype(np.float32)).to(dev)
  rng = np.random.default_rng(a.seed)
  vi = np.arange(len(va["force_mean"]))
  if len(vi) > a.max_val_rows:
    vi = np.sort(rng.choice(vi, a.max_val_rows, replace=False))
  Xva = torch.from_numpy(np.ascontiguousarray(va["obs"][vi])).to(dev)
  Yva = torch.from_numpy(va["force_mean"][vi].astype(np.float32)).to(dev)
  print(f"[train] train rows {len(Xtr):,}  val rows {len(Xva):,}  load {time.time() - t0:.0f}s", flush=True)
  for name, X, Y in (("train", Xtr, Ytr), ("val", Xva, Yva)):
    if not (torch.isfinite(X).all() and torch.isfinite(Y).all()):
      raise FloatingPointError(f"non-finite {name} data")

  obs_mean = Xtr.mean(0)
  obs_std = Xtr.std(0)
  force_scale = Ytr.pow(2).mean(0).sqrt().clamp_min(1e-3)  # RMS per finger, N
  model = FingertipForceEstimator(ForceEstimatorConfig(head=a.head)).to(dev)
  model.set_normalization(obs_mean, obs_std, force_scale)
  stats = {"obs_mean": obs_mean.cpu().tolist(), "obs_std": obs_std.cpu().tolist(), "force_scale_N": force_scale.cpu().tolist(),
           "train_rows": len(Xtr), "val_rows": len(Xva), "train_shards": tr["shards"], "val_shards": va["shards"]}
  (run / "normalization.json").write_text(json.dumps(stats))
  opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
  sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))

  def loss_fn(x, y):
    o = model.raw_outputs(x)
    yn = y / model.force_scale
    reg = F.huber_loss(o["force_n"], yn, delta=a.huber)
    if a.head == "direct":
      return reg, {"reg": reg.item()}
    c = (y > CONTACT_THRESHOLD_N).float()
    bce = F.binary_cross_entropy_with_logits(o["logit"], c)
    mag = (F.huber_loss(o["mag_n"], yn, delta=a.huber, reduction="none") * c).sum() / c.sum().clamp_min(1.0)
    return reg + bce + mag, {"reg": reg.item(), "bce": bce.item(), "mag": mag.item()}

  @torch.no_grad()
  def evaluate():
    model.eval()
    preds = torch.cat([model(Xva[i : i + 65536]) for i in range(0, len(Xva), 65536)])
    model.train()
    if not torch.isfinite(preds).all() or (preds < 0).any():
      raise FloatingPointError("invalid predictions (non-finite or negative)")
    err = preds - Yva
    mae = err.abs().mean(0)
    rmse = err.pow(2).mean(0).sqrt()
    inc = Yva > CONTACT_THRESHOLD_N
    mae_c = torch.stack([err[inc[:, k], k].abs().mean() for k in range(4)])
    return {"val_mae_N": mae.mean().item(), "val_rmse_N": rmse.mean().item(),
            "val_mae_per_finger": dict(zip(FORCE_FINGER_NAMES, mae.tolist())),
            "val_rmse_per_finger": dict(zip(FORCE_FINGER_NAMES, rmse.tolist())),
            "val_mae_in_contact_per_finger": dict(zip(FORCE_FINGER_NAMES, mae_c.tolist()))}

  hist, best, best_step = [], float("inf"), -1
  g = torch.Generator(device=dev)
  g.manual_seed(a.seed)
  model.train()
  run_loss = 0.0
  for step in range(1, a.steps + 1):
    idx = torch.randint(0, len(Xtr), (a.batch,), device=dev, generator=g)
    loss, parts = loss_fn(Xtr[idx], Ytr[idx])
    if not torch.isfinite(loss):
      raise FloatingPointError(f"non-finite loss at step {step}: {parts}")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    run_loss += loss.item()
    if step % a.eval_every == 0 or step == a.steps:
      ev = evaluate()
      rec = {"step": step, "train_loss": run_loss / a.eval_every, "grad_norm": float(gn), "lr": sched.get_last_lr()[0], **ev}
      run_loss = 0.0
      hist.append(rec)
      flag = ""
      if ev["val_mae_N"] < best:
        best, best_step = ev["val_mae_N"], step
        torch.save(model.checkpoint({"step": step, "val": ev, "args": vars(a)}), run / "best.pt")
        flag = " *"
      print(f"[{step:6d}] loss {rec['train_loss']:.4f} val MAE {ev['val_mae_N']:.4f} N RMSE {ev['val_rmse_N']:.4f} N "
            f"({time.time() - t0:.0f}s){flag}", flush=True)
      (run / "history.json").write_text(json.dumps(hist, indent=1))
  torch.save(model.checkpoint({"step": a.steps, "args": vars(a)}), run / "last.pt")
  (run / "summary.json").write_text(json.dumps({"best_step": best_step, "best_val_mae_N": best, "head": a.head,
                                                "seed": a.seed, "elapsed_s": time.time() - t0}, indent=2))
  print(f"[done] best step {best_step} val MAE {best:.4f} N")


if __name__ == "__main__":
  main()
