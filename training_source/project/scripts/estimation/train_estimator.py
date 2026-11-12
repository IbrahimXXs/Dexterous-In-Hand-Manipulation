#!/usr/bin/env python3
"""Train the proprioceptive cube-state estimator (MLP baseline).

    L = w_pos*L_pos + w_rot*L_rot + w_vel*L_vel + w_contact*L_contact

Data come from ``collect_estimator_data.py`` (split by seed).  The whole
training split is kept on the GPU; minibatches are random rows.  The model
with the lowest validation loss is kept (``best.pt``) and early stopping ends
training after ``--patience`` evaluations without improvement.  At the end the
best model is evaluated on the held-out test split.

Example::

    CUDA_VISIBLE_DEVICES=1 python scripts/estimation/train_estimator.py \
        --data-root data/estimator --run-name mlp_baseline
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from in_hand_rotation_mjlab.estimation.dataset import (
  compute_normalization,
  load_split_tensors,
)
from in_hand_rotation_mjlab.estimation.losses import (
  EstimatorLoss,
  LossWeights,
  MetricAccumulator,
)
from in_hand_rotation_mjlab.estimation.model import (
  CubeStateEstimator,
  EstimatorConfig,
  save_estimator,
)

BATCH_KEYS = ("obs", "pos", "quat", "lin_vel", "ang_vel", "contact", "contact_valid")


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--data-root", default="data/estimator")
  p.add_argument("--log-root", default="data/runs/estimator")
  p.add_argument("--run-name", default="mlp")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 512, 256])
  p.add_argument("--activation", default="elu")
  p.add_argument("--no-layer-norm", action="store_true")
  p.add_argument("--dropout", type=float, default=0.0)
  p.add_argument("--batch-size", type=int, default=4096)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--weight-decay", type=float, default=1e-4)
  p.add_argument("--max-steps", type=int, default=60_000)
  p.add_argument("--eval-every", type=int, default=2_000)
  p.add_argument("--patience", type=int, default=8, help="evaluations without val improvement")
  p.add_argument("--w-pos", type=float, default=1.0)
  p.add_argument("--w-rot", type=float, default=1.0)
  p.add_argument("--w-vel", type=float, default=0.5)
  p.add_argument("--w-contact", type=float, default=0.5)
  p.add_argument("--huber-delta", type=float, default=1.0)
  p.add_argument("--contact-pos-weight", type=float, default=None)
  p.add_argument("--max-train-rows", type=int, default=None)
  p.add_argument("--max-val-rows", type=int, default=1_000_000)
  p.add_argument("--max-test-rows", type=int, default=None)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--skip-test", action="store_true", help="do not touch the test split (evaluated separately)")
  return p.parse_args()


@torch.no_grad()
def evaluate(model, data, loss_fn, device, batch=65536, keep_rot=False) -> dict:
  model.eval()
  acc = MetricAccumulator(device)
  total, parts, nb = 0.0, {}, 0
  n = data["obs"].shape[0]
  for s in range(0, n, batch):
    b = {k: data[k][s : s + batch].to(device, non_blocking=True) for k in BATCH_KEYS}
    pred = model(b["obs"].float())
    loss, info = loss_fn(model, pred, b)
    w = b["obs"].shape[0]
    total += float(loss) * w
    for k, v in info.items():
      parts[k] = parts.get(k, 0.0) + float(v) * w
    nb += w
    acc.update(pred, b, keep_rot_errors=keep_rot)
  out = acc.compute()
  out.update({k: v / nb for k, v in parts.items()})
  model.train()
  return out


def main() -> None:
  args = parse_args()
  torch.manual_seed(args.seed)
  dev = torch.device(args.device)
  run_dir = Path(args.log_root) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
  writer = SummaryWriter(run_dir / "tb")

  t0 = time.time()
  train = load_split_tensors(args.data_root, "train", dev, max_rows=args.max_train_rows, seed=args.seed)
  val = load_split_tensors(args.data_root, "val", dev, max_rows=args.max_val_rows, seed=args.seed)
  n_train = train["obs"].shape[0]
  print(f"[data] train {n_train:,} rows, val {val['obs'].shape[0]:,} rows ({time.time() - t0:.0f}s)")

  cfg = EstimatorConfig(
    hidden_dims=tuple(args.hidden_dims),
    activation=args.activation,
    layer_norm=not args.no_layer_norm,
    dropout=args.dropout,
    meta={"data_root": str(Path(args.data_root).resolve()), "run_dir": str(run_dir)},
  )
  model = CubeStateEstimator(cfg).to(dev)
  stats = compute_normalization(train)
  model.set_normalization(**{k: v.to(dev) for k, v in stats.items()})
  weights = LossWeights(
    pos=args.w_pos,
    rot=args.w_rot,
    vel=args.w_vel,
    contact=args.w_contact,
    huber_delta=args.huber_delta,
    contact_pos_weight=args.contact_pos_weight,
  )
  loss_fn = EstimatorLoss(weights, dev)
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  warmup = 1000
  sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: min(1.0, (s + 1) / warmup)
    * 0.5
    * (1 + math.cos(math.pi * min(1.0, s / args.max_steps))),
  )

  best, best_step, bad = float("inf"), -1, 0
  history = []
  gen = torch.Generator(device=dev).manual_seed(args.seed)
  model.train()
  t0 = time.time()
  for step in range(1, args.max_steps + 1):
    idx = torch.randint(0, n_train, (args.batch_size,), device=dev, generator=gen)
    b = {k: train[k][idx] for k in BATCH_KEYS}
    pred = model(b["obs"].float())
    loss, info = loss_fn(model, pred, b)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 200 == 0:
      for k, v in info.items():
        writer.add_scalar(f"train/{k}", float(v), step)
      writer.add_scalar("train/grad_norm", float(gn), step)
      writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
    if step % args.eval_every == 0 or step == args.max_steps:
      m = evaluate(model, val, loss_fn, dev)
      for k, v in m.items():
        if isinstance(v, (int, float)):
          writer.add_scalar(f"val/{k}", v, step)
      vl = m["loss/total"]
      history.append({"step": step, **m})
      improved = vl < best - 1e-5
      if improved:
        best, best_step, bad = vl, step, 0
        save_estimator(model, run_dir / "best.pt", step=step, val=m, args=vars(args))
      else:
        bad += 1
      print(
        f"[{step:6d}] train {float(loss):.4f} | val {vl:.4f} pos {m['pos_rmse_mm']:.1f}mm "
        f"rot(sym) {m['rot_err_sym_mean_deg']:.1f}deg yawrate {m['yaw_rate_rmse_radps']:.3f} "
        f"contactF1 {m['contact_macro_f1']:.3f} {'*' if improved else ''} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
      )
      if bad >= args.patience:
        print(f"[early stop] no improvement for {bad} evaluations")
        break
  save_estimator(model, run_dir / "last.pt", step=step, args=vars(args))
  (run_dir / "val_history.json").write_text(json.dumps(history, indent=2))

  if args.skip_test:
    writer.close()
    print(f"[done] best step {best_step}; test split not evaluated (--skip-test)")
    return
  # Held-out test evaluation of the best checkpoint.
  from in_hand_rotation_mjlab.estimation.model import load_estimator

  del train
  torch.cuda.empty_cache()
  best_model = load_estimator(run_dir / "best.pt", map_location=dev)
  test = load_split_tensors(args.data_root, "test", dev, max_rows=args.max_test_rows, seed=args.seed)
  tm = evaluate(best_model, test, loss_fn, dev, keep_rot=True)
  tm["best_step"] = best_step
  (run_dir / "test_metrics.json").write_text(json.dumps(tm, indent=2))
  for k, v in tm.items():
    if isinstance(v, (int, float)):
      writer.add_scalar(f"test/{k}", v, best_step)
  writer.close()
  print(json.dumps({k: tm[k] for k in (
    "pos_rmse_mm", "rot_err_sym_mean_deg", "rot_err_raw_mean_deg", "lin_vel_rmse_mps",
    "ang_vel_rmse_radps", "yaw_rate_rmse_radps", "contact_micro_precision",
    "contact_micro_recall", "contact_micro_f1", "num_samples")}, indent=2))
  print(f"[done] {run_dir}")


if __name__ == "__main__":
  main()
