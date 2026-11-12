#!/usr/bin/env python3
"""Independent evaluation of a trained cube-state estimator (Task 4).

1. Held-out test split (seeds never used for training / model selection).
2. Baselines
     prior        training-set mean position, zero velocity, per-fingertip
                  majority contact, constant mean-yaw-free upright orientation
                  (the training-mean rotation, symmetry-reduced)
     init_hold    the *episode-initial* ground-truth state held constant
                  (privileged at t=0 - an optimistic "fixed initial pose")
     ridge        linear least squares from the same 320 inputs
3. Breakdown by randomised cube mass, friction, actuator delay, motor
   damping and initial grasp (number of fingertips touching at episode start).
4. Breakdown by phase (defined from ground truth):
     pre_failure  last 1 s before a failure termination (drop / pose
                  deviation / too fast)
     slipping     cube speed relative to palm > 3 cm/s (not pre_failure)
     rotating     yaw rate > 0.1 rad/s (not slipping / pre_failure)
     holding      everything else
5. Time-series plots of prediction vs ground truth for test episodes.
6. Leakage checks: the dataset's actor terms must be exactly
   (joint_pos, prev_commanded_joint_pos), the estimator must take 320 inputs,
   and inputs are verified not to contain the privileged labels.

Example::

    CUDA_VISIBLE_DEVICES=1 python scripts/estimation/evaluate_estimator.py \
        --estimator data/estimator_models/cube_state_estimator.pt \
        --data-root data/estimator --output-dir data/eval/estimator
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from in_hand_rotation_mjlab.estimation.dataset import (  # noqa: E402
  FAILURE_REASONS,
  load_run_trajectories,
  load_split_tensors,
  read_meta,
  run_dirs,
)
from in_hand_rotation_mjlab.estimation.layout import ACTOR_OBS_DIM, FINGERTIP_NAMES  # noqa: E402
from in_hand_rotation_mjlab.estimation.losses import MetricAccumulator  # noqa: E402
from in_hand_rotation_mjlab.estimation.model import load_estimator  # noqa: E402
from in_hand_rotation_mjlab.estimation.rotations import (  # noqa: E402
  matrix_to_rot6d,
  quat_to_matrix,
  rot6d_to_matrix,
  tilt_angle,
  yaw_mod_quarter_turn,
)

KEYS = ("obs", "pos", "quat", "lin_vel", "ang_vel", "contact", "contact_valid")


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--estimator", required=True)
  p.add_argument("--data-root", default="data/estimator")
  p.add_argument("--output-dir", required=True)
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--max-test-rows", type=int, default=None)
  p.add_argument("--ridge-train-rows", type=int, default=1_000_000)
  p.add_argument("--num-plot-episodes", type=int, default=4)
  return p.parse_args()


# ------------------------------------------------------------------ helpers
def episode_context(data: dict) -> dict[str, torch.Tensor]:
  """Per-row episode-initial state, phase labels and grasp descriptors.

  Rows are flattened [T*n] in time-major order per run; we rebuild [T, n]
  views run by run using ``run_index``.
  """
  n_rows = data["obs"].shape[0]
  dev = data["obs"].device
  init = {k: torch.empty_like(data[k]) for k in ("pos", "quat", "contact")}
  pre_fail = torch.zeros(n_rows, dtype=torch.bool, device=dev)
  init_contacts = torch.zeros(n_rows, dtype=torch.int64, device=dev)
  offset = 0
  for ri in torch.unique(data["run_index"]).tolist():
    m = data["run_index"] == ri
    rows = int(m.sum())
    ep = data["episode_id"][m]
    n_env = int(torch.unique(ep // 100_000).numel())
    T = rows // n_env
    ep2 = ep.view(T, n_env)
    end = data["end_reason"][m].view(T, n_env)
    sie = data["step_in_episode"][m].view(T, n_env)
    for k in init:
      src = data[k][m].view(T, n_env, *data[k].shape[1:])
      out = src.clone()
      # Initial state = first row with valid contacts (step_in_episode == 1),
      # carried forward through the rest of the episode.
      for t in range(1, T):
        keep = (ep2[t] == ep2[t - 1]) & (sie[t] > 1)
        out[t][keep] = out[t - 1][keep]
      init[k][offset : offset + rows] = out.reshape(rows, *data[k].shape[1:])
    # Rows within the last 20 steps (1 s) of an episode that ends in failure.
    fail_end = torch.zeros(T, n_env, dtype=torch.bool, device=dev)
    for code in FAILURE_REASONS:
      fail_end |= end == code
    pf = torch.zeros_like(fail_end)
    countdown = torch.zeros(n_env, dtype=torch.int64, device=dev)
    for t in range(T - 1, -1, -1):
      boundary = torch.ones_like(fail_end[t]) if t == T - 1 else ep2[t] != ep2[t + 1]
      reset_val = torch.where(fail_end[t], 20, 0)
      countdown = torch.where(boundary, reset_val, countdown - 1)
      pf[t] = countdown > 0
    pre_fail[offset : offset + rows] = pf.reshape(-1)
    offset += rows
  init_contacts = init["contact"].sum(-1).long()
  speed = data["lin_vel"].norm(dim=-1)
  slipping = (speed > 0.03) & ~pre_fail
  rotating = (data["ang_vel"][:, 2] > 0.1) & ~slipping & ~pre_fail
  holding = ~(pre_fail | slipping | rotating)
  return {
    "init_pos": init["pos"],
    "init_quat": init["quat"],
    "init_contacts": init_contacts,
    "init_contact": init["contact"],
    "phase_pre_failure": pre_fail,
    "phase_slipping": slipping,
    "phase_rotating": rotating,
    "phase_holding": holding,
  }


@torch.no_grad()
def predict_all(model, obs: torch.Tensor, batch: int = 65536) -> dict[str, torch.Tensor]:
  outs: dict[str, list] = {}
  for s in range(0, obs.shape[0], batch):
    p = model(obs[s : s + batch].float())
    for k in ("pos", "rotmat", "lin_vel", "ang_vel", "contact_prob"):
      outs.setdefault(k, []).append(p[k])
  return {k: torch.cat(v) for k, v in outs.items()}


def metrics(pred: dict, data: dict, mask: torch.Tensor | None = None) -> dict:
  acc = MetricAccumulator(data["obs"].device)
  sel = slice(None) if mask is None else mask
  b = {k: data[k][sel] for k in KEYS if k != "obs"}
  p = {k: v[sel] for k, v in pred.items()}
  if b["pos"].shape[0] == 0:
    return {"num_samples": 0}
  acc.update(p, b, keep_rot_errors=True)
  return acc.compute()


def ridge_baseline(train: dict, test: dict, lam: float = 1e-2) -> dict:
  x = train["obs"].double()
  mu, sd = x.mean(0), x.std(0).clamp_min(1e-4)
  xn = torch.cat([(x - mu) / sd, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], 1)
  r6 = matrix_to_rot6d(quat_to_matrix(train["quat"].double()))
  y = torch.cat([train["pos"].double(), r6, train["lin_vel"].double(), train["ang_vel"].double(), train["contact"].double()], 1)
  a = xn.T @ xn + lam * x.shape[0] * torch.eye(xn.shape[1], device=x.device, dtype=x.dtype)
  w = torch.linalg.solve(a, xn.T @ y)
  out = {k: [] for k in ("pos", "rotmat", "lin_vel", "ang_vel", "contact_prob")}
  for s in range(0, test["obs"].shape[0], 500_000):
    xt = test["obs"][s : s + 500_000].double()
    xt = torch.cat([(xt - mu) / sd, torch.ones(xt.shape[0], 1, device=xt.device, dtype=xt.dtype)], 1)
    yh = (xt @ w).float()
    out["pos"].append(yh[:, 0:3])
    out["rotmat"].append(rot6d_to_matrix(yh[:, 3:9]))
    out["lin_vel"].append(yh[:, 9:12])
    out["ang_vel"].append(yh[:, 12:15])
    out["contact_prob"].append(yh[:, 15:19].clamp(0, 1))
  return {k: torch.cat(v) for k, v in out.items()}


def leakage_checks(data_root: str, model) -> dict:
  checks = {}
  for split in ("train", "val", "test"):
    for run in run_dirs(data_root, split):
      meta = read_meta(run)
      checks[f"{split}/{run.name}/actor_terms"] = meta["actor_terms"]
      if meta["actor_terms"] != ["joint_pos", "prev_commanded_joint_pos"]:
        raise RuntimeError(f"{run}: actor observation contains {meta['actor_terms']}")
  if model.cfg.obs_dim != ACTOR_OBS_DIM:
    raise RuntimeError("estimator input is not the 320-D actor observation")
  checks["estimator_input_dim"] = model.cfg.obs_dim
  return checks


def label_correlation_check(data: dict) -> dict:
  """No input column may be (nearly) identical to a privileged label."""
  x = data["obs"][:200_000].double()
  labels = torch.cat(
    [data["pos"][:200_000], data["lin_vel"][:200_000], data["ang_vel"][:200_000]], 1
  ).double()
  xc = (x - x.mean(0)) / x.std(0).clamp_min(1e-9)
  yc = (labels - labels.mean(0)) / labels.std(0).clamp_min(1e-9)
  corr = (xc.T @ yc) / x.shape[0]
  return {"max_abs_input_label_correlation": float(corr.abs().max())}


def plot_episodes(model, run: Path, out_dir: Path, n_plot: int, device: str) -> list[str]:
  meta = read_meta(run)
  T, n = meta["num_steps"], meta["num_envs"]
  ep = np.load(run / "episode_id.npy", mmap_mode="r")
  end = np.load(run / "end_reason.npy", mmap_mode="r")
  # prefer long episodes, include one that ends in failure if available
  lengths = {}
  for e in range(min(n, 256)):
    ids, counts = np.unique(np.asarray(ep[:, e]), return_counts=True)
    j = counts.argmax()
    lengths[e] = (counts[j], ids[j], bool(np.isin(np.asarray(end[:, e])[np.asarray(ep[:, e]) == ids[j]], list(FAILURE_REASONS)).any()))
  failing = [e for e, v in lengths.items() if v[2] and v[0] > 100][:1]
  longest = sorted(lengths, key=lambda e: -lengths[e][0])[: max(0, n_plot - len(failing))]
  envs = longest + failing
  traj = load_run_trajectories(run, envs)
  dt = meta["control_dt"]
  files = []
  for j, e in enumerate(envs):
    rows = traj["episode_id"][:, j] == lengths[e][1]
    obs = torch.from_numpy(traj["obs"][rows, j]).to(device)
    with torch.no_grad():
      pr = model(obs)
    t = np.arange(rows.sum()) * dt
    gt_r = quat_to_matrix(torch.from_numpy(traj["quat"][rows, j]))
    fig, ax = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for i, c in enumerate("xyz"):
      ax[0].plot(t, traj["pos"][rows, j, i] * 1000, color=f"C{i}", label=f"{c} true")
      ax[0].plot(t, pr["pos"][:, i].cpu() * 1000, "--", color=f"C{i}", label=f"{c} est")
    ax[0].set_ylabel("position [mm]")
    ax[0].legend(ncol=6, fontsize=7)
    ax[1].plot(t, np.degrees(yaw_mod_quarter_turn(gt_r).numpy()), label="true")
    ax[1].plot(t, np.degrees(yaw_mod_quarter_turn(pr["rotmat"].cpu()).numpy()), "--", label="est")
    ax[1].set_ylabel("yaw mod 90° [deg]")
    ax[1].legend(fontsize=7)
    ax[2].plot(t, np.degrees(tilt_angle(gt_r).numpy()), label="true")
    ax[2].plot(t, np.degrees(tilt_angle(pr["rotmat"].cpu()).numpy()), "--", label="est")
    ax[2].set_ylabel("tilt [deg]")
    ax[3].plot(t, traj["ang_vel"][rows, j, 2], label="true")
    ax[3].plot(t, pr["ang_vel"][:, 2].cpu(), "--", label="est")
    ax[3].set_ylabel("yaw rate [rad/s]")
    for f in range(4):
      ax[4].fill_between(t, f, f + 0.8 * traj["contact"][rows, j, f], step="mid", alpha=0.3, color=f"C{f}")
      ax[4].plot(t, f + 0.8 * pr["contact_prob"][:, f].cpu(), color=f"C{f}", lw=1)
    ax[4].set_yticks(np.arange(4) + 0.4, FINGERTIP_NAMES)
    ax[4].set_ylabel("contact (fill=true, line=est)")
    ax[4].set_xlabel("time [s]")
    reason = int(traj["end_reason"][rows, j].max())
    fig.suptitle(f"{run.parent.name}/{run.name} env {e}  (end reason {reason})")
    fig.tight_layout()
    path = out_dir / f"timeseries_{run.name}_env{e}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    files.append(str(path))
  return files


def binned(pred, data, key: str, values: torch.Tensor, edges: list[float]) -> list[dict]:
  rows = []
  for lo, hi in zip(edges[:-1], edges[1:]):
    m = (values >= lo) & (values < hi)
    if int(m.sum()) < 1000:
      continue
    mm = metrics(pred, data, m)
    rows.append({
      key: [lo, hi],
      "n": int(m.sum()),
      "pos_rmse_mm": mm["pos_rmse_mm"],
      "rot_err_sym_mean_deg": mm["rot_err_sym_mean_deg"],
      "yaw_rate_rmse_radps": mm["yaw_rate_rmse_radps"],
      "contact_micro_f1": mm["contact_micro_f1"],
    })
  return rows


def main():
  args = parse_args()
  out = Path(args.output_dir)
  out.mkdir(parents=True, exist_ok=True)
  dev = args.device
  model = load_estimator(args.estimator, map_location=dev).eval()
  report: dict = {"estimator": str(Path(args.estimator).resolve())}
  report["leakage"] = leakage_checks(args.data_root, model)

  test = load_split_tensors(args.data_root, "test", dev, max_rows=args.max_test_rows, include_aux=True, include_dr=True)
  report["leakage"].update(label_correlation_check(test))
  ctx = episode_context(test)
  pred = predict_all(model, test["obs"])
  report["test"] = metrics(pred, test)

  # ---- baselines
  train = load_split_tensors(args.data_root, "train", dev, max_rows=args.ridge_train_rows)
  n = test["obs"].shape[0]
  pos_mean = train["pos"].mean(0)
  prior_contact = (train["contact"].float().mean(0) > 0.5).float()
  mean_r = quat_to_matrix(train["quat"][:200_000]).mean(0)
  u, _, vh = torch.linalg.svd(mean_r)
  mean_r = u @ vh
  prior = {
    "pos": pos_mean.expand(n, 3),
    "rotmat": mean_r.expand(n, 3, 3),
    "lin_vel": torch.zeros(n, 3, device=dev),
    "ang_vel": torch.zeros(n, 3, device=dev),
    "contact_prob": prior_contact.expand(n, 4),
  }
  init_hold = {
    "pos": ctx["init_pos"],
    "rotmat": quat_to_matrix(ctx["init_quat"]),
    "lin_vel": torch.zeros(n, 3, device=dev),
    "ang_vel": torch.zeros(n, 3, device=dev),
    "contact_prob": ctx["init_contact"].float(),
  }
  ridge = ridge_baseline(train, test)
  del train
  report["baselines"] = {
    "prior": metrics(prior, test),
    "init_hold_privileged": metrics(init_hold, test),
    "ridge_linear": metrics(ridge, test),
  }

  # ---- phases
  report["phases"] = {}
  for ph in ("rotating", "holding", "slipping", "pre_failure"):
    m = ctx[f"phase_{ph}"]
    report["phases"][ph] = {
      "fraction_of_rows": float(m.float().mean()),
      "estimator": metrics(pred, test, m),
      "init_hold_privileged": metrics(init_hold, test, m),
    }

  # ---- randomisation / grasp breakdowns
  bd = {}
  if "dr_cube_mass" in test:
    bd["cube_mass_kg"] = binned(pred, test, "cube_mass_kg", test["dr_cube_mass"], [0.0, 0.07, 0.085, 0.1, 0.11, 0.13])
  if "dr_friction" in test:
    bd["friction"] = binned(pred, test, "friction", test["dr_friction"], [0.6, 0.8, 1.0, 1.2, 1.41])
  if "dr_action_delay_steps" in test:
    bd["action_delay_physics_steps"] = binned(pred, test, "delay", test["dr_action_delay_steps"], [5, 8, 11, 13, 16])
  if "dr_motor_damping" in test:
    q = torch.quantile(test["dr_motor_damping"][:1_000_000], torch.tensor([0, 0.25, 0.5, 0.75, 1.0], device=dev)).tolist()
    q[-1] += 1e-6
    bd["motor_damping_quartile"] = binned(pred, test, "damping", test["dr_motor_damping"], q)
  bd["initial_contacts"] = binned(pred, test, "initial_contacts", ctx["init_contacts"].float(), [0, 1, 2, 3, 4, 5])
  bd["step_in_episode"] = binned(pred, test, "step_in_episode", test["step_in_episode"].float(), [0, 10, 40, 100, 200, 600])
  report["breakdowns"] = bd

  # ---- plots
  report["plots"] = plot_episodes(model, run_dirs(args.data_root, "test")[0], out, args.num_plot_episodes, dev)
  fig, ax = plt.subplots(1, 3, figsize=(15, 4))
  for name, src in (("estimator", report["test"]), ("ridge", report["baselines"]["ridge_linear"]), ("prior", report["baselines"]["prior"]), ("init_hold*", report["baselines"]["init_hold_privileged"])):
    ax[0].bar(name, src["pos_rmse_mm"])
    ax[1].bar(name, src["rot_err_sym_mean_deg"])
    ax[2].bar(name, src["yaw_rate_rmse_radps"])
  ax[0].set_ylabel("position RMSE [mm]")
  ax[1].set_ylabel("orientation error (symmetry-aware) [deg]")
  ax[2].set_ylabel("yaw-rate RMSE [rad/s]")
  fig.suptitle("Held-out test split (* = privileged at t=0)")
  fig.tight_layout()
  fig.savefig(out / "baselines.png", dpi=110)
  plt.close(fig)
  report["plots"].append(str(out / "baselines.png"))

  (out / "estimator_report.json").write_text(json.dumps(report, indent=2))
  t = report["test"]
  print(json.dumps({
    "test": {k: t[k] for k in ("pos_rmse_mm", "rot_err_sym_mean_deg", "rot_err_sym_median_deg", "lin_vel_rmse_mps", "ang_vel_rmse_radps", "yaw_rate_rmse_radps", "contact_micro_precision", "contact_micro_recall", "contact_micro_f1")},
    "baselines": {b: {k: v[k] for k in ("pos_rmse_mm", "rot_err_sym_mean_deg", "yaw_rate_rmse_radps", "contact_micro_f1")} for b, v in report["baselines"].items()},
    "phases": {p: {"frac": v["fraction_of_rows"], "pos_rmse_mm": v["estimator"].get("pos_rmse_mm"), "rot_deg": v["estimator"].get("rot_err_sym_mean_deg"), "yaw_rate_rmse": v["estimator"].get("yaw_rate_rmse_radps"), "contact_f1": v["estimator"].get("contact_micro_f1")} for p, v in report["phases"].items()},
    "leakage": {k: v for k, v in report["leakage"].items() if "actor_terms" not in k},
  }, indent=2))
  print(f"[done] {out / 'estimator_report.json'}")


if __name__ == "__main__":
  main()
