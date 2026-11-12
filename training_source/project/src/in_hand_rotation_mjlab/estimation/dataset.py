"""On-disk dataset written by ``scripts/estimation/collect_estimator_data.py``.

Layout (one directory per collection run; one run = one seed = one split)::

    <root>/<split>/seed_<seed>/
        meta.json
        obs.npy             float32 [T, N, 320]  actor observation (exactly as
                                                 fed to the PPO actor)
        pos.npy             float32 [T, N, 3]    cube pos in palm frame
        quat.npy            float32 [T, N, 4]    cube quat (wxyz) in palm frame
        lin_vel.npy         float32 [T, N, 3]
        ang_vel.npy         float32 [T, N, 3]
        contact.npy         uint8   [T, N, 4]    if, mf, rf, th
        contact_valid.npy   uint8   [T, N]       0 on the first step after reset
        episode_id.npy      int64   [T, N]       unique within the run
        step_in_episode.npy int32   [T, N]
        end_reason.npy      int8    [T, N]       nonzero on an episode's last row
        cube_pos_w.npy      float32 [T, N, 3]
        cube_quat_w.npy     float32 [T, N, 4]
        dr_*.npy            float32 [T, N]       per-row randomisation values

Row ``t`` holds the observation returned by ``env.step`` at control step
``t`` and the ground truth read from the *same* post-step physics state
(MJLab runs ``sim.forward(); sim.sense()`` immediately before computing
observations), so inputs and labels are synchronised.

Splits are separated by collection seed: train/val/test runs never share an
episode, an environment instance, or a startup-randomisation sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

END_REASONS = {
  0: "none",
  1: "time_out",
  2: "cube_fell",
  3: "cube_too_fast",
  4: "cube_pose_deviation",
  5: "nan",
  6: "other",
}
END_REASON_IDS = {v: k for k, v in END_REASONS.items()}
FAILURE_REASONS = (2, 3, 4, 5, 6)

LABEL_FILES = ("pos", "quat", "lin_vel", "ang_vel", "contact", "contact_valid")
AUX_FILES = (
  "episode_id",
  "step_in_episode",
  "end_reason",
  "cube_pos_w",
  "cube_quat_w",
)


def run_dirs(root: str | Path, split: str) -> list[Path]:
  d = Path(root) / split
  runs = sorted(p for p in d.glob("seed_*") if (p / "meta.json").is_file())
  if not runs:
    raise FileNotFoundError(f"No collection runs under {d}")
  return runs


def read_meta(run: Path) -> dict:
  return json.loads((run / "meta.json").read_text())


def _load(run: Path, name: str, mmap: bool = True) -> np.ndarray:
  return np.load(run / f"{name}.npy", mmap_mode="r" if mmap else None)


def load_split_tensors(
  root: str | Path,
  split: str,
  device: str | torch.device = "cpu",
  max_rows: int | None = None,
  include_aux: bool = False,
  include_dr: bool = False,
  seed: int = 0,
) -> dict[str, torch.Tensor]:
  """Load a whole split as flat [rows, ...] tensors.

  ``max_rows`` subsamples whole environment columns (i.e. whole episodes are
  kept together) to bound memory without breaking trajectories.
  """
  runs = run_dirs(root, split)
  names = list(("obs",) + LABEL_FILES)
  if include_aux:
    names += list(AUX_FILES)
  out: dict[str, list[np.ndarray]] = {k: [] for k in names}
  run_index: list[np.ndarray] = []
  dr_names: list[str] = []
  if include_dr:
    dr_names = sorted(p.stem for p in runs[0].glob("dr_*.npy"))
    for k in dr_names:
      out[k] = []
  rng = np.random.default_rng(seed)
  total = sum(_load(r, "obs").shape[0] * _load(r, "obs").shape[1] for r in runs)
  keep_frac = 1.0 if max_rows is None else min(1.0, max_rows / max(total, 1))
  for ri, run in enumerate(runs):
    t, n = _load(run, "obs").shape[:2]
    cols = np.arange(n)
    if keep_frac < 1.0:
      cols = np.sort(rng.choice(n, size=max(1, int(round(n * keep_frac))), replace=False))
    for k in names:
      arr = _load(run, k)
      sub = np.ascontiguousarray(arr[:, cols])
      out[k].append(sub.reshape(t * len(cols), *sub.shape[2:]))
    for k in dr_names:
      sub = np.ascontiguousarray(_load(run, k)[:, cols])
      out[k].append(sub.reshape(-1))
    run_index.append(np.full(t * len(cols), ri, dtype=np.int16))
  tensors = {k: torch.from_numpy(np.concatenate(v)).to(device) for k, v in out.items()}
  tensors["run_index"] = torch.from_numpy(np.concatenate(run_index)).to(device)
  return tensors


def load_run_trajectories(run: Path, env_ids: list[int] | None = None) -> dict[str, np.ndarray]:
  """Load [T, n, ...] arrays for selected environments of one run."""
  names = ["obs", *LABEL_FILES, *AUX_FILES]
  names += sorted(p.stem for p in run.glob("dr_*.npy"))
  data = {}
  for k in names:
    arr = _load(run, k)
    data[k] = np.asarray(arr[:, env_ids] if env_ids is not None else arr)
  return data


def compute_normalization(train: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  obs = train["obs"].float()
  # Chunked moments to keep memory bounded on large splits.
  mean = torch.zeros(obs.shape[1], dtype=torch.float64, device=obs.device)
  sq = torch.zeros_like(mean)
  for chunk in obs.split(1_000_000):
    mean += chunk.double().sum(0)
    sq += (chunk.double() ** 2).sum(0)
  n = obs.shape[0]
  mean /= n
  std = (sq / n - mean**2).clamp_min(0).sqrt()

  def rms(x: torch.Tensor) -> torch.Tensor:
    return x.double().pow(2).mean(0).sqrt().float()

  return {
    "obs_mean": mean.float(),
    "obs_std": std.float(),
    "pos_mean": train["pos"].double().mean(0).float(),
    "pos_std": train["pos"].double().std(0).float(),
    "lin_vel_std": rms(train["lin_vel"]),
    "ang_vel_std": rms(train["ang_vel"]),
  }
