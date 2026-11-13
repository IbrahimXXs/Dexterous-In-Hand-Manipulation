"""Load force-estimation shards (one directory per collection run) into flat row arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from in_hand_rotation_mjlab.estimation.layout import ACTOR_OBS_DIM

LABEL_KEYS = ("force_mean", "force_now", "force_max", "contact_frac", "found_now", "tangential_now")
META_KEYS = ("other_normal_now", "cube_lin_vel", "cube_ang_vel", "step_in_episode", "episode_id", "end_reason")


def shard_dirs(root: str | Path, split: str) -> list[Path]:
  d = Path(root) / split
  return sorted(p for p in d.glob("*") if (p / "meta.json").exists())


def load_shard(path: Path, *, with_obs: bool = True, mmap: bool = True, dr: bool = False) -> dict:
  """Rows with step_in_episode > 0 only (the first row of an episode has a zeroed substep history)."""
  meta = json.loads((path / "meta.json").read_text())
  sie = np.load(path / "step_in_episode.npy")
  T, N = sie.shape
  # time-major [T, N] -> keep (t, i) order; also return env index and t for sequence plots
  valid = sie > 0
  out = {"meta": meta, "valid": valid, "T": T, "N": N}
  flat = valid.reshape(-1)
  for k in LABEL_KEYS + META_KEYS:
    a = np.load(path / f"{k}.npy")
    out[k] = a.reshape(T * N, *a.shape[2:])[flat]
  if dr:
    for f in meta.get("dr_fields", []):
      out[f] = np.load(path / f"{f}.npy").reshape(-1)[flat]
  tt, ii = np.meshgrid(np.arange(T), np.arange(N), indexing="ij")
  out["t"] = tt.reshape(-1)[flat]
  out["env"] = ii.reshape(-1)[flat]
  if with_obs:
    obs = np.load(path / "obs.npy", mmap_mode="r" if mmap else None)
    assert obs.shape == (T, N, ACTOR_OBS_DIM), obs.shape
    out["obs"] = obs.reshape(T * N, ACTOR_OBS_DIM)[flat]
  return out


def load_split(root, split: str, *, with_obs: bool = True, dr: bool = False) -> dict:
  shards = shard_dirs(root, split)
  if not shards:
    raise FileNotFoundError(f"no shards in {Path(root) / split}")
  parts = [load_shard(s, with_obs=with_obs, dr=dr) for s in shards]
  out: dict = {"shards": [str(s) for s in shards], "metas": [p["meta"] for p in parts]}
  keys = [k for k in parts[0] if k not in ("meta", "valid", "T", "N")]
  for k in keys:
    if all(k in p for p in parts):
      out[k] = np.concatenate([np.asarray(p[k]) for p in parts])
  out["shard_id"] = np.concatenate([np.full(len(p["force_mean"]), i, dtype=np.int32) for i, p in enumerate(parts)])
  return out
