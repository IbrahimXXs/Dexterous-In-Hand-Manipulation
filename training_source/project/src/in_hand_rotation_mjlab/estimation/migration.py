"""Checkpoint migration: model_400.pt (320-input actor) -> augmented actor.

Every tensor is accounted for explicitly; nothing is loaded "where shapes
happen to match":

=====================  ====================================================
target key             source
=====================  ====================================================
obs_normalizer.*       copied (still 320-D: features are not normalised here)
mlp.0.weight [H,320+F] columns 0..319 copied, columns 320.. set to **zero**
mlp.0.bias, mlp.2/4/6  copied unchanged
std                    copied (exploration level of the converged policy)
estimator.*            copied from the trained estimator checkpoint
critic_state_dict      copied unchanged (critic input is still 83-D)
optimizer (Adam)       moments re-indexed by parameter name; the widened
                       first layer's moments are zero-padded; frozen
                       estimator parameters get no state
iter                   copied (training continues from iteration 400)
=====================  ====================================================

Because the new columns are zero, the migrated policy's action mean is
*identical* to model_400's at iteration 0 (checked in the unit tests); PPO
then learns how much to use the estimated state.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from .layout import ACTOR_OBS_DIM, FEATURE_MODES
from .model import load_estimator

NORMALIZER_PREFIX = "obs_normalizer."
FIRST_LAYER = "mlp.0.weight"
ACTOR_CLASS = "in_hand_rotation_mjlab.estimation.policy:EstimatorAugmentedMLPModel"


def _param_names(state_dict: dict[str, torch.Tensor]) -> list[str]:
  """Parameter names (in ``named_parameters`` order) of an MLPModel state dict."""
  return [k for k in state_dict if not k.startswith(NORMALIZER_PREFIX)]


def migrate_actor_state(
  src_actor: dict[str, torch.Tensor],
  estimator_state: dict[str, torch.Tensor],
  feature_dim: int,
) -> tuple[OrderedDict, list[str]]:
  report: list[str] = []
  w0 = src_actor[FIRST_LAYER]
  if w0.shape[1] != ACTOR_OBS_DIM:
    raise ValueError(f"Source actor first layer has {w0.shape[1]} inputs, expected {ACTOR_OBS_DIM}")
  out: OrderedDict[str, torch.Tensor] = OrderedDict()
  for k, v in src_actor.items():
    if k == FIRST_LAYER:
      new = torch.zeros(w0.shape[0], w0.shape[1] + feature_dim, dtype=w0.dtype)
      new[:, : w0.shape[1]] = w0
      out[k] = new
      report.append(f"{k}: {tuple(w0.shape)} -> {tuple(new.shape)} (new columns zero)")
    else:
      out[k] = v.clone()
      report.append(f"{k}: copied {tuple(v.shape)}")
  for k, v in estimator_state.items():
    out[f"estimator.{k}"] = v.clone()
  report.append(f"estimator.*: {len(estimator_state)} tensors from estimator checkpoint")
  return out, report


def migrate_optimizer_state(
  src_opt: dict,
  src_actor: dict[str, torch.Tensor],
  src_critic: dict[str, torch.Tensor],
  new_actor_param_names: list[str],
  new_critic_param_names: list[str],
  feature_dim: int,
) -> dict:
  """Re-index Adam state from the old parameter list to the new one."""
  old_names = [f"actor.{n}" for n in _param_names(src_actor)] + [
    f"critic.{n}" for n in _param_names(src_critic)
  ]
  group = src_opt["param_groups"]
  if len(group) != 1 or len(group[0]["params"]) != len(old_names):
    raise ValueError(
      f"Optimizer has {[len(g['params']) for g in group]} params, expected {len(old_names)}"
    )
  old_state = {old_names[i]: src_opt["state"][pid] for i, pid in enumerate(group[0]["params"]) if pid in src_opt["state"]}
  new_names = [f"actor.{n}" for n in new_actor_param_names] + [
    f"critic.{n}" for n in new_critic_param_names
  ]
  new_state: dict[int, dict[str, Any]] = {}
  for i, name in enumerate(new_names):
    st = old_state.get(name)
    if st is None:
      continue  # estimator params (frozen) or genuinely new
    st = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    if name == f"actor.{FIRST_LAYER}":
      for key in ("exp_avg", "exp_avg_sq"):
        m = st[key]
        st[key] = torch.cat([m, torch.zeros(m.shape[0], feature_dim, dtype=m.dtype)], dim=1)
    new_state[i] = st
  new_group = dict(group[0])
  new_group["params"] = list(range(len(new_names)))
  return {"state": new_state, "param_groups": [new_group]}


def augmented_actor_cfg(estimator, feature_mode: str, freeze: bool = True) -> dict:
  from dataclasses import asdict

  cfg = asdict(estimator.cfg)
  cfg["hidden_dims"] = list(cfg["hidden_dims"])
  return {
    "class_name": ACTOR_CLASS,
    "estimator_path": "",
    "estimator_config": cfg,
    "feature_mode": feature_mode,
    "freeze_estimator": freeze,
  }


def infer_actor_cfg(actor_state: dict[str, torch.Tensor], activation: str = "elu") -> dict | None:
  """Recover the augmented-actor kwargs from a saved actor state dict.

  RSL-RL checkpoints written during fine-tuning do not store the model
  config, so evaluation reconstructs it from tensor shapes: the feature mode
  from the first-layer width and the estimator architecture from its Linear
  layers.  Returns None for a plain 320-input MLP actor.
  """
  if not any(k.startswith("estimator.") for k in actor_state):
    return None
  extra = actor_state[FIRST_LAYER].shape[1] - ACTOR_OBS_DIM
  modes = [m for m, d in FEATURE_MODES.items() if d == extra]
  if len(modes) != 1:
    raise ValueError(f"Cannot infer feature mode from {extra} extra actor inputs")
  linear = sorted(
    (k for k in actor_state if k.startswith("estimator.trunk.") and k.endswith(".weight")
     and actor_state[k].ndim == 2),
    key=lambda k: int(k.split(".")[2]),
  )
  hidden = [int(actor_state[k].shape[0]) for k in linear]
  layer_norm = any(
    k.startswith("estimator.trunk.") and actor_state[k].ndim == 1 and k.endswith(".weight")
    for k in actor_state
  )
  return {
    "class_name": ACTOR_CLASS,
    "estimator_path": "",
    "estimator_config": {
      "obs_dim": ACTOR_OBS_DIM,
      "hidden_dims": hidden,
      "activation": activation,
      "layer_norm": layer_norm,
      "dropout": 0.0,
      "meta": {},
    },
    "feature_mode": modes[0],
    "freeze_estimator": True,
  }


def migrate_checkpoint(
  source_checkpoint: str | Path,
  estimator_checkpoint: str | Path,
  feature_mode: str,
  output: str | Path,
) -> dict:
  """Write a full RSL-RL checkpoint for the augmented actor.

  The optimizer state is migrated only when the parameter naming can be
  verified; otherwise it is dropped (``train.py`` then starts a fresh Adam).
  """
  from .model import CubeStateEstimator  # noqa: F401  (import check)

  src = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
  est = load_estimator(estimator_checkpoint)
  fdim = FEATURE_MODES[feature_mode]
  actor_sd, report = migrate_actor_state(src["actor_state_dict"], est.state_dict(), fdim)

  # Parameter order of the augmented actor = MLPModel params, then estimator.
  new_actor_params = _param_names(src["actor_state_dict"]) + [
    f"estimator.{n}" for n, _ in est.named_parameters()
  ]
  opt = None
  if "optimizer_state_dict" in src:
    opt = migrate_optimizer_state(
      src["optimizer_state_dict"],
      src["actor_state_dict"],
      src["critic_state_dict"],
      new_actor_params,
      _param_names(src["critic_state_dict"]),
      fdim,
    )
    report.append(f"optimizer: {len(opt['state'])} Adam states migrated")
  out = {
    "actor_state_dict": actor_sd,
    "critic_state_dict": src["critic_state_dict"],
    "iter": src.get("iter", 0),
    "infos": {"migrated_from": str(Path(source_checkpoint).resolve())},
    "estimation": {
      "actor_cfg": augmented_actor_cfg(est, feature_mode),
      "estimator_checkpoint": str(Path(estimator_checkpoint).resolve()),
      "feature_mode": feature_mode,
      "feature_dim": fdim,
      "migration_report": report,
    },
  }
  if opt is not None:
    out["optimizer_state_dict"] = opt
  Path(output).parent.mkdir(parents=True, exist_ok=True)
  torch.save(out, output)
  return out
