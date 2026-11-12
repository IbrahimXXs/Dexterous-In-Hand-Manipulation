"""PPO fine-tuning configs for the estimator study (same hyper-parameters as
the original ``leap_right_mbzuai_rotate`` fine-tuning run)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from in_hand_rotation_mjlab.estimation.migration import ACTOR_CLASS
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_mbzuai_rotate.rl_cfg import (
  leap_right_mbzuai_rotate_ppo_cfg,
)

_PROJECT = Path(__file__).resolve().parents[6]
DEFAULT_ESTIMATOR = os.environ.get(
  "LEAP_CUBE_ESTIMATOR",
  str(_PROJECT / "data" / "estimator_models" / "cube_state_estimator.pt"),
)


@dataclass
class EstimatorActorCfg(RslRlModelCfg):
  """Actor config forwarding the estimator options to the actor class."""

  estimator_path: str = DEFAULT_ESTIMATOR
  estimator_config: dict[str, Any] | None = None
  feature_mode: str = "full"
  freeze_estimator: bool = True
  class_name: str = ACTOR_CLASS


def estimator_ppo_cfg(feature_mode: str | None) -> RslRlOnPolicyRunnerCfg:
  cfg = leap_right_mbzuai_rotate_ppo_cfg()
  cfg.max_iterations = 300
  cfg.save_interval = 50
  base = cfg.actor
  if feature_mode is None:
    cfg.experiment_name = "leap_right_mbzuai_rotate_fixedreset"
    cfg.run_name = "control_finetune"
    return cfg
  cfg.experiment_name = f"leap_right_mbzuai_rotate_estimator_{feature_mode}"
  cfg.run_name = f"estimator_{feature_mode}_finetune"
  cfg.actor = EstimatorActorCfg(
    hidden_dims=base.hidden_dims,
    activation=base.activation,
    obs_normalization=base.obs_normalization,
    init_noise_std=base.init_noise_std,
    noise_std_type=base.noise_std_type,
    stochastic=base.stochastic,
    feature_mode=feature_mode,
  )
  return cfg
