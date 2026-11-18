"""PPO configuration of the hardware-limit A/C study.

Exactly the from-scratch study configuration (``scratch_ppo_cfg``: actor and
critic 512-512-256 ELU, InstrumentedPPO, adaptive-KL schedule from 1e-3,
4096 envs set at launch, zero-initialised actor output, frozen estimator B
for C), with two changes for both conditions:

* budget: ``HW_ITERATIONS`` = 3001 PPO updates (iterations 0..3000, final
  checkpoint ``model_3000.pt``) instead of 1001;
* experiment name ``hwlim_<cond>`` (separate log directory).

Condition C reads estimator B from its frozen study location (read-only);
the path is fixed here, not taken from an environment variable.
"""

from __future__ import annotations

from pathlib import Path

from mjlab.rl import RslRlOnPolicyRunnerCfg

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_scratch_study.rl_cfg import (
  EstimatedStateActorCfg,
  scratch_ppo_cfg,
)

_PROJECT = Path(__file__).resolve().parents[6]
ESTIMATOR_B = _PROJECT / "experiments" / "scratch_state_estimation" / "models" / "estimator_B.pt"
HW_ITERATIONS = 3001
SAVE_INTERVAL = 50


def hwlimit_ppo_cfg(condition: str) -> RslRlOnPolicyRunnerCfg:
  if condition not in ("A", "C"):
    raise ValueError(f"the hardware-limit study has conditions A and C only, got {condition!r}")
  cfg = scratch_ppo_cfg(condition)
  cfg.experiment_name = f"hwlim_{condition}"
  cfg.max_iterations = HW_ITERATIONS
  cfg.save_interval = SAVE_INTERVAL
  if condition == "C":
    a = cfg.actor
    assert isinstance(a, EstimatedStateActorCfg)
    a.estimator_path = str(ESTIMATOR_B)
  return cfg
