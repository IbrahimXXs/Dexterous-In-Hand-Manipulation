"""PPO configuration shared by all from-scratch study runs.

Starts from the original right-hand PPO configuration
(``leap_right/rl_cfg.py``: 512-512-256 ELU actor and critic, 32 steps/env,
5 epochs x 4 minibatches, gamma 0.99, lambda 0.95, clip 0.2, entropy 0.003,
init noise std 0.7).  Study-wide changes (identical for A, C, D):

* algorithm class ``InstrumentedPPO`` (same optimisation, extra logging);
* ``clip_actions=None`` when the environment squashes with tanh (the env
  receives the Gaussian sample and applies tanh itself);
* learning-rate schedule chosen by the Stage-0 pilot (see the README).

Only the actor class differs between conditions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right.rl_cfg import (
  leap_right_hand_cube_rotate_ppo_cfg,
)

_PROJECT = Path(__file__).resolve().parents[6]
EXPERIMENT_DIR = _PROJECT / "experiments" / "scratch_state_estimation"
DEFAULT_ESTIMATOR = os.environ.get(
  "LEAP_SCRATCH_ESTIMATOR", str(EXPERIMENT_DIR / "models" / "estimator_B.pt")
)

# Frozen study settings (decided by the Stage-0 pilot, 2026-11-12; see README):
# the original adaptive-KL schedule (initial 1e-3, desired KL 0.01) behaves as
# intended when training from scratch (LR 5e-4..1.5e-3, KL ~0.017); the 1e-5
# floor seen before was specific to fine-tuning model_400 (raw outputs ~10-30).
STUDY_SCHEDULE = "adaptive"
STUDY_LR = 1e-3
# Budget: 1001 PPO updates (iterations 0..1000; final checkpoint model_1000.pt)
# = 131.2 M environment steps per run.  Chosen at launch from the measured
# throughput (3 shared GPUs, ~24k steps/s each) so that 9 runs + estimator +
# evaluation fit one overnight session.  Phase-1 runs (A, D) were started
# with a 1500 cap and stopped right after model_1000.pt (the adaptive LR does
# not depend on the cap, so this is equivalent).
STUDY_ITERATIONS = 1001


@dataclass
class InstrumentedPpoCfg(RslRlPpoAlgorithmCfg):
  action_transform: str = "tanh"
  class_name: str = "in_hand_rotation_mjlab.scratch_study.ppo:InstrumentedPPO"


@dataclass
class EstimatedStateActorCfg(RslRlModelCfg):
  estimator_path: str = DEFAULT_ESTIMATOR
  class_name: str = "in_hand_rotation_mjlab.scratch_study.actors:EstimatedStateMLPModel"


@dataclass
class OracleStateActorCfg(RslRlModelCfg):
  class_name: str = "in_hand_rotation_mjlab.scratch_study.actors:OracleStateMLPModel"


def scratch_ppo_cfg(condition: str, squash: str = "tanh") -> RslRlOnPolicyRunnerCfg:
  cfg = leap_right_hand_cube_rotate_ppo_cfg()
  base, alg = cfg.actor, cfg.algorithm
  cfg.experiment_name = f"scratch_{condition}" + ("" if squash == "tanh" else f"_{squash}action")
  cfg.run_name = condition
  cfg.max_iterations = STUDY_ITERATIONS
  cfg.save_interval = 50
  cfg.logger = "tensorboard"
  cfg.obs_groups = {"actor": ("actor",), "critic": ("critic",)}
  cfg.clip_actions = None if squash == "tanh" else 1.0
  cfg.algorithm = InstrumentedPpoCfg(
    num_learning_epochs=alg.num_learning_epochs,
    num_mini_batches=alg.num_mini_batches,
    learning_rate=STUDY_LR,
    schedule=STUDY_SCHEDULE,
    gamma=alg.gamma,
    lam=alg.lam,
    entropy_coef=alg.entropy_coef,
    desired_kl=alg.desired_kl,
    max_grad_norm=alg.max_grad_norm,
    value_loss_coef=alg.value_loss_coef,
    use_clipped_value_loss=alg.use_clipped_value_loss,
    clip_param=alg.clip_param,
    action_transform=squash,
  )
  common = dict(
    hidden_dims=base.hidden_dims,
    activation=base.activation,
    obs_normalization=base.obs_normalization,
    init_noise_std=base.init_noise_std,
    noise_std_type=base.noise_std_type,
    stochastic=base.stochastic,
  )
  if condition == "C":
    cfg.actor = EstimatedStateActorCfg(**common)
  elif condition == "D":
    cfg.actor = OracleStateActorCfg(**common)
  elif condition != "A":
    raise ValueError(condition)
  return cfg
