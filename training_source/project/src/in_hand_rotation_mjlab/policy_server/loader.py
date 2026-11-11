from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from mjlab.tasks.registry import load_runner_cls


def _prepare_agent_cfg(agent_cfg) -> dict[str, Any]:
  """Convert dataclass config to runner dict and drop incompatible MLP fields."""
  cfg_dict = asdict(agent_cfg)
  for model_key in ("actor", "critic"):
    model_cfg = cfg_dict.get(model_key)
    if isinstance(model_cfg, dict):
      if model_cfg.get("class_name", "MLPModel") != "CNNModel":
        model_cfg.pop("cnn_cfg", None)
  return cfg_dict


class _PolicyLoadVecEnv:
  """Minimal VecEnv-like object used only to construct/load the policy network."""

  def __init__(
    self,
    env_cfg: Any,
    obs: dict[str, torch.Tensor],
    num_actions: int,
    step_dt: float,
    max_episode_length: int,
    device: str,
  ):
    self.cfg = env_cfg
    self._obs = obs
    self.num_envs = 1
    self.num_actions = num_actions
    self.step_dt = step_dt
    self.max_episode_length = max_episode_length
    self.device = torch.device(device)
    self.common_step_counter = 0
    self.episode_length_buf = torch.zeros(
      (1,), device=self.device, dtype=torch.long
    )

  @property
  def unwrapped(self) -> "_PolicyLoadVecEnv":
    return self

  def get_observations(self) -> TensorDict:
    cloned = {k: v.clone() for k, v in self._obs.items()}
    return TensorDict(cloned, batch_size=[1])

  def reset(self):
    return self.get_observations(), {}

  def close(self) -> None:
    return None


def _infer_obs_dim(ckpt: dict[str, Any], key: str) -> int:
  state_dict = ckpt.get(key, {})
  if "obs_normalizer._mean" in state_dict:
    tensor = state_dict["obs_normalizer._mean"]
    return int(tensor.shape[-1])
  if "mlp.0.weight" in state_dict:
    tensor = state_dict["mlp.0.weight"]
    return int(tensor.shape[1])
  raise ValueError(f"Unable to infer observation dimension from checkpoint entry '{key}'.")


_INFERENCE_NOISE_STATE_KEYS = frozenset({"std", "log_std"})


def _is_inference_noise_state_key(key: str) -> bool:
  return key.rsplit(".", 1)[-1] in _INFERENCE_NOISE_STATE_KEYS


def _actor_state_for_deterministic_inference(
  source: dict[str, torch.Tensor],
  target: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
  """Merge actor mean state while ignoring stochastic noise parameter format.

  ``std`` and ``log_std`` affect sampled training actions only. Hardware
  inference uses the deterministic actor mean, so those keys may differ while
  every mean-network and observation-normalizer tensor remains strict.
  """
  source_mean_keys = {
    key for key in source if not _is_inference_noise_state_key(key)
  }
  target_mean_keys = {
    key for key in target if not _is_inference_noise_state_key(key)
  }
  missing = sorted(target_mean_keys - source_mean_keys)
  unexpected = sorted(source_mean_keys - target_mean_keys)
  shape_mismatch = sorted(
    key
    for key in source_mean_keys & target_mean_keys
    if source[key].shape != target[key].shape
  )
  if missing or unexpected or shape_mismatch:
    raise RuntimeError(
      "Incompatible deterministic actor state: "
      f"missing={missing}, unexpected={unexpected}, "
      f"shape_mismatch={shape_mismatch}"
    )

  merged = dict(target)
  for key in target_mean_keys:
    merged[key] = source[key].to(
      device=target[key].device, dtype=target[key].dtype
    )
  return merged


def load_inference_policy(
  *,
  task_id: str,
  checkpoint_file: str,
  env_cfg: Any,
  agent_cfg: Any,
  num_actions: int,
  step_dt: float,
  max_episode_length: int,
  device: str,
) -> tuple[Callable[[Any], torch.Tensor], int]:
  """Load RSL-RL inference policy and checkpoint actor-observation dimension."""

  checkpoint = Path(checkpoint_file)
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

  ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
  actor_obs_dim = _infer_obs_dim(ckpt, "actor_state_dict")
  critic_obs_dim = _infer_obs_dim(ckpt, "critic_state_dict")

  obs_groups = {
    "actor": torch.zeros((1, actor_obs_dim), device=device),
    "critic": torch.zeros((1, critic_obs_dim), device=device),
  }

  loader_env = _PolicyLoadVecEnv(
    env_cfg=env_cfg,
    obs=obs_groups,
    num_actions=num_actions,
    step_dt=step_dt,
    max_episode_length=max_episode_length,
    device=device,
  )
  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner = runner_cls(loader_env, _prepare_agent_cfg(agent_cfg), device=device)
  actor = runner.alg.get_policy()
  try:
    source_actor_state = ckpt.get("actor_state_dict")
    if not isinstance(source_actor_state, dict):
      raise RuntimeError("Checkpoint has no actor_state_dict")
    actor.load_state_dict(
      _actor_state_for_deterministic_inference(
        source_actor_state, actor.state_dict()
      ),
      strict=True,
    )
  except RuntimeError as exc:
    raise RuntimeError(
      "Failed to load checkpoint actor mean into policy network. "
      "This usually means checkpoint/task config mismatch (e.g., different "
      "observation or action dimensions). "
      f"Checkpoint: {checkpoint}"
    ) from exc
  policy = runner.get_inference_policy(device=device)
  policy.eval()
  return policy, actor_obs_dim

