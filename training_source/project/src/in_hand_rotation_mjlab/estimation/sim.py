"""MJLab-side helpers: environment construction, policy loading, and
privileged ground-truth extraction.  Requires the full MJLab/MuJoCo-Warp stack.

Ground truth is taken from the *same* observation-manager call that produced
the actor observation: the privileged critic group already computes
``cube_pose_palm``, ``cube_lin_vel_palm`` and ``cube_ang_vel_palm`` with the
task's own functions (``tasks/hand_cube/mdp/observations.py``).  Slices are
resolved by term name at runtime, never hard-coded.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

TASK_ID = "Mjlab-Leap-Right-HandCube-MBZUAI-Rotate"
CONTACT_SENSOR = "fingertip_cube_contact"

TERMINATION_CODES = {
  "time_out": 1,
  "cube_fell": 2,
  "cube_too_fast": 3,
  "cube_pose_deviation": 4,
  "nan": 5,
}


def _import_tasks() -> None:
  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401


def prepare_agent_cfg(agent_cfg) -> dict:
  cfg = asdict(agent_cfg) if not isinstance(agent_cfg, dict) else copy.deepcopy(agent_cfg)
  for key in ("actor", "critic"):
    if cfg[key].get("class_name", "MLPModel") != "CNNModel":
      cfg[key].pop("cnn_cfg", None)
  return cfg


def make_env(
  num_envs: int,
  seed: int,
  device: str,
  *,
  task: str = TASK_ID,
  play: bool = False,
  episode_length_s: float | None = None,
  disable_curriculum: bool = True,
  obs_corruption: bool | None = None,
  dr_overrides: dict[str, Any] | None = None,
  fix_reset_anchor: bool = False,
  termination_variant: str | None = None,
):
  """Build the original task environment (domain randomisation untouched).

  ``play=False`` keeps the *training* configuration (observation noise and
  delays, 20 s episodes).  ``disable_curriculum`` freezes the penalty weights
  at their configured values so that rewards are comparable across policies.
  ``dr_overrides`` maps event names to replacement ``params`` entries for
  stress tests (e.g. ``{"dr_cube_mass": {"mass_range": (0.5, 2.0)}}``).
  """
  _import_tasks()
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  env_cfg = load_env_cfg(task, play=play)
  env_cfg.scene.num_envs = int(num_envs)
  env_cfg.seed = int(seed)
  if episode_length_s is not None:
    env_cfg.episode_length_s = float(episode_length_s)
  if disable_curriculum:
    env_cfg.curriculum = {}
  if obs_corruption is not None:
    env_cfg.observations["actor"].enable_corruption = bool(obs_corruption)
  for event_name, params in (dr_overrides or {}).items():
    env_cfg.events[event_name].params.update(params)
  if termination_variant is not None:
    from .env_fixes import apply_termination_variant

    apply_termination_variant(env_cfg, termination_variant)
  elif fix_reset_anchor:
    from .env_fixes import apply_reset_anchor_fix

    apply_reset_anchor_fix(env_cfg)
  agent_cfg = load_rl_cfg(task)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  return env, raw_env, agent_cfg


def load_policy(env, agent_cfg, checkpoint: str | Path, device: str):
  """Load ``model_400.pt`` or an estimator-augmented checkpoint.

  Augmented checkpoints carry an ``estimation`` block with the actor class and
  kwargs, so the matching actor is built before strict state-dict loading.
  Returns (runner, actor_model).
  """
  from rsl_rl.runners import OnPolicyRunner

  checkpoint = Path(checkpoint).expanduser().resolve()
  blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
  cfg = prepare_agent_cfg(agent_cfg)
  from .migration import infer_actor_cfg

  est = blob.get("estimation")
  if est is not None:
    cfg["actor"].update(est["actor_cfg"])
  else:
    inferred = infer_actor_cfg(blob["actor_state_dict"])
    if inferred is not None:
      cfg["actor"].update(inferred)
  for k in ("estimator_path", "estimator_config", "feature_mode", "freeze_estimator"):
    if k in cfg["actor"] and cfg["actor"].get("class_name", "MLPModel") == "MLPModel":
      cfg["actor"].pop(k)
  runner = OnPolicyRunner(env, cfg, log_dir=None, device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True, "critic": True},
    strict=True,
    map_location=device,
  )
  actor = runner.get_inference_policy(device=device)
  return runner, actor


class GroundTruth:
  """Reads synchronised privileged labels after every ``env.step``."""

  def __init__(self, raw_env):
    self.env = raw_env
    om = raw_env.observation_manager
    names = om.active_terms["critic"]
    dims = om.group_obs_term_dim["critic"]
    self.critic_slices: dict[str, slice] = {}
    start = 0
    for name, dim in zip(names, dims):
      width = int(torch.tensor(dim).prod())
      self.critic_slices[name] = slice(start, start + width)
      start += width
    for req in ("cube_pose_palm", "cube_lin_vel_palm", "cube_ang_vel_palm"):
      if req not in self.critic_slices:
        raise RuntimeError(f"Critic group lacks {req}; terms: {names}")
    self.actor_terms = list(om.active_terms["actor"])
    self.actor_term_dims = [tuple(d) for d in om.group_obs_term_dim["actor"]]

    self.cube = raw_env.scene["cube"]
    self.robot = raw_env.scene["robot"]
    self.sensor = raw_env.scene[CONTACT_SENSOR]
    self.contact_names = self._contact_primary_names()
    self.term_names = list(raw_env.termination_manager.active_terms)

    # Cached ids for domain-randomisation read-out.
    self._cube_body = int(self.cube.indexing.body_ids[0].item())
    geom_local, _ = self.cube.find_geoms("cube_geom", preserve_order=True)
    self._cube_geom = int(self.cube.indexing.geom_ids[geom_local[0]].item())
    self._robot_dofs = self.robot.indexing.joint_v_adr.long()
    self._robot_act = getattr(self.robot.indexing, "ctrl_ids", None)
    self._nominal_frictionloss = None
    self._nominal_damping = None

  def _contact_primary_names(self) -> list[str]:
    for attr in ("primary_names", "_primary_names"):
      names = getattr(self.sensor, attr, None)
      if names:
        return list(names)
    cfg = self.sensor.cfg
    robot = self.env.scene[cfg.primary.entity]
    _, names = robot.find_geoms(cfg.primary.pattern, preserve_order=True)
    return list(names)

  def labels(self, critic_obs: torch.Tensor) -> dict[str, torch.Tensor]:
    pose = critic_obs[:, self.critic_slices["cube_pose_palm"]]
    quat = pose[:, 3:7]
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    found = self.sensor.data.found
    if found is None:
      raise RuntimeError("Contact sensor has no 'found' field.")
    return {
      "pos": pose[:, :3],
      "quat": quat,
      "lin_vel": critic_obs[:, self.critic_slices["cube_lin_vel_palm"]],
      "ang_vel": critic_obs[:, self.critic_slices["cube_ang_vel_palm"]],
      "contact": (found > 0).to(torch.uint8),
    }

  def world_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    return self.cube.data.root_link_pos_w, self.cube.data.root_link_quat_w

  def end_reasons(self, dones: torch.Tensor) -> torch.Tensor:
    """Code for each env that ended this step (0 when not done).

    Failure terms take precedence over ``time_out``.
    """
    tm = self.env.termination_manager
    code = torch.zeros(dones.shape[0], dtype=torch.int8, device=dones.device)
    ordered = ["time_out", "nan", "cube_too_fast", "cube_pose_deviation", "cube_fell"]
    for name in ordered:
      if name in self.term_names:
        hit = tm.get_term(name).bool()
        code = torch.where(hit, torch.full_like(code, TERMINATION_CODES[name]), code)
    other = dones.bool() & (code == 0)
    code = torch.where(other, torch.full_like(code, 6), code)
    return torch.where(dones.bool(), code, torch.zeros_like(code))

  def randomization(self) -> dict[str, torch.Tensor]:
    """Per-env physical parameters currently in effect (float32 [N])."""
    m = self.env.sim.model
    out = {
      "dr_cube_mass": m.body_mass[:, self._cube_body],
      "dr_friction": m.geom_friction[:, self._cube_geom, 0],
      "dr_com_offset_norm": m.body_ipos[:, self._cube_body].norm(dim=-1),
      "dr_motor_frictionloss": m.dof_frictionloss[:, self._robot_dofs].mean(-1),
      "dr_motor_damping": m.dof_damping[:, self._robot_dofs].mean(-1),
    }
    fr = getattr(m, "actuator_forcerange", None)
    if fr is not None and fr.ndim == 3:
      out["dr_effort_limit"] = fr[:, :, 1].mean(-1)
    lag = self._actuator_lag()
    if lag is not None:
      out["dr_action_delay_steps"] = lag.float()
    return {k: v.float() for k, v in out.items()}

  def _actuator_lag(self) -> torch.Tensor | None:
    for act in getattr(self.robot, "actuators", []) or []:
      bufs = getattr(act, "_delay_buffers", None)
      if bufs:
        buf = next(iter(bufs.values()))
        lag = getattr(buf, "_current_lags", None)
        if lag is not None:
          return lag.reshape(lag.shape[0], -1)[:, 0]
    return None
