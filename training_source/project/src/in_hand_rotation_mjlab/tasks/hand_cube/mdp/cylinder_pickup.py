"""Autonomous table pickup lifecycle for the measured Task 1 cylinder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_from_euler_xyz,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


OPEN_PHASE = 0
PICKUP_PHASE = 1
MANIPULATE_PHASE = 2
CYLINDER_PICKUP_PHASES = ("open", "pickup", "manipulate")
NUM_PICKUP_PHASES = len(CYLINDER_PICKUP_PHASES)


def pickup_phase_id(value: str | int) -> int:
  if isinstance(value, str):
    try:
      return CYLINDER_PICKUP_PHASES.index(value)
    except ValueError as exc:
      raise ValueError(
        f"Unknown cylinder pickup phase {value!r}; expected one of "
        f"{CYLINDER_PICKUP_PHASES}."
      ) from exc
  phase = int(value)
  if phase < 0 or phase >= NUM_PICKUP_PHASES:
    raise ValueError(
      f"Cylinder pickup phase must be in [0, {NUM_PICKUP_PHASES}), got {phase}."
    )
  return phase


def pickup_phase_one_hot(value: str | int) -> torch.Tensor:
  """Encode the externally supervised hardware pickup phase exactly."""
  phase = torch.tensor(pickup_phase_id(value), dtype=torch.long)
  return torch.nn.functional.one_hot(
    phase, num_classes=NUM_PICKUP_PHASES
  ).float()


def pickup_progress_observation(
  env: "ManagerBasedRlEnv", command_name: str, total_pickup_steps: int
) -> torch.Tensor:
  """Normalized phase clock shared by simulation and the hardware supervisor."""
  if total_pickup_steps <= 0:
    raise ValueError("total_pickup_steps must be positive")
  command = env.command_manager.get_term(command_name)
  progress = torch.clamp(
    command.pickup_elapsed_steps.float() / float(total_pickup_steps),
    min=0.0,
    max=1.0,
  )
  progress = torch.where(
    command.phase == OPEN_PHASE, torch.zeros_like(progress), progress
  )
  progress = torch.where(
    command.phase == MANIPULATE_PHASE, torch.ones_like(progress), progress
  )
  return progress.unsqueeze(-1)


def _env_ids_tensor(
  env: "ManagerBasedRlEnv", env_ids: torch.Tensor | slice | None
) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if isinstance(env_ids, slice):
    start, stop, step = env_ids.indices(env.num_envs)
    return torch.arange(start, stop, step, device=env.device, dtype=torch.long)
  return env_ids.to(device=env.device, dtype=torch.long)


def _geom_world_id(entity: Entity, entity_cfg: SceneEntityCfg) -> int:
  geom_ids = entity_cfg.geom_ids
  if isinstance(geom_ids, slice):
    local_ids = list(range(entity.num_geoms))[geom_ids]
  elif isinstance(geom_ids, torch.Tensor):
    local_ids = [int(value) for value in geom_ids.tolist()]
  else:
    local_ids = [int(value) for value in geom_ids]
  if len(local_ids) != 1:
    raise ValueError(f"Expected one cylinder geom, got {local_ids}.")
  return int(entity.indexing.geom_ids[local_ids[0]].item())


def cylinder_bottom_clearance(
  env: "ManagerBasedRlEnv",
  table_height: float = 0.0,
  cylinder_cfg: SceneEntityCfg = SceneEntityCfg(
    "cube", geom_names=["cube_geom"]
  ),
) -> torch.Tensor:
  """Lowest point of an oriented capped cylinder above the table plane."""
  cylinder: Entity = env.scene[cylinder_cfg.name]
  geom_world_id = _geom_world_id(cylinder, cylinder_cfg)
  geom_size = env.sim.model.geom_size[:, geom_world_id]
  radius = geom_size[:, 0]
  half_length = geom_size[:, 1]
  rotation = matrix_from_quat(cylinder.data.root_link_quat_w)
  world_z_in_local = rotation[:, 2, :]
  axial_extent = torch.abs(world_z_in_local[:, 2]) * half_length
  radial_extent = torch.linalg.vector_norm(
    world_z_in_local[:, :2], dim=-1
  ) * radius
  lowest = cylinder.data.root_link_pos_w[:, 2] - axial_extent - radial_extent
  return lowest - table_height


@requires_model_fields("geom_size")
def reset_cylinder_on_table(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | slice | None,
  nominal_spawn_xy: tuple[float, float],
  xy_jitter: tuple[float, float] = (-0.01, 0.01),
  yaw_range: tuple[float, float] = (-3.14159, 3.14159),
  table_height: float = 0.0,
  spawn_clearance: float = 0.0,
  scale_list: tuple[float, ...] = (1.0,),
  scale_jitter: float = 0.0,
  cylinder_cfg: SceneEntityCfg = SceneEntityCfg(
    "cube", geom_names=["cube_geom"]
  ),
) -> None:
  """Place an upright cylinder on the real table collision plane."""
  resolved_ids = _env_ids_tensor(env, env_ids)
  if resolved_ids.numel() == 0:
    return
  if not scale_list:
    raise ValueError("scale_list cannot be empty")

  cylinder: Entity = env.scene[cylinder_cfg.name]
  geom_world_id = _geom_world_id(cylinder, cylinder_cfg)
  default_size = env.sim.get_default_field("geom_size")[geom_world_id].clone()
  scales = torch.tensor(scale_list, device=env.device, dtype=torch.float32)
  selected_scale = scales[resolved_ids % len(scales)]
  if scale_jitter > 0.0:
    selected_scale += sample_uniform(
      -scale_jitter,
      scale_jitter,
      (resolved_ids.numel(),),
      device=env.device,
    )
  selected_scale = torch.clamp(selected_scale, min=0.5)
  size = default_size.unsqueeze(0) * selected_scale.unsqueeze(-1)
  env.sim.model.geom_size[resolved_ids, geom_world_id] = size

  count = resolved_ids.numel()
  jitter_x = sample_uniform(*xy_jitter, count, device=env.device)
  jitter_y = sample_uniform(*xy_jitter, count, device=env.device)
  yaw = sample_uniform(*yaw_range, count, device=env.device)
  zero = torch.zeros_like(yaw)
  pose_w = torch.zeros((count, 7), device=env.device)
  pose_w[:, :3] = env.scene.env_origins[resolved_ids]
  pose_w[:, 0] += nominal_spawn_xy[0] + jitter_x
  pose_w[:, 1] += nominal_spawn_xy[1] + jitter_y
  pose_w[:, 2] += table_height + size[:, 1] + max(0.0, spawn_clearance)
  pose_w[:, 3:7] = quat_from_euler_xyz(zero, zero, yaw)
  cylinder.write_root_link_pose_to_sim(pose_w, env_ids=resolved_ids)
  cylinder.write_root_link_velocity_to_sim(
    torch.zeros((count, 6), device=env.device), env_ids=resolved_ids
  )


class CylinderTablePickupCommand(CommandTerm):
  """Gate translation until the hand genuinely lifts the cylinder."""

  cfg: "CylinderTablePickupCommandCfg"

  def __init__(self, cfg: "CylinderTablePickupCommandCfg", env):
    super().__init__(cfg, env)
    self._env = env
    self.object: Entity = env.scene[cfg.entity_name]
    self.hand: Entity = env.scene[cfg.hand_name]
    sensor = env.scene[cfg.sensor_name]
    if not isinstance(sensor, ContactSensor):
      raise TypeError(f"{cfg.sensor_name!r} is not a ContactSensor")
    self.sensor = sensor
    if len(cfg.open_joint_pos) != self.hand.num_joints:
      raise ValueError(
        f"open_joint_pos has {len(cfg.open_joint_pos)} values; expected "
        f"{self.hand.num_joints}."
      )
    for name in (
      "open_dwell_steps",
      "acquire_dwell_steps",
      "minimum_pickup_steps",
      "table_dwell_steps",
    ):
      if int(getattr(cfg, name)) <= 0:
        raise ValueError(f"{name} must be positive")
    if cfg.target_clearance_m <= cfg.table_recovery_clearance_m:
      raise ValueError(
        "target_clearance_m must exceed table_recovery_clearance_m"
      )

    self._phase = torch.full(
      (self.num_envs,), OPEN_PHASE, device=self.device, dtype=torch.long
    )
    self._open_dwell = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )
    self._acquire_dwell = torch.zeros_like(self._open_dwell)
    self._table_dwell = torch.zeros_like(self._open_dwell)
    self._pickup_elapsed_steps = torch.zeros_like(self._open_dwell)
    self._completed_pickups = torch.zeros_like(self._open_dwell)
    self._recovery_requests = torch.zeros_like(self._open_dwell)
    self._replacement_pending = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._open_joint_target = torch.tensor(
      cfg.open_joint_pos, device=self.device, dtype=torch.float32
    ).expand(self.num_envs, -1).clone()
    self.metrics["phase"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["acquired"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["completed_pickups"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["recovery_requests"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.nn.functional.one_hot(
      self._phase, num_classes=NUM_PICKUP_PHASES
    ).float()

  @property
  def phase(self) -> torch.Tensor:
    return self._phase

  @property
  def open_joint_target(self) -> torch.Tensor:
    return self._open_joint_target

  @property
  def completed_pickups(self) -> torch.Tensor:
    return self._completed_pickups

  @property
  def recovery_requests(self) -> torch.Tensor:
    return self._recovery_requests

  @property
  def pickup_elapsed_steps(self) -> torch.Tensor:
    """Number of policy steps elapsed in the current pickup attempt."""
    return self._pickup_elapsed_steps

  def _contact_count(self) -> torch.Tensor:
    found = self.sensor.data.found
    if found is None:
      return torch.zeros(self.num_envs, device=self.device)
    return (found > 0).float().sum(dim=-1)

  def _translation(self):
    name = self.cfg.translation_command_name
    if name is None or name not in self._env.command_manager.active_terms:
      return None
    return self._env.command_manager.get_term(name)

  def _hold_translation(self, env_ids: torch.Tensor) -> None:
    translation = self._translation()
    if translation is not None:
      translation.hold_at_current(env_ids)

  def _restart_translation(self, env_ids: torch.Tensor) -> None:
    translation = self._translation()
    if translation is not None:
      translation.restart_from_current(env_ids)

  def _set_phase(self, env_ids: torch.Tensor, phase: int) -> None:
    if len(env_ids) == 0:
      return
    self._phase[env_ids] = phase
    self._open_dwell[env_ids] = 0
    self._acquire_dwell[env_ids] = 0
    self._table_dwell[env_ids] = 0
    self._pickup_elapsed_steps[env_ids] = 0

  def request_table_pickup(
    self, env_ids: torch.Tensor, *, count_recovery: bool = True
  ) -> None:
    env_ids = env_ids.to(device=self.device, dtype=torch.long)
    if len(env_ids) == 0:
      return
    if count_recovery:
      self._recovery_requests[env_ids] += 1
    self._replacement_pending[env_ids] = False
    self._set_phase(env_ids, OPEN_PHASE)
    self._hold_translation(env_ids)

  def request_external_table_pickup(self, env_ids: torch.Tensor) -> None:
    """Open first, then replace an externally removed object on the table."""
    env_ids = env_ids.to(device=self.device, dtype=torch.long)
    self.request_table_pickup(env_ids)
    self._replacement_pending[env_ids] = True

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self._phase[env_ids] = OPEN_PHASE
    self._open_dwell[env_ids] = 0
    self._acquire_dwell[env_ids] = 0
    self._table_dwell[env_ids] = 0
    self._pickup_elapsed_steps[env_ids] = 0
    self._completed_pickups[env_ids] = 0
    self._recovery_requests[env_ids] = 0
    self._replacement_pending[env_ids] = False
    self._hold_translation(env_ids)
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def _update_metrics(self) -> None:
    self.metrics["phase"][:] = self._phase.float()
    self.metrics["acquired"][:] = (self._phase == MANIPULATE_PHASE).float()
    self.metrics["completed_pickups"][:] = self._completed_pickups.float()
    self.metrics["recovery_requests"][:] = self._recovery_requests.float()

  def _update_command(self) -> None:
    not_acquired = self._phase != MANIPULATE_PHASE
    self._hold_translation(not_acquired.nonzero(as_tuple=False).squeeze(-1))
    q = self.hand.data.joint_pos
    contact_count = self._contact_count()
    clearance = cylinder_bottom_clearance(
      self._env,
      table_height=self.cfg.table_height_m,
      cylinder_cfg=SceneEntityCfg(
        self.cfg.entity_name, geom_names=(self.cfg.geom_name,)
      ),
    )

    open_ids = (self._phase == OPEN_PHASE).nonzero(as_tuple=False).squeeze(-1)
    if len(open_ids) > 0:
      error = torch.max(
        torch.abs(q[open_ids] - self._open_joint_target[open_ids]), dim=-1
      ).values
      open_ok = error <= self.cfg.open_joint_tolerance_rad
      self._open_dwell[open_ids] = torch.where(
        open_ok,
        self._open_dwell[open_ids] + 1,
        torch.zeros_like(self._open_dwell[open_ids]),
      )
      ready = self._open_dwell[open_ids] >= self.cfg.open_dwell_steps
      ready_ids = open_ids[ready]
      replace_ids = ready_ids[self._replacement_pending[ready_ids]]
      if len(replace_ids) > 0:
        reset_cylinder_on_table(
          self._env,
          replace_ids,
          nominal_spawn_xy=self.cfg.recovery_spawn_xy,
          xy_jitter=self.cfg.recovery_xy_jitter,
          yaw_range=self.cfg.recovery_yaw_range,
          table_height=self.cfg.table_height_m,
          spawn_clearance=0.0,
          scale_list=(1.0,),
          scale_jitter=0.0,
          cylinder_cfg=SceneEntityCfg(
            self.cfg.entity_name, geom_names=(self.cfg.geom_name,)
          ),
        )
        self._replacement_pending[replace_ids] = False
      self._set_phase(ready_ids, PICKUP_PHASE)

    pickup_ids = (
      (self._phase == PICKUP_PHASE).nonzero(as_tuple=False).squeeze(-1)
    )
    if len(pickup_ids) > 0:
      self._pickup_elapsed_steps[pickup_ids] += 1
      lifted = clearance[pickup_ids] >= self.cfg.target_clearance_m
      contacted = contact_count[pickup_ids] >= self.cfg.minimum_contacts
      acquired = lifted & contacted
      self._acquire_dwell[pickup_ids] = torch.where(
        acquired,
        self._acquire_dwell[pickup_ids] + 1,
        torch.zeros_like(self._acquire_dwell[pickup_ids]),
      )
      complete = (
        self._acquire_dwell[pickup_ids] >= self.cfg.acquire_dwell_steps
      ) & (
        self._pickup_elapsed_steps[pickup_ids]
        >= self.cfg.minimum_pickup_steps
      )
      complete_ids = pickup_ids[complete]
      if len(complete_ids) > 0:
        self._completed_pickups[complete_ids] += 1
        self._set_phase(complete_ids, MANIPULATE_PHASE)
        self._restart_translation(complete_ids)

    manipulate_ids = (
      (self._phase == MANIPULATE_PHASE).nonzero(as_tuple=False).squeeze(-1)
    )
    if len(manipulate_ids) > 0:
      speed = torch.linalg.vector_norm(
        self.object.data.root_link_vel_w[manipulate_ids, :3], dim=-1
      )
      stable_on_table = (
        clearance[manipulate_ids] <= self.cfg.table_recovery_clearance_m
      ) & (speed <= self.cfg.maximum_table_speed_mps)
      self._table_dwell[manipulate_ids] = torch.where(
        stable_on_table,
        self._table_dwell[manipulate_ids] + 1,
        torch.zeros_like(self._table_dwell[manipulate_ids]),
      )
      returned = (
        self._table_dwell[manipulate_ids] >= self.cfg.table_dwell_steps
      )
      self.request_table_pickup(manipulate_ids[returned])


@dataclass(kw_only=True)
class CylinderTablePickupCommandCfg(CommandTermCfg):
  entity_name: str = "cube"
  hand_name: str = "robot"
  sensor_name: str = "fingertip_cube_contact"
  geom_name: str = "cube_geom"
  translation_command_name: str | None = None
  open_joint_pos: tuple[float, ...]
  table_height_m: float = 0.0
  target_clearance_m: float = 0.008
  table_recovery_clearance_m: float = 0.004
  maximum_table_speed_mps: float = 0.10
  minimum_contacts: int = 2
  recovery_spawn_xy: tuple[float, float] = (0.0, 0.0)
  recovery_xy_jitter: tuple[float, float] = (0.0, 0.0)
  recovery_yaw_range: tuple[float, float] = (-3.14159, 3.14159)
  open_joint_tolerance_rad: float = 0.12
  open_dwell_steps: int = 5
  acquire_dwell_steps: int = 40
  minimum_pickup_steps: int = 1
  table_dwell_steps: int = 5

  def build(self, env) -> CylinderTablePickupCommand:
    return CylinderTablePickupCommand(self, env)


def place_cylinder_on_table_for_recovery(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | slice | None,
  command_name: str,
  nominal_spawn_xy: tuple[float, float],
  xy_jitter: tuple[float, float],
  yaw_range: tuple[float, float],
  table_height: float,
  cylinder_cfg: SceneEntityCfg,
  probability: float = 1.0,
) -> None:
  """Remove an acquired cylinder, then replace it only after the hand opens."""
  if not 0.0 <= probability <= 1.0:
    raise ValueError("probability must be in [0, 1]")
  ids = _env_ids_tensor(env, env_ids)
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, CylinderTablePickupCommand):
    raise TypeError(f"{command_name!r} is not a CylinderTablePickupCommand")
  ids = ids[command.phase[ids] == MANIPULATE_PHASE]
  if probability < 1.0 and len(ids) > 0:
    selected = torch.rand(len(ids), device=env.device) < probability
    ids = ids[selected]
  if len(ids) == 0:
    return
  reset_cylinder_on_table(
    env,
    ids,
    nominal_spawn_xy=(nominal_spawn_xy[0] + 0.12, nominal_spawn_xy[1]),
    xy_jitter=(0.0, 0.0),
    yaw_range=yaw_range,
    table_height=table_height,
    spawn_clearance=0.0,
    scale_list=(1.0,),
    scale_jitter=0.0,
    cylinder_cfg=cylinder_cfg,
  )
  command.request_external_table_pickup(ids)


def pickup_open_pose_reward(
  env: "ManagerBasedRlEnv", command_name: str, std_rad: float = 0.20
) -> torch.Tensor:
  if std_rad <= 0.0:
    raise ValueError("std_rad must be positive")
  command = env.command_manager.get_term(command_name)
  error = torch.mean(
    torch.square(command.hand.data.joint_pos - command.open_joint_target), dim=-1
  )
  return torch.exp(-error / (std_rad**2)) * (
    command.phase == OPEN_PHASE
  ).float()


def pickup_joint_pose_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  open_joint_pos: tuple[float, ...],
  approach_joint_pos: tuple[float, ...],
  lift_joint_pos: tuple[float, ...],
  secure_joint_pos: tuple[float, ...],
  approach_steps: int,
  lift_steps: int,
  secure_steps: int,
  std_rad: float = 0.75,
) -> torch.Tensor:
  """Track the verified approach, lift, and secure joint trajectory."""
  if (
    std_rad <= 0.0
    or approach_steps <= 0
    or lift_steps <= 0
    or secure_steps <= 0
  ):
    raise ValueError("std and all trajectory steps must be positive")
  command = env.command_manager.get_term(command_name)
  q = command.hand.data.joint_pos
  poses = [
    torch.as_tensor(values, device=q.device, dtype=q.dtype)
    for values in (
      open_joint_pos,
      approach_joint_pos,
      lift_joint_pos,
      secure_joint_pos,
    )
  ]
  if any(pose.shape != (command.hand.num_joints,) for pose in poses):
    raise ValueError(f"Each joint pose must have {command.hand.num_joints} values")
  open_pose, approach_pose, lift_pose, secure_pose = poses
  elapsed = command.pickup_elapsed_steps.to(dtype=q.dtype)
  approach_alpha = torch.clamp(elapsed / float(approach_steps), 0.0, 1.0)
  lift_alpha = torch.clamp(
    (elapsed - float(approach_steps)) / float(lift_steps), 0.0, 1.0
  )
  secure_alpha = torch.clamp(
    (elapsed - float(approach_steps + lift_steps)) / float(secure_steps),
    0.0,
    1.0,
  )
  target = open_pose + approach_alpha.unsqueeze(-1) * (
    approach_pose - open_pose
  )
  target = target + lift_alpha.unsqueeze(-1) * (lift_pose - approach_pose)
  target = target + secure_alpha.unsqueeze(-1) * (secure_pose - lift_pose)
  error = torch.mean(torch.square(q - target), dim=-1)
  return torch.exp(-error / (std_rad**2)) * (
    command.phase == PICKUP_PHASE
  ).float()


def pickup_reference_action_reward(
  env: "ManagerBasedRlEnv",
  command_name: str,
  action_name: str,
  open_joint_pos: tuple[float, ...],
  approach_joint_pos: tuple[float, ...],
  lift_joint_pos: tuple[float, ...],
  secure_joint_pos: tuple[float, ...],
  approach_steps: int,
  lift_steps: int,
  secure_steps: int,
  maximum_command_delta_rad: float,
  std: float = 0.75,
) -> torch.Tensor:
  """Teach the exact constant-increment approach, curl, and hold sequence."""
  if (
    maximum_command_delta_rad <= 0.0
    or std <= 0.0
    or approach_steps <= 0
    or lift_steps <= 0
    or secure_steps <= 0
  ):
    raise ValueError("action scale, std, and trajectory steps must be positive")
  command = env.command_manager.get_term(command_name)
  action = env.action_manager.get_term(action_name)
  dtype = command.hand.data.joint_pos.dtype
  device = command.hand.data.joint_pos.device
  poses = [
    torch.as_tensor(values, device=device, dtype=dtype)[action.target_ids]
    for values in (
      open_joint_pos,
      approach_joint_pos,
      lift_joint_pos,
      secure_joint_pos,
    )
  ]
  if any(pose.shape != (action.action_dim,) for pose in poses):
    raise ValueError(f"Each joint pose must have {action.action_dim} values")
  open_pose, approach_pose, lift_pose, secure_pose = poses
  approach_action = torch.clamp(
    (approach_pose - open_pose)
    / float(approach_steps)
    / maximum_command_delta_rad,
    -1.0,
    1.0,
  )
  lift_action = torch.clamp(
    (lift_pose - approach_pose)
    / float(lift_steps)
    / maximum_command_delta_rad,
    -1.0,
    1.0,
  )
  secure_action = torch.clamp(
    (secure_pose - lift_pose)
    / float(secure_steps)
    / maximum_command_delta_rad,
    -1.0,
    1.0,
  )
  elapsed = command.pickup_elapsed_steps
  desired = torch.zeros_like(action.raw_action)
  approach_mask = elapsed <= approach_steps
  lift_mask = (elapsed > approach_steps) & (
    elapsed <= approach_steps + lift_steps
  )
  secure_mask = (elapsed > approach_steps + lift_steps) & (
    elapsed <= approach_steps + lift_steps + secure_steps
  )
  desired[approach_mask] = approach_action
  desired[lift_mask] = lift_action
  desired[secure_mask] = secure_action
  error = torch.mean(torch.square(action.raw_action - desired), dim=-1)
  return torch.exp(-error / (std**2)) * (
    command.phase == PICKUP_PHASE
  ).float()


class cylinder_fingertip_proximity:
  """Dense distance reward to the measured capped-cylinder surface."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.hand_cfg: SceneEntityCfg = cfg.params["hand_cfg"]
    self.cylinder_cfg: SceneEntityCfg = cfg.params["cylinder_cfg"]
    self.hand: Entity = env.scene[self.hand_cfg.name]
    self.cylinder: Entity = env.scene[self.cylinder_cfg.name]
    if isinstance(self.hand_cfg.geom_ids, slice):
      self.tip_ids = list(range(self.hand.num_geoms))[self.hand_cfg.geom_ids]
    else:
      self.tip_ids = self.hand_cfg.geom_ids
    if isinstance(self.cylinder_cfg.geom_ids, slice):
      object_ids = list(range(self.cylinder.num_geoms))[
        self.cylinder_cfg.geom_ids
      ]
    else:
      object_ids = self.cylinder_cfg.geom_ids
    if len(self.tip_ids) < 1 or len(object_ids) != 1:
      raise ValueError("Expected fingertips and one cylinder collision geom")
    self.cylinder_local_geom_id = int(object_ids[0])
    self.cylinder_world_geom_id = int(
      self.cylinder.indexing.geom_ids[self.cylinder_local_geom_id].item()
    )

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    hand_cfg: SceneEntityCfg,
    cylinder_cfg: SceneEntityCfg,
    surface_margin_m: float = 0.004,
    distance_scale_m: float = 0.020,
    worst_tip_weight: float = 0.45,
  ) -> torch.Tensor:
    del env, hand_cfg, cylinder_cfg
    if distance_scale_m <= 0.0 or surface_margin_m < 0.0:
      raise ValueError("Invalid cylinder proximity distances")
    if not 0.0 <= worst_tip_weight <= 1.0:
      raise ValueError("worst_tip_weight must be in [0, 1]")
    tip_pos_w = self.hand.data.geom_pos_w[:, self.tip_ids]
    object_pos_w = self.cylinder.data.geom_pos_w[
      :, self.cylinder_local_geom_id
    ]
    object_quat_w = self.cylinder.data.geom_quat_w[
      :, self.cylinder_local_geom_id
    ]
    quat = object_quat_w[:, None, :].expand(-1, len(self.tip_ids), -1)
    tip_local = quat_apply_inverse(
      quat, tip_pos_w - object_pos_w[:, None, :]
    )
    size = self.cylinder.data.model.geom_size[
      :, self.cylinder_world_geom_id
    ]
    radial = torch.linalg.vector_norm(tip_local[:, :, :2], dim=-1)
    radial_outside = torch.clamp(radial - size[:, None, 0], min=0.0)
    axial_outside = torch.clamp(
      torch.abs(tip_local[:, :, 2]) - size[:, None, 1], min=0.0
    )
    distance = torch.sqrt(radial_outside**2 + axial_outside**2)
    proximity = torch.exp(
      -torch.clamp(distance - surface_margin_m, min=0.0) / distance_scale_m
    )
    mean_score = proximity.mean(dim=-1)
    worst_score = proximity.amin(dim=-1)
    return (
      (1.0 - worst_tip_weight) * mean_score
      + worst_tip_weight * worst_score
    )


def contact_gated_cylinder_lift_progress(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  target_clearance: float,
  minimum_contacts: int,
  table_height: float,
  cylinder_cfg: SceneEntityCfg,
) -> torch.Tensor:
  clearance = torch.clamp(
    cylinder_bottom_clearance(env, table_height, cylinder_cfg)
    / target_clearance,
    min=0.0,
    max=1.0,
  )
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contact_ok = (sensor.data.found > 0).sum(dim=-1) >= minimum_contacts
  return clearance * contact_ok.float()


def sustained_cylinder_grasp_lift(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  target_clearance: float,
  minimum_contacts: int,
  table_height: float,
  cylinder_cfg: SceneEntityCfg,
) -> torch.Tensor:
  clearance_ok = (
    cylinder_bottom_clearance(env, table_height, cylinder_cfg)
    >= target_clearance
  )
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contact_ok = (sensor.data.found > 0).sum(dim=-1) >= minimum_contacts
  return (clearance_ok & contact_ok).float()


def cylinder_grasp_success_metric(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  target_clearance: float,
  minimum_contacts: int,
  table_height: float,
  cylinder_cfg: SceneEntityCfg,
) -> torch.Tensor:
  return sustained_cylinder_grasp_lift(
    env,
    sensor_name=sensor_name,
    target_clearance=target_clearance,
    minimum_contacts=minimum_contacts,
    table_height=table_height,
    cylinder_cfg=cylinder_cfg,
  )


class pickup_completion_reward:
  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    del cfg
    self._previous = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._previous[env_ids] = 0

  def __call__(
    self, env: "ManagerBasedRlEnv", command_name: str
  ) -> torch.Tensor:
    completed = env.command_manager.get_term(command_name).completed_pickups
    pulse = completed > self._previous
    self._previous[:] = completed
    return pulse.float()


def cylinder_upright_reward(
  env: "ManagerBasedRlEnv",
  cylinder_cfg: SceneEntityCfg,
  std_rad: float = 0.20,
) -> torch.Tensor:
  if std_rad <= 0.0:
    raise ValueError("std_rad must be positive")
  cylinder: Entity = env.scene[cylinder_cfg.name]
  rotation = matrix_from_quat(cylinder.data.root_link_quat_w)
  cosine = torch.abs(rotation[:, 2, 2])
  tilt = torch.acos(torch.clamp(cosine, -1.0, 1.0))
  return torch.exp(-0.5 * torch.square(tilt / std_rad))


def manipulating_cylinder_tilt_above(
  env: "ManagerBasedRlEnv",
  command_name: str,
  cylinder_cfg: SceneEntityCfg,
  max_angle_rad: float,
) -> torch.Tensor:
  cylinder: Entity = env.scene[cylinder_cfg.name]
  rotation = matrix_from_quat(cylinder.data.root_link_quat_w)
  cosine = torch.abs(rotation[:, 2, 2])
  tilt = torch.acos(torch.clamp(cosine, -1.0, 1.0))
  phase = env.command_manager.get_term(command_name).phase
  return (phase == MANIPULATE_PHASE) & (tilt > max_angle_rad)


def pickup_phase_metric(
  env: "ManagerBasedRlEnv", command_name: str
) -> torch.Tensor:
  return env.command_manager.get_term(command_name).phase.float()


def completed_pickups_metric(
  env: "ManagerBasedRlEnv", command_name: str
) -> torch.Tensor:
  return env.command_manager.get_term(command_name).completed_pickups.float()
