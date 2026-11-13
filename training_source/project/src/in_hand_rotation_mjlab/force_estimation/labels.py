"""Ground-truth fingertip-cube normal contact forces from MuJoCo (labels only).

Two independent extraction paths are implemented; they must agree (tested):

1. ``ForceLabeler`` reads a passive MJLab ``ContactSensor`` that matches every
   geom of each distal finger body against the cube geom (MuJoCo ``contact``
   sensor, top-``FORCE_SENSOR_SLOTS`` contacts by force, contact-frame force).
   Component 0 of a contact-frame force is the normal force; the per-finger
   label is the sum over that finger's contacts.  ``history_length = decimation``
   gives the force of every physics substep of the last control interval.
2. ``raw_contact_normal_forces`` walks MuJoCo-Warp's raw contact list and
   reads the normal force from ``efc.force`` at each contact's ``efc_address``
   (elliptic cone: the first constraint row of a contact is its normal force).

Conventions
-----------
* Units: newtons (MuJoCo SI).
* Compressive normal force is non-negative under the elliptic friction cone
  used by this model (``opt.cone = 1``); tiny negative round-off is clipped and
  counted.
* Multiple contacts (mesh/box vs box produce several points) are summed;
  contacts of different geoms of the same distal body are summed (no geom is
  matched twice: each contact has exactly one (robot geom, cube geom) pair).
* No contact -> exactly 0.
* Sensor values are read from the live ``sensordata`` buffer, never from the
  sensor's per-substep cache (which reward/metric terms may fill before the
  final ``forward()`` of ``env.step``).
"""

from __future__ import annotations

import torch

from .constants import DISTAL_BODIES, FORCE_FINGERS, FORCE_SENSOR_NAME, FORCE_SENSOR_SLOTS


def force_sensor_cfg(decimation: int, num_slots: int = FORCE_SENSOR_SLOTS):
  """Passive contact sensor: distal finger bodies x cube geom."""
  from mjlab.sensor import ContactMatch, ContactSensorCfg

  return ContactSensorCfg(
    name=FORCE_SENSOR_NAME,
    primary=ContactMatch(mode="body", pattern=DISTAL_BODIES, entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=num_slots,
    history_length=int(decimation),
  )


def aggregate_normal_force(slot_forces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """[..., K, 3] contact-frame forces -> ([...] summed normal force >= 0, [...] negative-count).

  Empty slots are all-zero, so they contribute nothing.
  """
  fn = slot_forces[..., 0]
  neg = (fn < 0).sum(-1)
  return fn.clamp_min(0.0).sum(-1), neg


def tangential_magnitude(slot_forces: torch.Tensor) -> torch.Tensor:
  """[..., K, 3] -> [...] sum over contacts of |friction force| (contact frame, components 1, 2)."""
  return torch.linalg.vector_norm(slot_forces[..., 1:3], dim=-1).sum(-1)


class ForceLabeler:
  """Per-finger labels from the passive force sensor, in ``FORCE_FINGERS`` order."""

  def __init__(self, raw_env, sensor_name: str = FORCE_SENSOR_NAME):
    self.env = raw_env
    self.sensor = raw_env.scene[sensor_name]
    self.K = self.sensor.cfg.num_slots
    prims: list[str] = []
    for s in self.sensor._slots:
      if s.primary_name not in prims:
        prims.append(s.primary_name)
    want = list(DISTAL_BODIES)
    short = [p.split("/")[-1] for p in prims]
    if sorted(short) != sorted(want):
      raise RuntimeError(f"force sensor primaries {prims} != {want}")
    # permutation from sensor order to FORCE_FINGERS order
    self.perm = [short.index(b) for b in want]
    self._views = {f: [None] * len(prims) for f in ("found", "force")}
    for s in self.sensor._slots:
      self._views[s.field_name][prims.index(s.primary_name)] = s.data_view

  def _live(self, field: str) -> torch.Tensor:
    vs = [self._views[field][i] for i in self.perm]
    if field == "found":
      return torch.stack([v[:, 0] for v in vs], 1)
    return torch.stack([v.reshape(v.shape[0], self.K, 3) for v in vs], 1)  # [B, 4, K, 3]

  def now(self) -> dict[str, torch.Tensor]:
    """Forces at the current state (after env.step's final forward(): aligned with the observation)."""
    f = self._live("force")
    fn, neg = aggregate_normal_force(f)
    return {"normal": fn, "tangential": tangential_magnitude(f), "found": self._live("found"), "neg": neg}

  def interval(self) -> dict[str, torch.Tensor]:
    """Per-substep forces of the last control interval (forces actually applied by mj_step).

    The sensor history is [B, 4*K, H, 3] in sensor primary order, index 0 = most recent substep.
    Reset envs have a zeroed history (their rows are masked downstream).
    """
    h = self.sensor._history_state["force"]
    B, NK, H, _ = h.shape
    h = h.view(B, NK // self.K, self.K, H, 3)[:, self.perm]  # [B, 4, K, H, 3]
    fn = h[..., 0].clamp_min(0.0).sum(2)  # [B, 4, H]
    return {"mean": fn.mean(-1), "max": fn.amax(-1), "contact_frac": (fn > 0).float().mean(-1)}


_STATIC: dict[int, dict] = {}


def _static(raw_env) -> dict:
  """Model-constant lookup tables (cached per env instance)."""
  key = id(raw_env)
  if key not in _STATIC:
    m = raw_env.sim.mj_model
    dev = raw_env.device
    cube = [i for i in range(m.ngeom) if m.geom(i).name.endswith("cube_geom")]
    assert len(cube) == 1, cube
    distal = torch.zeros(m.nbody, dtype=torch.bool, device=dev)
    for b in DISTAL_BODIES:
      distal[m.body(f"robot/{b}").id] = True
    _STATIC[key] = {
      "cube_geom": cube[0],
      "geom_body": torch.tensor(m.geom_bodyid, device=dev),
      "bodies": [m.body(f"robot/{b}").id for b in DISTAL_BODIES],
      "is_robot": torch.tensor([m.geom(i).name.startswith("robot/") for i in range(m.ngeom)], device=dev),
      "distal": distal,
    }
  return _STATIC[key]


def raw_contact_normal_forces(raw_env) -> torch.Tensor:
  """Independent path: [B, 4] summed normal force from the raw contact list + efc.force.

  Reflects the state of the most recent forward()/mj_step (same instant as ``ForceLabeler.now``).
  """
  import warp as wp

  st = _static(raw_env)
  d = raw_env.sim.data.struct
  dev = raw_env.device
  cube_geom, geom_body, bodies = st["cube_geom"], st["geom_body"], st["bodies"]
  nacon = int(wp.to_torch(d.nacon)[0])
  out = torch.zeros(raw_env.num_envs, len(bodies), device=dev)
  if nacon == 0:
    return out
  geom = wp.to_torch(d.contact.geom)[:nacon].long()
  wid = wp.to_torch(d.contact.worldid)[:nacon].long()
  adr = wp.to_torch(d.contact.efc_address)[:nacon, 0].long()
  efc_f = wp.to_torch(d.efc.force)
  g0, g1 = geom[:, 0], geom[:, 1]
  for k, b in enumerate(bodies):
    sel = (((geom_body[g0] == b) & (g1 == cube_geom)) | ((geom_body[g1] == b) & (g0 == cube_geom))) & (adr >= 0)
    if sel.any():
      out[:, k].index_add_(0, wid[sel], efc_f[wid[sel], adr[sel]].clamp_min(0.0))
  return out


def raw_other_hand_cube_normal_force(raw_env) -> torch.Tensor:
  """[B] summed normal force between the cube and robot geoms NOT on a distal finger body (palm, other links)."""
  import warp as wp

  st = _static(raw_env)
  d = raw_env.sim.data.struct
  dev = raw_env.device
  cube_geom, geom_body, is_robot, distal = st["cube_geom"], st["geom_body"], st["is_robot"], st["distal"]
  nacon = int(wp.to_torch(d.nacon)[0])
  out = torch.zeros(raw_env.num_envs, device=dev)
  if nacon == 0:
    return out
  geom = wp.to_torch(d.contact.geom)[:nacon].long()
  wid = wp.to_torch(d.contact.worldid)[:nacon].long()
  adr = wp.to_torch(d.contact.efc_address)[:nacon, 0].long()
  efc_f = wp.to_torch(d.efc.force)
  g0, g1 = geom[:, 0], geom[:, 1]
  other = torch.where(g0 == cube_geom, g1, g0)
  sel = ((g0 == cube_geom) | (g1 == cube_geom)) & is_robot[other] & ~distal[geom_body[other]] & (adr >= 0)
  if sel.any():
    out.index_add_(0, wid[sel], efc_f[wid[sel], adr[sel]].clamp_min(0.0))
  return out


def make_force_env(condition: str, num_envs: int, seed: int, device: str, *, preset: str = "nominal"):
  """The completed study's environment (``Mjlab-Leap-Scratch-<condition>``, curriculum frozen at
  final weights, optional robustness preset) plus the passive force sensor.  Nothing else changes."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.noise import UniformNoiseCfg

  from in_hand_rotation_mjlab.estimation.sim import _import_tasks
  from in_hand_rotation_mjlab.scratch_study.evaluation import PRESETS

  _import_tasks()
  task = f"Mjlab-Leap-Scratch-{condition}"
  env_cfg = load_env_cfg(task)  # deep copy: the registered config is untouched
  env_cfg.scene.num_envs = int(num_envs)
  env_cfg.seed = int(seed)
  env_cfg.curriculum = {}
  p = PRESETS[preset]
  for name, params in p.get("events", {}).items():
    env_cfg.events[name].params.update(params)
  if "obs_noise" in p:
    n = p["obs_noise"]
    env_cfg.observations["actor"].terms["joint_pos"].noise = UniformNoiseCfg(n_min=-n, n_max=n)
  env_cfg.scene.sensors = tuple(env_cfg.scene.sensors or ()) + (force_sensor_cfg(env_cfg.decimation),)
  agent_cfg = load_rl_cfg(task)
  raw = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
  return env, raw, agent_cfg


__all__ = [
  "FORCE_FINGERS",
  "ForceLabeler",
  "aggregate_normal_force",
  "force_sensor_cfg",
  "make_force_env",
  "raw_contact_normal_forces",
  "raw_other_hand_cube_normal_force",
  "tangential_magnitude",
]
