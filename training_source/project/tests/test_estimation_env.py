"""Environment-level checks (need CUDA + MJLab).  Skipped without a GPU.

Run:  CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_estimation_env.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
  pytest.skip("CUDA not available", allow_module_level=True)

from in_hand_rotation_mjlab.estimation.layout import FINGERTIP_NAMES, JOINT_NAMES
from in_hand_rotation_mjlab.estimation.sim import GroundTruth, load_policy, make_env

REPO = Path(__file__).resolve().parents[3]
CKPT = REPO / "checkpoint" / "model_400.pt"
DEV = "cuda:0"


@pytest.fixture(scope="module")
def sim():
  env, raw, agent_cfg = make_env(32, 123, DEV, obs_corruption=False, fix_reset_anchor=True)
  _, actor = load_policy(env, agent_cfg, CKPT, DEV)
  return env, raw, actor


def _roll(env, actor, steps):
  obs = env.get_observations()
  with torch.inference_mode():
    for _ in range(steps):
      obs, *_ = env.step(actor(obs))
  return obs


def test_actor_observation_layout(sim):
  env, raw, actor = sim
  gt = GroundTruth(raw)
  assert gt.actor_terms == ["joint_pos", "prev_commanded_joint_pos"]
  assert gt.actor_term_dims == [(160,), (160,)]
  assert tuple(raw.scene["robot"].joint_names) == JOINT_NAMES
  obs = _roll(env, actor, 15)["actor"]
  term = raw.action_manager.get_term("joint_pos")
  # newest command frame == current action target; history shifts by one step
  assert torch.allclose(obs[:, 304:320], term._target, atol=1e-6)
  assert torch.allclose(obs[:, 288:304], term._prev_target, atol=1e-6)
  # newest joint frame == joint_pos_rel now or one step ago (obs delay 0..1)
  robot = raw.scene["robot"]
  q_rel = robot.data.joint_pos - robot.data.default_joint_pos
  newest = obs[:, 144:160]
  close_now = (newest - q_rel).abs().max(-1).values < 1e-5
  close_prev = (newest - obs[:, 128:144]).abs().max(-1).values < 1e-5
  assert bool((close_now | close_prev).all())


def test_no_privileged_terms_in_actor(sim):
  _, raw, _ = sim
  om = raw.observation_manager
  assert om.group_obs_dim["actor"] == (320,)
  privileged = {"cube_pose_palm", "cube_lin_vel_palm", "cube_ang_vel_palm", "cube_mass", "cube_friction"}
  assert not privileged & set(om.active_terms["actor"])


def test_palm_frame_labels_recomputed(sim):
  env, raw, actor = sim
  obs = _roll(env, actor, 5)
  gt = GroundTruth(raw)
  lab = gt.labels(obs["critic"])
  from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul, quat_inv

  robot, cube = raw.scene["robot"], raw.scene["cube"]
  palm = robot.find_bodies("palm")[0][0]
  palm_q = robot.data.body_link_quat_w[:, palm]
  geoms, _ = robot.find_geoms("palm_collision_.*")
  centre = robot.data.geom_pos_w[:, geoms].mean(1)
  pos = quat_apply_inverse(palm_q, cube.data.root_link_pos_w - centre)
  quat = quat_mul(quat_inv(palm_q), cube.data.root_link_quat_w)
  assert torch.allclose(lab["pos"], pos, atol=1e-5)
  assert torch.allclose((lab["quat"] * quat).sum(-1).abs(), torch.ones(32, device=DEV), atol=1e-4)
  # cube centre within a hand-sized neighbourhood of the palm
  assert float(lab["pos"].norm(dim=-1).max()) < 0.15


def test_contact_labels_geometry(sim):
  env, raw, actor = sim
  gt = GroundTruth(raw)
  assert [n.replace("_tip", "") for n in gt.contact_names] == list(FINGERTIP_NAMES)
  robot, cube = raw.scene["robot"], raw.scene["cube"]
  tips, _ = robot.find_geoms(".*_tip", preserve_order=True)
  from mjlab.utils.lab_api.math import quat_apply_inverse

  dist_contact, dist_free = [], []
  obs = env.get_observations()
  with torch.inference_mode():
    for _ in range(40):
      obs, *_ = env.step(actor(obs))
      c = gt.labels(obs["critic"])["contact"].bool()
      tip = robot.data.geom_pos_w[:, tips]
      rel = quat_apply_inverse(
        cube.data.root_link_quat_w[:, None].expand(-1, 4, -1), tip - cube.data.root_link_pos_w[:, None]
      )
      half = 0.029
      d = (rel.abs() - half).clamp_min(0).norm(dim=-1)
      dist_contact.append(d[c])
      dist_free.append(d[~c])
  dc, df = torch.cat(dist_contact), torch.cat(dist_free)
  assert dc.numel() > 0
  # fingertip geom origins sit within ~2 cm of the cube surface when touching
  assert float(dc.quantile(0.95)) < 0.02
  assert float(dc.mean()) < float(df.mean())


def test_reset_anchor_fix():
  counts = {}
  for fixed in (False, True):
    env, raw, agent_cfg = make_env(64, 77, DEV, fix_reset_anchor=fixed)
    _, actor = load_policy(env, agent_cfg, CKPT, DEV)
    obs = env.get_observations()
    n = 0
    with torch.inference_mode():
      for _ in range(40):
        obs, _, d, _ = env.step(actor(obs))
        n += int(raw.termination_manager.get_term("cube_pose_deviation").sum())
    counts[fixed] = n
    env.close()
  # Original anchor: spurious cascade (most envs end almost every step).
  assert counts[False] > 5 * max(counts[True], 1)
