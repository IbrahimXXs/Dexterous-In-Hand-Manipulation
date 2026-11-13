"""Tests for the sensorless fingertip force estimator (experiments/force_estimation).

CPU tests: label aggregation, MuJoCo contact-sensor semantics (units, passivity,
multi-contact sums), model/inference contracts.  GPU tests (skipped without CUDA):
label extraction on the real LEAP env, privileged-information isolation, and
dataset synchronisation of a tiny collected shard.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from in_hand_rotation_mjlab.estimation.layout import ACTOR_OBS_DIM
from in_hand_rotation_mjlab.force_estimation.constants import DISTAL_BODIES, FORCE_FINGER_NAMES
from in_hand_rotation_mjlab.force_estimation.labels import aggregate_normal_force, tangential_magnitude
from in_hand_rotation_mjlab.force_estimation.model import (
  FingertipForceEstimator,
  ForceEstimatorConfig,
  load_force_estimator,
)

PROJECT = Path(__file__).resolve().parents[1]
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# ----------------------------------------------------------------- label aggregation (CPU)
def test_aggregation_sums_multiple_contacts_and_ignores_empty_slots():
  f = torch.zeros(2, 4, 8, 3)
  f[0, 0, 0] = torch.tensor([1.5, 0.3, -0.2])
  f[0, 0, 1] = torch.tensor([0.5, -0.1, 0.0])
  f[0, 2, 3] = torch.tensor([2.0, 0.0, 0.4])
  fn, neg = aggregate_normal_force(f)
  assert torch.allclose(fn[0], torch.tensor([2.0, 0.0, 2.0, 0.0]))
  assert torch.all(fn[1] == 0)  # no contact -> exactly zero
  assert neg.sum() == 0
  assert torch.allclose(tangential_magnitude(f)[0, 0], torch.tensor(np.hypot(0.3, 0.2) + 0.1, dtype=torch.float32))


def test_aggregation_clips_negative_normal_and_counts_it():
  f = torch.zeros(1, 1, 3, 3)
  f[0, 0, 0, 0] = -1e-7
  f[0, 0, 1, 0] = 1.0
  fn, neg = aggregate_normal_force(f)
  assert fn.item() == pytest.approx(1.0) and neg.item() == 1


# --------------------------------------------- MuJoCo contact-sensor semantics (CPU, deterministic)
def _box_scene(with_sensor: bool, two_boxes: bool = False) -> mujoco.MjModel:
  spec = mujoco.MjSpec()
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1, 1, 0.1))
  # a "finger" body with two collision geoms (like the distal link: tip + box)
  finger = spec.worldbody.add_body(name="finger", pos=(0, 0, 0.0))
  finger.add_geom(name="pad_a", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.01), pos=(-0.06, 0, 0.01))
  finger.add_geom(name="pad_b", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, 0.01), pos=(0.06, 0, 0.01))
  cube = spec.worldbody.add_body(name="cube", pos=(0, 0, 0.05))
  cube.add_freejoint()
  cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=(0.1, 0.03, 0.03), mass=0.085)
  if with_sensor:
    spec.add_sensor(name="s", type=mujoco.mjtSensor.mjSENS_CONTACT, objtype=mujoco.mjtObj.mjOBJ_BODY, objname="finger",
                    reftype=mujoco.mjtObj.mjOBJ_GEOM, refname="cube_geom", intprm=[1 << 1, 2, 8])  # force, maxforce, 8 slots
  return spec.compile()


def test_contact_sensor_is_passive_and_reports_newtons():
  m0, m1 = _box_scene(False), _box_scene(True)
  d0, d1 = mujoco.MjData(m0), mujoco.MjData(m1)
  for _ in range(600):
    mujoco.mj_step(m0, d0)
    mujoco.mj_step(m1, d1)
  # adding the sensor does not change the dynamics (bitwise)
  assert np.array_equal(d0.qpos, d1.qpos) and np.array_equal(d0.qvel, d1.qvel)
  mujoco.mj_forward(m1, d1)
  slots = d1.sensordata.reshape(8, 3)
  fn, _ = aggregate_normal_force(torch.tensor(slots)[None])
  # independent: mj_contactForce over finger-geom/cube contacts
  cube = m1.geom("cube_geom").id
  finger = m1.body("finger").id
  ref, ncon = 0.0, 0
  for i in range(d1.ncon):
    c = d1.contact[i]
    g = {int(c.geom1), int(c.geom2)}
    if cube in g and any(m1.geom_bodyid[x] == finger for x in g - {cube}):
      out = np.zeros(6)
      mujoco.mj_contactForce(m1, d1, i, out)
      ref += out[0]
      ncon += 1
  assert ncon >= 2  # multiple contact points over two geoms of the same body
  assert fn.item() == pytest.approx(ref, rel=1e-6)
  # static equilibrium: the cube rests only on the finger body -> sum of normal forces = m g (N)
  assert fn.item() == pytest.approx(0.085 * 9.81, rel=2e-2)


def test_contact_sensor_zero_without_contact():
  m = _box_scene(True)
  d = mujoco.MjData(m)
  d.qpos[2] = 0.5  # cube high above the finger
  mujoco.mj_forward(m, d)
  assert np.all(d.sensordata == 0)


# --------------------------------------------------------------- model / inference (CPU)
@pytest.mark.parametrize("head", ["direct", "hurdle"])
def test_model_outputs_nonnegative_finite_and_checks_dim(head):
  torch.manual_seed(0)
  m = FingertipForceEstimator(ForceEstimatorConfig(head=head)).eval()
  m.set_normalization(torch.zeros(320), torch.ones(320), torch.full((4,), 1.7))
  x = torch.randn(256, ACTOR_OBS_DIM) * torch.tensor([1.0, 100.0, 1e4]).repeat_interleave(torch.tensor([100, 110, 110]))
  y = m(x)
  assert y.shape == (256, 4) and torch.isfinite(y).all() and (y >= 0).all()
  with pytest.raises(ValueError):
    m(torch.zeros(3, ACTOR_OBS_DIM + 1))
  assert m.cfg.obs_dim == ACTOR_OBS_DIM == 320


def test_checkpoint_roundtrip_and_reproducible_inference(tmp_path):
  from in_hand_rotation_mjlab.force_estimation.inference import ForceEstimatorRuntime

  torch.manual_seed(1)
  m = FingertipForceEstimator(ForceEstimatorConfig(head="hurdle"))
  m.set_normalization(torch.randn(320), torch.rand(320) + 0.5, torch.tensor([1.8, 1.4, 1.0, 1.3]))
  p = tmp_path / "fe.pt"
  torch.save(m.checkpoint(), p)
  x = np.random.default_rng(0).normal(size=(5, 320)).astype(np.float32)
  r1, r2 = ForceEstimatorRuntime(str(p)), ForceEstimatorRuntime(str(p))
  a, b, c = r1.predict(x), r1.predict(x), r2.predict(x)
  assert np.array_equal(a, b) and np.array_equal(a, c)
  assert np.allclose(a, m.eval()(torch.from_numpy(x)).detach().numpy(), atol=1e-6)
  assert r1.predict(x[0]).shape == (4,)
  assert r1.output_names == tuple(f"{n}_force_N" for n in ("thumb", "index", "middle", "ring"))
  with pytest.raises(ValueError):
    r1.predict(np.full(320, np.nan, np.float32))
  bad = m.checkpoint()
  bad["format"] = "something_else"
  torch.save(bad, tmp_path / "bad.pt")
  with pytest.raises(ValueError):
    load_force_estimator(tmp_path / "bad.pt")


def test_force_estimator_is_independent_of_estimator_b():
  m = FingertipForceEstimator(ForceEstimatorConfig())
  keys = set(m.state_dict())
  assert all(k.startswith(("trunk.", "head.", "obs_", "force_scale")) for k in keys)
  assert FORCE_FINGER_NAMES == ("thumb", "index", "middle", "ring")


def test_released_checkpoint_loads_if_present():
  p = PROJECT / "experiments/force_estimation/models/force_estimator.pt"
  if not p.exists():
    pytest.skip("model not trained yet")
  from in_hand_rotation_mjlab.force_estimation.inference import ForceEstimatorRuntime

  r = ForceEstimatorRuntime(str(p))
  y = r.predict(np.zeros((3, 320), np.float32))
  assert y.shape == (3, 4) and np.isfinite(y).all() and (y >= 0).all()


# ----------------------------------------------------------------------- GPU: real LEAP env
@pytest.fixture(scope="module")
def leap_env():
  if not torch.cuda.is_available():
    pytest.skip("needs CUDA")
  from in_hand_rotation_mjlab.force_estimation.labels import make_force_env

  env, raw, agent_cfg = make_force_env("A", 32, 4242, "cuda:0")
  return env, raw, agent_cfg


@GPU
def test_actor_observation_has_no_privileged_terms(leap_env):
  env, raw, _ = leap_env
  om = raw.observation_manager
  terms = list(om.active_terms["actor"])
  assert terms == ["joint_pos", "prev_commanded_joint_pos"], terms
  obs = env.get_observations()
  assert obs["actor"].shape[-1] == ACTOR_OBS_DIM
  # the force sensor must not appear in any actor term
  assert not any(("force" in t) or ("contact" in t) or ("cube" in t) for t in terms)


@GPU
def test_labels_correspond_to_actual_fingertip_cube_contacts(leap_env):
  import warp as wp

  from in_hand_rotation_mjlab.force_estimation.labels import ForceLabeler, raw_contact_normal_forces

  env, raw, _ = leap_env
  lab = ForceLabeler(raw)
  m = raw.sim.mj_model
  cube = [i for i in range(m.ngeom) if m.geom(i).name.endswith("cube_geom")][0]
  bodies = [m.body(f"robot/{b}").id for b in DISTAL_BODIES]
  obs = env.get_observations()
  seen_contact = 0
  torch.manual_seed(0)
  with torch.inference_mode():
    for _ in range(40):
      obs, *_ = env.step(torch.randn(raw.num_envs, 16, device=raw.device) * 0.3)
      now = lab.now()
      fn = now["normal"]
      assert torch.isfinite(fn).all() and (fn >= 0).all()
      assert torch.allclose(fn, raw_contact_normal_forces(raw), atol=1e-4)
      d = raw.sim.data.struct
      nacon = int(wp.to_torch(d.nacon)[0])
      geom = wp.to_torch(d.contact.geom)[:nacon].cpu().numpy()
      wid = wp.to_torch(d.contact.worldid)[:nacon].cpu().numpy()
      for k, b in enumerate(bodies):
        touching = np.zeros(raw.num_envs, bool)
        for (g0, g1), w in zip(geom, wid):
          if (g0 == cube and m.geom_bodyid[g1] == b) or (g1 == cube and m.geom_bodyid[g0] == b):
            touching[w] = True
        f = fn[:, k].cpu().numpy()
        assert np.all(f[~touching] == 0), "force reported without a fingertip-cube contact"
        assert np.all((now["found"][:, k].cpu().numpy() > 0) == touching)
        seen_contact += int((f > 0).sum())
      itv = lab.interval()
      assert (itv["mean"] >= 0).all() and (itv["mean"] <= itv["max"] + 1e-6).all()
  assert seen_contact > 0


@GPU
def test_collected_shard_is_synchronised(tmp_path):
  ck = PROJECT / "experiments/scratch_state_estimation/checkpoints/A_seed1_model_1000.pt"
  env = dict(os.environ, MUJOCO_GL="egl")
  r = subprocess.run([sys.executable, str(PROJECT / "experiments/force_estimation/scripts/collect_force_data.py"),
                      "--condition", "A", "--checkpoint", str(ck), "--policy-name", "A_seed1_it1000", "--split", "unit",
                      "--seed", "999", "--num-envs", "16", "--num-steps", "30", "--out", str(tmp_path)],
                     cwd=PROJECT, env=env, capture_output=True, text=True, timeout=1200)
  assert r.returncode == 0, r.stderr[-3000:]
  d = next((tmp_path / "unit").glob("*"))
  meta = json.loads((d / "meta.json").read_text())
  T, N = 30, 16
  obs = np.load(d / "obs.npy")
  assert obs.shape == (T, N, 320) and np.isfinite(obs).all()
  for k in ("force_mean", "force_now", "force_max", "contact_frac", "tangential_now", "found_now"):
    assert np.load(d / f"{k}.npy").shape == (T, N, 4), k
  for k in ("step_in_episode", "episode_id", "end_reason", "other_normal_now"):
    assert np.load(d / f"{k}.npy").shape == (T, N), k
  fm = np.load(d / "force_mean.npy")
  assert (fm >= 0).all() and np.isfinite(fm).all()
  sie = np.load(d / "step_in_episode.npy")
  # first row of an episode has a zeroed substep history -> its interval label is all-zero (and masked)
  assert np.all(fm[sie == 0] == 0)
  assert meta["audit_sensor_vs_raw_max_abs_diff_N"] < 1e-3 and meta["rows_with_slot_overflow"] == 0
  assert meta["actor_terms"] == ["joint_pos", "prev_commanded_joint_pos"]
  from in_hand_rotation_mjlab.force_estimation.dataset import load_shard

  s = load_shard(d)
  assert len(s["obs"]) == len(s["force_mean"]) == int((sie > 0).sum())
  # row alignment: loader row j <-> (t, env) grid position
  j = len(s["t"]) // 2
  assert np.array_equal(s["obs"][j], obs[s["t"][j], s["env"][j]])
  assert np.array_equal(s["force_mean"][j], fm[s["t"][j], s["env"][j]])
