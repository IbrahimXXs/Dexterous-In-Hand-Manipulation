"""Live monitor / viewer checks (kit/monitor.py, kit/hand_scene.py). No hardware."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "LEAP_Hand_RL_Deploy_Kit"
sys.path.insert(0, str(KIT / "kit"))

from hand_scene import SERVER_JOINT_NAMES, HandScene  # noqa: E402
from monitor import HandMonitor, SymmetryUnwrapper, _rotvec  # noqa: E402


@pytest.fixture(scope="module")
def monitor():
    return HandMonitor(KIT / "monitors")  # raises if either estimator fails its self-check


@pytest.fixture(scope="module")
def scene():
    return HandScene()


def test_monitor_outputs(monitor):
    obs = monitor.push(np.zeros(16) + monitor.cube.obs_mean[:16] * 0, np.zeros(16), reset=True)
    out = monitor.update(obs)
    assert out["force_n"].shape == (4,) and np.all(out["force_n"] >= 0)
    assert np.isclose(np.linalg.norm(out["cube_quat_palm"]), 1.0, atol=1e-5)
    assert monitor.cube.self_check_error < 1e-3 and monitor.force.self_check_error < 1e-3


def test_monitor_matches_c_actor_estimator(monitor):
    """The monitor's cube estimator is the same file C's actor uses."""
    c_file = ROOT / "training_source/project/experiments/hwlimit_ac/deploy/policies/C_seed1/cube_estimator.npz"
    assert (KIT / "monitors/cube_estimator.npz").read_bytes() == c_file.read_bytes()


def test_palm_frame_and_unwrapped_yaw_on_reference(scene, monitor):
    """True MJLab states: palm-frame position matches the estimator's training range,
    and yaw integrated from symmetry-scrambled rotations equals the true world yaw."""
    ref = np.load(ROOT / "reference/exact_mjlab_seed11_state_trajectory.npz")
    q, cube = ref["joint_pos_rad"], ref["cube_pose_wxyz"].astype(np.float64)
    sym = monitor.cube.sym
    rng = np.random.default_rng(0)
    unw = SymmetryUnwrapper(sym, HandMonitor.YAW_AXIS_PALM)
    Rw, pos = [], []
    for t in range(len(q)):
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, cube[t, 3:])
        Rw.append(m.reshape(3, 3))
    gt, est = 0.0, 0.0
    for t in range(len(q)):
        jump = t > 0 and np.linalg.norm(_rotvec(Rw[t] @ Rw[t - 1].T)) > 0.3  # episode resets in the recording
        if t > 0 and not jump:
            gt += _rotvec(Rw[t] @ Rw[t - 1].T)[2]
        if jump:
            est += unw.yaw
            unw.reset()
        scene.set_hand(q[t])
        p, Rp = scene.world_to_palm(cube[t, :3], Rw[t])
        pos.append(p)
        unw.step(Rp @ sym[rng.integers(24)])
    est += unw.yaw
    assert abs(gt) > 3.0 and abs(est - gt) < 1e-6
    assert np.allclose(np.mean(pos[50:], axis=0), [0.0405, 0.0048, -0.0537], atol=0.012)


def test_palm_round_trip(scene):
    scene.set_hand(np.full(16, 0.3))
    scene.set_cube_palm(np.array([0.04, 0.0, -0.05]), np.array([0.9, 0.1, -0.2, 0.3]) / np.linalg.norm([0.9, 0.1, -0.2, 0.3]))
    p, R = scene.cube_world()
    pp, _ = scene.world_to_palm(p, R)
    assert np.allclose(pp, [0.04, 0.0, -0.05], atol=1e-12)


def test_tip_force_from_torque_known_answer(scene):
    m, d = scene.model, scene.data
    scene.set_hand(np.array([0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0, 0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37]))
    mujoco.mj_comPos(m, d)
    rng = np.random.default_rng(1)
    tau, true = np.zeros(16), {}
    for finger, prefix in (("thumb", "th"), ("index", "if"), ("middle", "mf"), ("ring", "rf")):
        idx = [i for i, n in enumerate(SERVER_JOINT_NAMES) if n.startswith(prefix)]
        dofs = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, SERVER_JOINT_NAMES[i])] for i in idx]
        g = scene.tip_geoms[finger]
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, d.geom_xpos[g], m.geom_bodyid[g])
        F = rng.normal(size=3)
        true[finger] = np.linalg.norm(F)
        tau[idx] = jacp[:, dofs].T @ F
    got = scene.tip_force_from_torque(tau, damping=1e-6)
    for f in true:
        assert got[f] == pytest.approx(true[f], rel=1e-3)
