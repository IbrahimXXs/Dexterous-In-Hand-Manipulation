from pathlib import Path

import mujoco
import numpy as np

from policy.yaw_policy import ContinuousYawPolicy

ROOT = Path(__file__).resolve().parents[1]


def test_portable_model_and_policy_load() -> None:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "models" / "scene_right_mbzuai_cube.xml"))
    assert model.nu == 16
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube") >= 0

    policy = ContinuousYawPolicy(ROOT / "checkpoint" / "policy_weights.npz")
    q = policy.reset()
    command, action, delta = policy.step(q)
    assert command.shape == action.shape == delta.shape == (16,)
    assert np.all(np.isfinite(command))
    assert np.max(np.abs(delta)) <= policy.delta_range + 1e-7

def test_training_source_is_included() -> None:
    task_root = ROOT / "training_source" / "project" / "src" / "in_hand_rotation_mjlab" / "tasks"
    assert task_root.is_dir()
    assert (ROOT / "training_source" / "training_params" / "env.yaml").is_file()
    assert (ROOT / "checkpoint" / "model_400.pt").is_file()
    trajectory = np.load(ROOT / "reference" / "exact_mjlab_seed11_state_trajectory.npz")
    assert trajectory["joint_pos_rad"].shape == (601, 16)
    assert trajectory["cube_pose_wxyz"].shape == (601, 7)
    assert float(np.min(trajectory["cube_pose_wxyz"][:, 2])) > 0.10

