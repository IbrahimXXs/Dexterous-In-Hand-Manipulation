#!/usr/bin/env python3
"""Kinematic MuJoCo scene of the right LEAP hand + cube for visualisation.

No physics is stepped: the hand joints are set from measured encoder angles
and the cube from the estimated palm-frame pose, then ``mj_kinematics`` places
everything.  The palm frame is the one the estimators were trained in
(MJLab ``palm_center_pose_w``): origin = mean world position of the
``palm_collision_*`` geoms, axes = the ``palm`` body frame.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

SERVER_JOINT_NAMES = (
    "if_mcp", "if_rot", "if_pip", "if_dip", "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip", "th_cmc", "th_axl", "th_mcp", "th_ipl",
)
TIP_GEOMS = {"thumb": "th_tip", "index": "if_tip", "middle": "mf_tip", "ring": "rf_tip"}
DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "models" / "scene_right_mbzuai_cube.xml"
CUBE_HALF = 0.029


class HandScene:
    def __init__(self, scene_xml: str | Path = DEFAULT_SCENE):
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                              for n in SERVER_JOINT_NAMES])
        cube_j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qadr = m.jnt_qposadr[cube_j]
        self.palm_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
        self.palm_geoms = [i for i in range(m.ngeom)
                           if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("palm_collision_")]
        if len(self.palm_geoms) != 10:
            raise RuntimeError(f"expected 10 palm_collision geoms, found {len(self.palm_geoms)}")
        self.tip_geoms = {f: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for f, g in TIP_GEOMS.items()}
        # Plain, see-through cube: face identity is not observable from proprioception.
        for name, rgba in (("cube_visual", (0, 0, 0, 0)), ("cube_geom", (1.0, 0.55, 0.1, 0.55))):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                m.geom_rgba[gid] = rgba
        self.set_hand(np.array([0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0, 0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37]))

    def set_hand(self, q_server: np.ndarray) -> None:
        self.data.qpos[self.qadr] = np.asarray(q_server, np.float64)
        mujoco.mj_kinematics(self.model, self.data)

    def palm_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """(palm-centre position [3], palm rotation [3, 3]) in world."""
        c = self.data.geom_xpos[self.palm_geoms].mean(axis=0)
        return c.copy(), self.data.xmat[self.palm_body].reshape(3, 3).copy()

    def set_cube_palm(self, pos_palm: np.ndarray, quat_palm_wxyz: np.ndarray) -> None:
        c, R = self.palm_frame()
        Rc = np.zeros(9)
        mujoco.mju_quat2Mat(Rc, np.asarray(quat_palm_wxyz, np.float64))
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (R @ Rc.reshape(3, 3)).reshape(-1))
        a = self.cube_qadr
        self.data.qpos[a:a + 3] = c + R @ np.asarray(pos_palm, np.float64)
        self.data.qpos[a + 3:a + 7] = q
        mujoco.mj_kinematics(self.model, self.data)

    def cube_world(self) -> tuple[np.ndarray, np.ndarray]:
        a = self.cube_qadr
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[a + 3:a + 7])
        return self.data.qpos[a:a + 3].copy(), R.reshape(3, 3)

    def world_to_palm(self, pos_w: np.ndarray, R_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, R = self.palm_frame()
        return R.T @ (pos_w - c), R.T @ R_w

    def tip_positions(self) -> dict[str, np.ndarray]:
        return {f: self.data.geom_xpos[g].copy() for f, g in self.tip_geoms.items()}

    def tip_force_from_torque(self, tau_server: np.ndarray, toward: np.ndarray | None = None,
                              damping: float = 0.005) -> dict[str, float]:
        """Fingertip force implied by joint torques (quasi-static: tau = J^T F).

        For each finger, damped least-squares solve J_tip^T F = tau over its 4 joints
        (F = force the fingertip exerts on the environment, world frame).
        Returns the component along ``toward - tip`` (e.g. the cube centre) if
        given, else |F|.  Ignores gravity, friction and joint dynamics, so it is
        an approximate, independent cross-check, not ground truth.
        """
        m, d = self.model, self.data
        mujoco.mj_comPos(m, d)  # mj_jac needs the COM-based quantities, not just kinematics
        tau = np.asarray(tau_server, np.float64)
        out = {}
        for finger, prefix in (("thumb", "th"), ("index", "if"), ("middle", "mf"), ("ring", "rf")):
            idx = [i for i, n in enumerate(SERVER_JOINT_NAMES) if n.startswith(prefix)]
            dofs = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, SERVER_JOINT_NAMES[i])] for i in idx]
            g = self.tip_geoms[finger]
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jacp, None, d.geom_xpos[g], m.geom_bodyid[g])
            J = jacp[:, dofs]  # (3, 4): tip velocity per joint velocity
            # damped least squares for J^T F = tau: stays bounded when a finger is
            # near-singular (e.g. straight), where plain least squares explodes
            F = np.linalg.solve(J @ J.T + damping ** 2 * np.eye(3), J @ tau[idx])
            if toward is not None:
                u = np.asarray(toward, np.float64) - d.geom_xpos[g]
                n = np.linalg.norm(u)
                out[finger] = float(F @ u / n) if n > 1e-9 else float(np.linalg.norm(F))
            else:
                out[finger] = float(np.linalg.norm(F))
        return out
