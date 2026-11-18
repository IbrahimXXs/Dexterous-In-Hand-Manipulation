#!/usr/bin/env python3
"""Hand monitor: cube-state and fingertip-force estimates from proprioception.

Runs next to ANY policy (A, C, model_400) on the same 320 numbers the policy
reads (10 frames of joint positions relative to the default pose + 10 frames
of the previous applied commands).  It never feeds anything back to the policy
or the motors; for C the cube estimator is the same network the actor uses.

  * cube estimator B (``cube_estimator.npz``): cube position, orientation,
    linear / angular velocity and fingertip-contact probabilities, palm frame
  * force estimator (``force_estimator.npz``): normal force between each
    fingertip and the cube [N] (thumb, index, middle, ring) and, for the hurdle
    head, the probability of contact

Orientation: a cube looks the same under its 24 rotational symmetries, so the
estimator may answer with any of them.  ``HandMonitor`` keeps the displayed
orientation continuous by picking, every step, the equivalent rotation closest
to the previous one, and integrates the resulting yaw about the palm normal
(an *estimated* cumulative rotation; it accumulates estimator error).

Both estimator files carry a probe batch with training-side outputs and are
re-checked on load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .cube_estimator import CubeEstimatorNP, _elu
except ImportError:
    from cube_estimator import CubeEstimatorNP, _elu

FORCE_FORMAT = "leap_kit_force_estimator/v1"
NUM_JOINTS = 16
HISTORY = 10
# Policy default pose (server order); identical for model_400 and all study policies.
DEFAULT_JOINT_POS = np.array(
    [0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0, 0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37], dtype=np.float32
)
FINGERS = ("thumb", "index", "middle", "ring")  # force-estimator order
CONTACT_ORDER_B = ("index", "middle", "ring", "thumb")  # cube-estimator contact order


class ForceEstimatorNP:
    def __init__(self, path: str | Path, check_tol: float = 1e-3):
        d = np.load(path, allow_pickle=False)
        if str(d["format"]) != FORCE_FORMAT:
            raise ValueError(f"{path} is not a {FORCE_FORMAT} file")
        self.path = Path(path)
        self.head = str(d["head"])
        self.source = str(d["source"])
        self.source_sha256 = str(d["source_sha256"])
        if tuple(str(x) for x in d["finger_order"]) != FINGERS:
            raise ValueError(f"unexpected finger order {d['finger_order']}")
        g = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
        self.obs_mean, self.obs_std, self.scale = g("obs_mean"), g("obs_std"), g("force_scale")
        self.layers = [(g(f"w{k}"), g(f"b{k}"), g(f"ln{k}_w"), g(f"ln{k}_b")) for k in range(3)]
        self.head_w, self.head_b = g("head_w"), g("head_b")
        self.ln_eps = float(d["ln_eps"])
        f, pc = self.predict_batch(g("probe_obs"))
        err = float(np.abs(f - g("probe_forces")).max())
        if not err < check_tol:
            raise RuntimeError(f"force estimator self-check failed: max force error {err:.2e} N")
        self.self_check_error = err

    def predict_batch(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[N, 320] -> forces [N, 4] (N, newtons), contact probability [N, 4] (NaN for 'direct')."""
        x = (np.asarray(obs, np.float64) - self.obs_mean) / self.obs_std
        for w, b, lw, lb in self.layers:
            x = x @ w.T + b
            mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
            x = _elu((x - mu) / np.sqrt(var + self.ln_eps) * lw + lb)
        z = x @ self.head_w.T + self.head_b
        if self.head == "direct":
            return np.logaddexp(0.0, z) * self.scale, np.full_like(z, np.nan)
        p = 1.0 / (1.0 + np.exp(-z[:, :4]))
        return p * np.logaddexp(0.0, z[:, 4:]) * self.scale, p


def _rotvec(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(ang)
    if s < 1e-6:
        return 0.5 * v
    return ang / (2.0 * s) * v


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = R
    w = 0.5 * np.sqrt(max(0.0, 1 + m[0, 0] + m[1, 1] + m[2, 2]))
    x = np.copysign(0.5 * np.sqrt(max(0.0, 1 + m[0, 0] - m[1, 1] - m[2, 2])), m[2, 1] - m[1, 2])
    y = np.copysign(0.5 * np.sqrt(max(0.0, 1 - m[0, 0] + m[1, 1] - m[2, 2])), m[0, 2] - m[2, 0])
    z = np.copysign(0.5 * np.sqrt(max(0.0, 1 - m[0, 0] - m[1, 1] + m[2, 2])), m[1, 0] - m[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class SymmetryUnwrapper:
    """Continuous cube orientation from symmetry-ambiguous estimates.

    Each step picks, among the 24 equivalent rotations R S, the one closest to
    the previous output, and integrates the rotation about ``yaw_axis``.
    """

    def __init__(self, sym: np.ndarray, yaw_axis: np.ndarray):
        self.sym = np.asarray(sym, np.float64)
        self.yaw_axis = np.asarray(yaw_axis, np.float64)
        self.reset()

    def reset(self) -> None:
        self.R_prev: np.ndarray | None = None
        self.yaw = 0.0

    def step(self, R: np.ndarray) -> np.ndarray:
        if self.R_prev is None:
            tr = np.einsum("ij,kji->k", R, self.sym)  # first output: closest to the palm frame
        else:
            tr = np.einsum("ij,kji->k", self.R_prev.T @ R, self.sym)
        R_out = R @ self.sym[int(np.argmax(tr))]
        if self.R_prev is not None:
            self.yaw += float(_rotvec(R_out @ self.R_prev.T) @ self.yaw_axis)
        self.R_prev = R_out
        return R_out


class HandMonitor:
    """Estimators + continuity bookkeeping.  ``update(obs_320)`` once per policy step."""

    # Palm -z is world +z for the palm-up mount used in training (palm quat 0 1 0 0),
    # so a positive world-yaw turn is a negative rotation about the palm z axis.
    YAW_AXIS_PALM = np.array([0.0, 0.0, -1.0])

    def __init__(self, monitor_dir: str | Path, cube_estimator: CubeEstimatorNP | None = None):
        monitor_dir = Path(monitor_dir)
        self.cube = cube_estimator or CubeEstimatorNP(monitor_dir / "cube_estimator.npz")
        self.force = ForceEstimatorNP(monitor_dir / "force_estimator.npz")
        self.reset()

    def describe(self) -> str:
        return (f"cube estimator {self.cube.source} (sha256 {self.cube.source_sha256[:12]}...), "
                f"force estimator {self.force.source} ({self.force.head} head, sha256 "
                f"{self.force.source_sha256[:12]}...)")

    def reset(self) -> None:
        self.unwrap = SymmetryUnwrapper(self.cube.sym, self.YAW_AXIS_PALM)
        self._q_hist = None
        self._c_hist = None

    # --- own history (offline replay of logs; live runs pass the policy's obs) ---
    def push(self, q_server: np.ndarray, prev_cmd_server: np.ndarray, reset: bool = False) -> np.ndarray:
        q_rel = np.asarray(q_server, np.float32) - DEFAULT_JOINT_POS
        c = np.asarray(prev_cmd_server, np.float32)
        if reset or self._q_hist is None:
            self._q_hist = np.repeat(q_rel[None], HISTORY, 0)
            self._c_hist = np.repeat(c[None], HISTORY, 0)
        else:
            self._q_hist = np.concatenate([self._q_hist[1:], q_rel[None]])
            self._c_hist = np.concatenate([self._c_hist[1:], c[None]])
        return np.concatenate([self._q_hist.reshape(-1), self._c_hist.reshape(-1)])

    def update(self, obs_320: np.ndarray) -> dict[str, np.ndarray | float]:
        obs = np.asarray(obs_320, np.float64).reshape(1, -1)
        e = self.cube.estimate_batch(obs)
        R_disp = self.unwrap.step(e["rotmat"][0])
        forces, pc = self.force.predict_batch(obs)
        return {
            "cube_pos_palm": e["pos"][0].astype(np.float32),
            "cube_quat_palm": mat_to_quat_wxyz(R_disp).astype(np.float32),
            "cube_lin_vel_palm": e["lin_vel"][0].astype(np.float32),
            "cube_ang_vel_palm": e["ang_vel"][0].astype(np.float32),
            "contact_prob_b": e["contact_prob"][0].astype(np.float32),  # if, mf, rf, th
            "force_n": forces[0].astype(np.float32),  # thumb, index, middle, ring
            "force_contact_prob": pc[0].astype(np.float32),
            "est_yaw_rad": np.float32(self.unwrap.yaw),
        }


MONITOR_KEYS = ("cube_pos_palm", "cube_quat_palm", "cube_lin_vel_palm", "cube_ang_vel_palm",
                "contact_prob_b", "force_n", "force_contact_prob", "est_yaw_rad")
