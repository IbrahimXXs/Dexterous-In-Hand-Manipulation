#!/usr/bin/env python3
"""Torch-free cube-state estimator for estimator-augmented policies (Condition C).

Loads ``cube_estimator.npz`` (written by the training repo's
``scripts/estimation/export_kit_estimator.py``) and reproduces, in NumPy:

  obs_320 -> (obs - mean) / std -> 3 x [Linear -> LayerNorm -> ELU] -> Linear(19)
          -> cube position, 6D rotation (Gram-Schmidt), linear / angular velocity,
             4 fingertip contact probabilities (palm frame)
          -> 19 policy features (scratch_study/features.py): standardised position,
             canonical rotation (closest of the 24 cube symmetries), standardised
             velocities, contact probabilities

The estimator only reads the same 320 proprioceptive numbers the policy reads.
On load it re-runs the probe batch stored in the file and refuses to work if
its features differ from the training-side features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FORMAT = "leap_kit_cube_estimator/v1"
FEATURE_DIM = 19


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 20.0)))


class CubeEstimatorNP:
    def __init__(self, path: str | Path, check_tol: float = 1e-3):
        d = np.load(path, allow_pickle=False)
        if str(d["format"]) != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        self.path = Path(path)
        self.source = str(d["source"])
        self.source_sha256 = str(d["source_sha256"])
        g = lambda k: np.asarray(d[k], dtype=np.float64)  # noqa: E731
        self.obs_mean, self.obs_std = g("obs_mean"), g("obs_std")
        self.layers = [(g(f"w{k}"), g(f"b{k}"), g(f"ln{k}_w"), g(f"ln{k}_b")) for k in range(3)]
        self.head_w, self.head_b = g("head_w"), g("head_b")
        self.pos_mean, self.pos_std = g("pos_mean"), g("pos_std")
        self.lin_std, self.ang_std = g("lin_vel_std"), g("ang_vel_std")
        self.ln_eps = float(d["ln_eps"])
        self.sym = g("sym")
        self.p_mean, self.p_std = g("feat_p_mean"), g("feat_p_std")
        self.v_std, self.w_std = g("feat_v_std"), g("feat_w_std")
        self.obs_dim = self.obs_mean.shape[0]
        self.last: dict[str, np.ndarray] | None = None
        got = self.features_batch(g("probe_obs"))
        err = float(np.abs(got - g("probe_features")).max())
        if not err < check_tol:
            raise RuntimeError(f"cube estimator self-check failed: max feature error {err:.2e}")
        self.self_check_error = err

    # ------------------------------------------------------------------
    def estimate_batch(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        x = (np.asarray(obs, np.float64) - self.obs_mean) / self.obs_std
        for w, b, lw, lb in self.layers:
            x = x @ w.T + b
            mu = x.mean(-1, keepdims=True)
            var = x.var(-1, keepdims=True)
            x = _elu((x - mu) / np.sqrt(var + self.ln_eps) * lw + lb)
        y = x @ self.head_w.T + self.head_b
        a1, a2 = y[:, 3:6], y[:, 6:9]
        b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
        a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
        b2 = a2 / np.maximum(np.linalg.norm(a2, axis=-1, keepdims=True), 1e-8)
        R = np.stack([b1, b2, np.cross(b1, b2)], axis=-1)  # columns
        return {
            "pos": y[:, 0:3] * self.pos_std + self.pos_mean,
            "rotmat": R,
            "lin_vel": y[:, 9:12] * self.lin_std,
            "ang_vel": y[:, 12:15] * self.ang_std,
            "contact_prob": 1.0 / (1.0 + np.exp(-y[:, 15:19])),
        }

    def features_batch(self, obs: np.ndarray) -> np.ndarray:
        return self._features_from(self.estimate_batch(obs))

    def _features_from(self, e: dict[str, np.ndarray]) -> np.ndarray:
        R = e["rotmat"]
        tr = np.einsum("nij,kji->nk", R, self.sym)
        Rc = R @ self.sym[tr.argmax(-1)]
        return np.concatenate(
            [
                np.clip((e["pos"] - self.p_mean) / self.p_std, -5, 5),
                Rc[:, :, 0],
                Rc[:, :, 1],
                np.clip(e["lin_vel"] / self.v_std, -5, 5),
                np.clip(e["ang_vel"] / self.w_std, -5, 5),
                e["contact_prob"],
            ],
            axis=-1,
        )

    def features(self, obs_320: np.ndarray) -> np.ndarray:
        """One observation -> 19 features (float32); keeps ``self.last`` for logging."""
        obs = np.asarray(obs_320, np.float64).reshape(1, -1)
        e = self.estimate_batch(obs)
        self.last = {k: v[0].astype(np.float32) for k, v in e.items()}
        return self._features_from(e)[0].astype(np.float32)
