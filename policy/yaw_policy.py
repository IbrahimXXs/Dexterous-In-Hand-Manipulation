#!/usr/bin/env python3
"""Continuous Z-rotation (yaw) policy runner for the right LEAP hand.

Copied and unified from CONTINUOUS_Z_ROTATION_DEPLOYMENT_BUNDLE_20261014.
Implements the 320-dim history actor policy (10 frames x 32 terms:
joint_pos_rel + prev_commanded_joint_pos_rel) with normalized action
mapping to per-step joint position deltas.

Can run with pure NumPy (zero ML dependencies) or with PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

# Server joint order:
# ('if_mcp', 'if_rot', 'if_pip', 'if_dip',
#  'mf_mcp', 'mf_rot', 'mf_pip', 'mf_dip',
#  'rf_mcp', 'rf_rot', 'rf_pip', 'rf_dip',
#  'th_cmc', 'th_axl', 'th_mcp', 'th_ipl')

_NUM_JOINTS = 16
_HISTORY_LEN = 10
_DELTA_RANGE = 0.041666666666666664  # 1/24 rad = 2.38 deg max per step

_DEFAULT_JOINT_POS = np.array(
    [0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0, 0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37],
    dtype=np.float32,
)

_LOWER_LIMITS = np.array(
    [-0.314, -1.047, -0.506, -0.366, -0.314, -1.047, -0.506, -0.366,
     -0.314, -1.047, -0.506, -0.366, -0.349, -0.349, -0.470, -1.340],
    dtype=np.float32,
)

_UPPER_LIMITS = np.array(
    [2.230, 1.047, 1.885, 2.042, 2.230, 1.047, 1.885, 2.042,
     2.230, 1.047, 1.885, 2.042, 2.094, 2.094, 2.443, 1.880],
    dtype=np.float32,
)

# Hardware order swaps rot/mcp for index, middle, ring fingers:
_HW_TO_SERVER = np.array(
    [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15],
    dtype=np.int64,
)
_SERVER_TO_HW = np.argsort(_HW_TO_SERVER).astype(np.int64)


def hw_to_server(q_hw: np.ndarray) -> np.ndarray:
    """Convert hardware joint vector to server/policy order."""
    return np.asarray(q_hw, dtype=np.float32)[_HW_TO_SERVER].copy()


def server_to_hw(q_server: np.ndarray) -> np.ndarray:
    """Convert server/policy joint vector to hardware order."""
    return np.asarray(q_server, dtype=np.float32)[_SERVER_TO_HW].copy()


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.exp(np.minimum(x, 20.0)) - 1.0)


class ContinuousYawPolicy:
    """In-hand continuous Z-rotation (yaw) policy runner."""

    def __init__(self, checkpoint_or_weights_path: str | Path | None = None):
        if checkpoint_or_weights_path is None:
            root = Path(__file__).resolve().parents[1]
            npz_path = root / "checkpoint" / "policy_weights.npz"
            pt_path = root / "checkpoint" / "model_400.pt"
            if npz_path.is_file():
                checkpoint_or_weights_path = npz_path
            elif pt_path.is_file():
                checkpoint_or_weights_path = pt_path
            else:
                raise FileNotFoundError(f"Neither {npz_path} nor {pt_path} exists.")

        self.path = Path(checkpoint_or_weights_path)
        self.default_joint_pos = _DEFAULT_JOINT_POS.copy()
        self.lower_limits = _LOWER_LIMITS.copy()
        self.upper_limits = _UPPER_LIMITS.copy()
        self.delta_range = _DELTA_RANGE

        self._load_weights(self.path)

        # History circular buffers: shape (10, 16) each
        self._q_rel_history = np.zeros((_HISTORY_LEN, _NUM_JOINTS), dtype=np.float32)
        self._cmd_history = np.zeros((_HISTORY_LEN, _NUM_JOINTS), dtype=np.float32)
        self._pointer = -1
        self._num_pushes = 0
        self._prev_command = self.default_joint_pos.copy()

    def _load_weights(self, path: Path) -> None:
        if path.suffix == ".npz":
            data = np.load(path)
            self._mean = data["mean"].reshape(1, 320).astype(np.float32)
            self._std = data["std"].reshape(1, 320).astype(np.float32)
            self._eps = 1e-8
            self._w0 = data["w0"].astype(np.float32)
            self._b0 = data["b0"].astype(np.float32)
            self._w2 = data["w2"].astype(np.float32)
            self._b2 = data["b2"].astype(np.float32)
            self._w4 = data["w4"].astype(np.float32)
            self._b4 = data["b4"].astype(np.float32)
            self._w6 = data["w6"].astype(np.float32)
            self._b6 = data["b6"].astype(np.float32)
        elif path.suffix == ".pt":
            import torch
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            sd = ckpt["actor_state_dict"]
            self._mean = sd["obs_normalizer._mean"].numpy().reshape(1, 320).astype(np.float32)
            self._std = sd["obs_normalizer._std"].numpy().reshape(1, 320).astype(np.float32)
            self._eps = 1e-8
            self._w0 = sd["mlp.0.weight"].numpy().astype(np.float32)
            self._b0 = sd["mlp.0.bias"].numpy().astype(np.float32)
            self._w2 = sd["mlp.2.weight"].numpy().astype(np.float32)
            self._b2 = sd["mlp.2.bias"].numpy().astype(np.float32)
            self._w4 = sd["mlp.4.weight"].numpy().astype(np.float32)
            self._b4 = sd["mlp.4.bias"].numpy().astype(np.float32)
            self._w6 = sd["mlp.6.weight"].numpy().astype(np.float32)
            self._b6 = sd["mlp.6.bias"].numpy().astype(np.float32)
        else:
            raise ValueError(f"Unsupported policy format: {path}")

    def reset(self, initial_q_server: np.ndarray | None = None) -> np.ndarray:
        """Reset observation history buffer and setpoint."""
        if initial_q_server is None:
            initial_q_server = self.default_joint_pos.copy()
        else:
            initial_q_server = np.asarray(initial_q_server, dtype=np.float32).copy()

        self._prev_command = np.clip(initial_q_server, self.lower_limits, self.upper_limits)
        q_rel = (self._prev_command - self.default_joint_pos).reshape(1, _NUM_JOINTS)

        # Backfill history with initial pose
        self._q_rel_history[:] = q_rel
        self._cmd_history[:] = self._prev_command.reshape(1, _NUM_JOINTS)
        self._pointer = _HISTORY_LEN - 1
        self._num_pushes = _HISTORY_LEN
        return self._prev_command.copy()

    def _get_chronological_obs(self) -> np.ndarray:
        """Retrieve flattened 320-dim observation in oldest->newest order."""
        start = (self._pointer + 1) % _HISTORY_LEN
        idx = (np.arange(_HISTORY_LEN) + start) % _HISTORY_LEN
        q_chrono = self._q_rel_history[idx].reshape(1, -1)
        cmd_chrono = self._cmd_history[idx].reshape(1, -1)
        return np.concatenate([q_chrono, cmd_chrono], axis=-1)

    def _forward(self, obs_320: np.ndarray) -> np.ndarray:
        """Run policy MLP."""
        norm_obs = (obs_320 - self._mean) / (self._std + self._eps)
        h = _elu(norm_obs @ self._w0.T + self._b0)
        h = _elu(h @ self._w2.T + self._b2)
        h = _elu(h @ self._w4.T + self._b4)
        out = h @ self._w6.T + self._b6
        return out.reshape(-1)

    def step(self, current_q_server: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Execute one policy step at 20 Hz.

        Args:
            current_q_server: current measured joint angles in server order (16,)

        Returns:
            new_target_server: next joint position targets (16,)
            raw_action: unclipped policy network output (16,)
            delta: applied joint position delta (16,)
        """
        q = np.asarray(current_q_server, dtype=np.float32)
        q_rel = q - self.default_joint_pos
        cmd_abs = self._prev_command

        self._pointer = (self._pointer + 1) % _HISTORY_LEN
        self._q_rel_history[self._pointer] = q_rel
        self._cmd_history[self._pointer] = cmd_abs
        self._num_pushes += 1

        obs_320 = self._get_chronological_obs()
        raw_action = self._forward(obs_320)
        clipped_action = np.clip(raw_action, -1.0, 1.0)

        delta = clipped_action * self.delta_range
        new_target = np.clip(self._prev_command + delta, self.lower_limits, self.upper_limits)
        self._prev_command = new_target.copy()
        return new_target, raw_action, delta
