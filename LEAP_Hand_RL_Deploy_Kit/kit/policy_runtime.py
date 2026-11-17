#!/usr/bin/env python3
"""Torch-free runtime for MJLab / rsl_rl LEAP-hand policies.

Loads an rsl_rl checkpoint (``model_*.pt``) WITHOUT needing PyTorch, reads the
training ``env.yaml`` / ``agent.yaml`` (if supplied) WITHOUT needing PyYAML, and
reproduces the actor's deterministic inference in NumPy:

    obs  -> (obs - mean) / (std + 1e-2)        rsl_rl EmpiricalNormalization
         -> MLP (hidden layers + activation)   rsl_rl MLP
         -> action (mean, no sampling)
    action -> clip [-1, 1] -> joint target     MJLab action term
              (tanh instead of clip when env.yaml declares ``squash: tanh``)

Supported actor observation terms (each 16-D, MJLab right-hand joint order):
    joint_pos                 measured q - default q          (joint_pos_rel)
    joint_vel                 measured qd                     (joint_vel_rel)
    prev_commanded_joint_pos  previous absolute command
    joint_pos_error           previous command - measured q
    last_action               previous raw network output
Policies that need cube pose, camera, goal, or tactile inputs are rejected with
an explicit message rather than silently fed zeros.
"""

from __future__ import annotations

import io
import json
import pickle
import re
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

NUM_JOINTS = 16

# MJLab / MJCF right-hand joint order used by the policy ("server order").
SERVER_JOINT_NAMES = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)

# Defaults of the reference task Mjlab-Leap-Right-HandCube-MBZUAI-Rotate.
REFERENCE_DEFAULT_JOINT_POS = np.array(
    [0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0,
     0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37],
    dtype=np.float32,
)
# MJLab soft joint limits (server order) of the right-hand model.
REFERENCE_LOWER_LIMITS = np.array(
    [-0.314, -1.047, -0.506, -0.366, -0.314, -1.047, -0.506, -0.366,
     -0.314, -1.047, -0.506, -0.366, -0.349, -0.349, -0.470, -1.340],
    dtype=np.float32,
)
REFERENCE_UPPER_LIMITS = np.array(
    [2.230, 1.047, 1.885, 2.042, 2.230, 1.047, 1.885, 2.042,
     2.230, 1.047, 1.885, 2.042, 2.094, 2.094, 2.443, 1.880],
    dtype=np.float32,
)

SUPPORTED_TERMS = (
    "joint_pos",
    "joint_vel",
    "prev_commanded_joint_pos",
    "joint_pos_error",
    "last_action",
)

RSL_RL_NORMALIZER_EPS = 1e-2  # rsl_rl.modules.EmpiricalNormalization default


# --------------------------------------------------------------------------
# 1. Torch-free .pt reader
# --------------------------------------------------------------------------

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "BFloat16Storage": "bfloat16",
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name: str):
        self.name = name


class _Opaque:
    """Stand-in for any non-tensor object we do not need (optimizer, infos)."""

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def __setstate__(self, state: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> "_Opaque":
        return _Opaque()


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *args, **kwargs):
    flat = storage
    if not size:
        return flat[storage_offset].copy()
    itemsize = flat.dtype.itemsize
    arr = np.lib.stride_tricks.as_strided(
        flat[storage_offset:],
        shape=tuple(size),
        strides=tuple(s * itemsize for s in stride),
    )
    return np.array(arr, copy=True)


def _rebuild_parameter(data, requires_grad=False, backward_hooks=None, *args):
    return data


def load_torch_checkpoint_without_torch(path: str | Path) -> dict[str, Any]:
    """Read a modern (zip) torch.save file into nested dicts of numpy arrays."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkl_name = next((n for n in names if n.endswith("data.pkl")), None)
        if pkl_name is None:
            raise ValueError(f"{path} is not a zip-format torch checkpoint")
        prefix = pkl_name[: -len("data.pkl")]
        byteorder = "little"
        if f"{prefix}byteorder" in names:
            byteorder = zf.read(f"{prefix}byteorder").decode().strip()
        storages: dict[str, np.ndarray] = {}

        def load_storage(storage_type: Any, key: str) -> np.ndarray:
            if key in storages:
                return storages[key]
            raw = zf.read(f"{prefix}data/{key}")
            name = getattr(storage_type, "name", str(storage_type))
            dtype = _STORAGE_DTYPES.get(name)
            if dtype is None:
                raise ValueError(f"Unsupported tensor storage type {name}")
            if dtype == "bfloat16":
                u16 = np.frombuffer(raw, dtype="<u2" if byteorder == "little" else ">u2")
                arr = (u16.astype(np.uint32) << 16).view(np.float32)
            else:
                dt = np.dtype(dtype).newbyteorder("<" if byteorder == "little" else ">")
                arr = np.frombuffer(raw, dtype=dt).astype(np.dtype(dtype))
            storages[key] = arr
            return arr

        class _Unpickler(pickle.Unpickler):
            # Restricted: never imports or executes arbitrary code from the file.
            def find_class(self, module: str, name: str) -> Any:
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return _rebuild_tensor_v2
                if module == "torch._utils" and name == "_rebuild_parameter":
                    return _rebuild_parameter
                if module == "torch" and name.endswith("Storage"):
                    return _StorageType(name)
                if module == "builtins" and name in ("set", "frozenset", "slice", "complex"):
                    return {"set": set, "frozenset": frozenset, "slice": slice, "complex": complex}[name]
                return _Opaque

            def persistent_load(self, pid: Any) -> Any:
                kind = pid[0] if isinstance(pid, tuple) else pid
                if kind != "storage":
                    raise pickle.UnpicklingError(f"Unknown persistent id {pid!r}")
                _, storage_type, key, _location, _numel = pid
                return load_storage(storage_type, str(key))

        return _Unpickler(io.BytesIO(zf.read(pkl_name))).load()


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        return load_torch_checkpoint_without_torch(path)
    except Exception as first_error:  # legacy format: fall back to torch
        try:
            import torch  # type: ignore
        except ImportError:
            raise RuntimeError(
                f"Could not read {path} without torch ({first_error}); "
                "install torch or re-save the checkpoint with a modern torch."
            ) from first_error
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

        def conv(x: Any) -> Any:
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            return x

        return conv(ckpt)


# --------------------------------------------------------------------------
# 2. Minimal env.yaml / agent.yaml readers (no PyYAML needed)
# --------------------------------------------------------------------------

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block(lines: list[str], start: int) -> list[str]:
    """Lines belonging to the mapping that starts at lines[start]."""
    base = _indent(lines[start])
    out = []
    for line in lines[start + 1:]:
        if line.strip() and _indent(line) <= base:
            break
        out.append(line)
    return out


@dataclass
class TaskSpec:
    obs_terms: list[str] = field(default_factory=lambda: ["joint_pos", "prev_commanded_joint_pos"])
    obs_term_scales: dict[str, float] = field(default_factory=dict)
    history_length: int = 10
    action_mode: str = "delta"  # "delta" or "absolute"
    delta_min: float = -1.0 / 24.0
    delta_max: float = 1.0 / 24.0
    action_scale: float = 1.0
    action_use_default_offset: bool = True
    clip_actions: float = 1.0
    action_squash: str = "clip"  # "clip" (MJLab default) or "tanh" (env applies tanh(u))
    default_joint_pos: np.ndarray = field(default_factory=lambda: REFERENCE_DEFAULT_JOINT_POS.copy())
    lower_limits: np.ndarray = field(default_factory=lambda: REFERENCE_LOWER_LIMITS.copy())
    upper_limits: np.ndarray = field(default_factory=lambda: REFERENCE_UPPER_LIMITS.copy())
    policy_hz: float = 20.0
    activation: str = "elu"
    source: str = "built-in reference defaults"
    notes: list[str] = field(default_factory=list)


def parse_env_yaml(path: Path, spec: TaskSpec) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    spec.source = str(path)

    # decimation * sim timestep -> policy rate
    dec = next((float(m.group(1)) for l in lines if (m := re.match(rf"^decimation:\s*({_NUM})", l))), None)
    ts = next((float(m.group(1)) for l in lines if (m := re.match(rf"^\s+timestep:\s*({_NUM})", l))), None)
    if dec and ts:
        spec.policy_hz = 1.0 / (dec * ts)

    # observations.actor
    try:
        obs_i = next(i for i, l in enumerate(lines) if l.rstrip() == "observations:")
        obs_block = _block(lines, obs_i)
        a_rel = next(i for i, l in enumerate(obs_block) if l.strip() == "actor:")
        actor = _block(obs_block, a_rel)
        actor_indent = _indent(obs_block[a_rel])
        t_rel = next(i for i, l in enumerate(actor) if l.strip() == "terms:")
        terms_block = _block(actor, t_rel)
        term_indent = _indent(actor[t_rel]) + 2
        names = []
        for j, l in enumerate(terms_block):
            m = re.match(r"^\s*([A-Za-z0-9_]+):\s*$", l)
            if m and _indent(l) == term_indent:
                names.append(m.group(1))
                body = _block(terms_block, j)
                for bl in body:
                    sm = re.match(rf"^\s*scale:\s*({_NUM})\s*$", bl)
                    if sm and _indent(bl) == term_indent + 2:
                        spec.obs_term_scales[m.group(1)] = float(sm.group(1))
        if names:
            spec.obs_terms = names
        for l in actor:
            hm = re.match(r"^\s*history_length:\s*(\d+)", l)
            if hm and _indent(l) == actor_indent + 2:
                spec.history_length = max(1, int(hm.group(1)))
    except StopIteration:
        spec.notes.append("env.yaml: could not locate observations.actor; using defaults")

    # actions.joint_pos
    try:
        act_i = next(i for i, l in enumerate(lines) if l.rstrip() == "actions:")
        act = "\n".join(_block(lines, act_i))
        dmin = re.search(rf"delta_min:\s*({_NUM})", act)
        dmax = re.search(rf"delta_max:\s*({_NUM})", act)
        sq = re.search(r"squash:\s*([A-Za-z_]+)", act)
        if sq:
            if sq.group(1) not in ("tanh", "clip"):
                raise ValueError(f"env.yaml: unsupported action squash {sq.group(1)!r}")
            spec.action_squash = sq.group(1)
        if dmin and dmax:
            spec.action_mode = "delta"
            spec.delta_min, spec.delta_max = float(dmin.group(1)), float(dmax.group(1))
        else:
            spec.action_mode = "absolute"
            sc = re.search(rf"\n\s+scale:\s*({_NUM})", "\n" + act)
            spec.action_scale = float(sc.group(1)) if sc else 1.0
            off = re.search(r"use_default_offset:\s*(true|false)", act)
            spec.action_use_default_offset = (off.group(1) == "true") if off else True
    except StopIteration:
        spec.notes.append("env.yaml: no actions block; assuming delta action")

    # default joint pose = robot init_state.joint_pos
    for i, l in enumerate(lines):
        if l.strip() == "joint_pos:" and i > 0:
            vals = {}
            for bl in _block(lines, i):
                m = re.match(rf"^\s*([a-z]{{2}}_[a-z]+):\s*({_NUM})\s*$", bl)
                if m:
                    vals[m.group(1)] = float(m.group(2))
            if all(n in vals for n in SERVER_JOINT_NAMES):
                spec.default_joint_pos = np.array([vals[n] for n in SERVER_JOINT_NAMES], dtype=np.float32)
                break
    else:
        spec.notes.append("env.yaml: default joint_pos not found; using reference default pose")


def parse_agent_yaml(path: Path, spec: TaskSpec) -> None:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^actor:\s*$(.*?)^\S", text + "\nX", re.S | re.M)
    scope = m.group(1) if m else text
    a = re.search(r"activation:\s*([A-Za-z_]+)", scope)
    if a:
        spec.activation = a.group(1).lower()
    c = re.search(rf"^clip_actions:\s*({_NUM}|null)", text, re.M)
    if c:
        # null = the env receives the raw action (e.g. it squashes with tanh itself)
        spec.clip_actions = 0.0 if c.group(1) == "null" else float(c.group(1))


# --------------------------------------------------------------------------
# 3. Actor network
# --------------------------------------------------------------------------

def _act(name: str):
    if name == "elu":
        return lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 20.0)))
    if name == "relu":
        return lambda x: np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh
    if name == "selu":
        a, s = 1.6732632423543772, 1.0507009873554805
        return lambda x: s * np.where(x > 0, x, a * np.expm1(np.minimum(x, 20.0)))
    if name in ("lrelu", "leaky_relu"):
        return lambda x: np.where(x > 0, x, 0.01 * x)
    if name == "sigmoid":
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    if name in ("silu", "swish"):
        return lambda x: x / (1.0 + np.exp(-x))
    if name == "gelu":
        return lambda x: 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))
    if name == "softplus":
        return lambda x: np.logaddexp(0.0, x)
    raise ValueError(f"Unsupported activation '{name}'")


class ActorMLP:
    def __init__(self, ckpt: dict[str, Any], activation: str):
        sd, prefix, norm = self._find_actor(ckpt)
        idx = sorted(
            {int(m.group(1)) for k in sd if (m := re.match(rf"^{re.escape(prefix)}(\d+)\.weight$", k))}
        )
        if not idx:
            raise ValueError("No MLP weights found in checkpoint")
        self.weights = [np.asarray(sd[f"{prefix}{i}.weight"], dtype=np.float32) for i in idx]
        self.biases = [np.asarray(sd[f"{prefix}{i}.bias"], dtype=np.float32) for i in idx]
        self.obs_dim = self.weights[0].shape[1]
        self.act_dim = self.weights[-1].shape[0]
        self.hidden = [w.shape[0] for w in self.weights[:-1]]
        self.activation = activation
        self._f = _act(activation)
        if norm is not None:
            self.mean = np.asarray(norm[0], dtype=np.float32).reshape(-1)
            self.std = np.asarray(norm[1], dtype=np.float32).reshape(-1)
        else:
            self.mean = None
            self.std = None

    @staticmethod
    def _find_actor(ckpt: dict[str, Any]):
        # rsl_rl >= 3 (MJLab): actor_state_dict with mlp.N.* and obs_normalizer.*
        if isinstance(ckpt.get("actor_state_dict"), dict):
            sd = ckpt["actor_state_dict"]
            norm = None
            if "obs_normalizer._mean" in sd:
                std = sd.get("obs_normalizer._std")
                if std is None:
                    std = np.sqrt(sd["obs_normalizer._var"])
                norm = (sd["obs_normalizer._mean"], std)
            prefix = "mlp." if any(k.startswith("mlp.") for k in sd) else "actor."
            return sd, prefix, norm
        # rsl_rl 1.x / 2.x: model_state_dict with actor.N.* (+ obs_norm_state_dict)
        if isinstance(ckpt.get("model_state_dict"), dict):
            sd = ckpt["model_state_dict"]
            norm = None
            ons = ckpt.get("obs_norm_state_dict")
            if isinstance(ons, dict) and "_mean" in ons:
                norm = (ons["_mean"], ons.get("_std", np.sqrt(ons.get("_var", 1.0))))
            elif "actor_obs_normalizer._mean" in sd:
                norm = (sd["actor_obs_normalizer._mean"], sd["actor_obs_normalizer._std"])
            return sd, "actor.", norm
        raise ValueError(
            "Unrecognised checkpoint layout (expected rsl_rl 'actor_state_dict' "
            "or 'model_state_dict')."
        )

    def __call__(self, obs: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.mean is not None:
            x = (x - self.mean) / (self.std + RSL_RL_NORMALIZER_EPS)
        if extra is not None:
            # estimator features are appended AFTER the observation normaliser
            # (scratch_study/actors.py), so they are not normalised here either
            x = np.concatenate([x, np.asarray(extra, dtype=np.float32).reshape(-1)])
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = self._f(x @ w.T + b)
        return x @ self.weights[-1].T + self.biases[-1]


# --------------------------------------------------------------------------
# 4. Policy: history buffer + action mapping
# --------------------------------------------------------------------------

def find_policy_files(folder: Path) -> tuple[Path, Path | None, Path | None]:
    folder = Path(folder)
    if folder.is_file():
        ckpt, folder = folder, folder.parent
    else:
        cands = sorted(folder.rglob("*.pt"))
        if not cands:
            raise FileNotFoundError(
                f"No .pt checkpoint found in {folder}. Copy your model_XXXX.pt there."
            )

        def it(p: Path) -> int:
            m = re.search(r"(\d+)", p.stem)
            return int(m.group(1)) if m else -1

        ckpt = max(cands, key=lambda p: (it(p), p.stat().st_mtime))
    def near(name: str) -> Path | None:
        # Prefer the yaml that sits with this checkpoint (rsl_rl writes params/ next to it).
        for d in (ckpt.parent, ckpt.parent / "params", ckpt.parent.parent / "params"):
            if (d / name).is_file():
                return d / name
        return next(iter(sorted(folder.rglob(name))), None)

    return ckpt, near("env.yaml"), near("agent.yaml")


class LeapPolicy:
    """Stateful 20 Hz policy: give measured q (server order), get joint targets."""

    def __init__(self, policy_path: str | Path):
        ckpt_path, env_yaml, agent_yaml = find_policy_files(Path(policy_path))
        self.checkpoint_path = ckpt_path
        self.spec = TaskSpec()
        if env_yaml is not None:
            parse_env_yaml(env_yaml, self.spec)
        else:
            self.spec.notes.append(
                "No env.yaml found next to the checkpoint: assuming the reference "
                "task (joint_pos + prev_commanded_joint_pos, history 10, delta 1/24 rad, 20 Hz)."
            )
        if agent_yaml is not None:
            parse_agent_yaml(agent_yaml, self.spec)
        ckpt = load_checkpoint(ckpt_path)
        self.iteration = ckpt.get("iter") if isinstance(ckpt, dict) else None
        self.actor = ActorMLP(ckpt, self.spec.activation)
        # Estimator-augmented policy (cube_estimator.npz next to the checkpoint)
        self.estimator = None
        for d in (ckpt_path.parent, ckpt_path.parent / "params"):
            if (d / "cube_estimator.npz").is_file():
                from cube_estimator import CubeEstimatorNP

                self.estimator = CubeEstimatorNP(d / "cube_estimator.npz")
                break
        self._validate()
        self.default_joint_pos = self.spec.default_joint_pos.astype(np.float32)
        self.lower_limits = self.spec.lower_limits
        self.upper_limits = self.spec.upper_limits
        H = self.spec.history_length
        self._hist = {t: np.zeros((H, NUM_JOINTS), np.float32) for t in self.spec.obs_terms}
        self._prev_command = self.default_joint_pos.copy()
        self._last_action = np.zeros(NUM_JOINTS, np.float32)
        self.reset()

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        bad = [t for t in self.spec.obs_terms if t not in SUPPORTED_TERMS]
        if bad:
            raise ValueError(
                f"This policy observes {bad}, which a bare LEAP hand cannot provide "
                f"(only {list(SUPPORTED_TERMS)} are available from the motors). "
                "Camera / cube-pose / goal policies need a perception pipeline."
            )
        expected = self.spec.history_length * NUM_JOINTS * len(self.spec.obs_terms)
        if self.estimator is not None:
            if self.estimator.obs_dim != expected:
                raise ValueError(
                    f"cube_estimator.npz expects {self.estimator.obs_dim} inputs, the task gives {expected}"
                )
            if self.actor.mean is not None and self.actor.mean.shape[0] != expected:
                raise ValueError("Observation normalizer size does not match the proprioceptive input")
            expected += 19
        elif self.actor.obs_dim == expected + 19:
            raise ValueError(
                "This checkpoint expects 19 extra cube-state features (estimator-augmented "
                "policy): put its cube_estimator.npz next to the .pt file."
            )
        if self.actor.obs_dim != expected:
            raise ValueError(
                f"Checkpoint expects {self.actor.obs_dim} inputs but the task spec gives "
                f"{expected} = history {self.spec.history_length} x {len(self.spec.obs_terms)} "
                f"terms {self.spec.obs_terms} x 16 joints. Put the run's params/env.yaml "
                "next to the .pt file."
            )
        if self.actor.act_dim != NUM_JOINTS:
            raise ValueError(f"Checkpoint outputs {self.actor.act_dim} actions, expected 16")
        if (self.estimator is None and self.actor.mean is not None
                and self.actor.mean.shape[0] != self.actor.obs_dim):
            raise ValueError("Observation normalizer size does not match network input")

    def describe(self) -> dict[str, Any]:
        s = self.spec
        return {
            "checkpoint": str(self.checkpoint_path),
            "iteration": self.iteration,
            "config_source": s.source,
            "network": f"{self.actor.obs_dim} -> {self.actor.hidden} -> {self.actor.act_dim} ({s.activation})",
            "obs_normalizer": self.actor.mean is not None,
            "obs_terms": s.obs_terms,
            "history_length": s.history_length,
            "action_mode": s.action_mode,
            "action_squash": s.action_squash,
            "cube_estimator": (
                f"{self.estimator.path.name} ({self.estimator.source}, sha256 "
                f"{self.estimator.source_sha256[:12]}..., self-check err {self.estimator.self_check_error:.1e})"
                if self.estimator is not None else None
            ),
            "delta_range_rad": [s.delta_min, s.delta_max] if s.action_mode == "delta" else None,
            "policy_hz": round(s.policy_hz, 3),
            "default_joint_pos": [round(float(v), 4) for v in s.default_joint_pos],
            "notes": s.notes,
        }

    # ------------------------------------------------------------------
    def _term_values(self, q: np.ndarray, qd: np.ndarray, cmd: np.ndarray) -> dict[str, np.ndarray]:
        sc = self.spec.obs_term_scales
        vals = {
            "joint_pos": q - self.default_joint_pos,
            "joint_vel": qd,
            "prev_commanded_joint_pos": cmd,
            "joint_pos_error": cmd - q,
            "last_action": self._last_action,
        }
        return {t: (vals[t] * sc.get(t, 1.0)).astype(np.float32) for t in self.spec.obs_terms}

    def reset(self, q_server: np.ndarray | None = None) -> np.ndarray:
        q = self.default_joint_pos.copy() if q_server is None else np.asarray(q_server, np.float32).copy()
        self._prev_command = np.clip(q, self.lower_limits, self.upper_limits)
        self._last_action[:] = 0.0
        vals = self._term_values(q, np.zeros(NUM_JOINTS, np.float32), self._prev_command)
        for t, buf in self._hist.items():
            buf[:] = vals[t]
        return self._prev_command.copy()

    def observation(self) -> np.ndarray:
        return np.concatenate([self._hist[t].reshape(-1) for t in self.spec.obs_terms])

    def step(
        self,
        q_server: np.ndarray,
        qd_server: np.ndarray | None = None,
        applied_command_server: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One control step. Returns (joint target, raw action), server order.

        ``applied_command_server`` is the command actually sent to the motors at
        the previous step (after any safety clipping by the bridge); when given,
        it replaces the policy's internal command so observations stay truthful.
        """
        q = np.asarray(q_server, np.float32)
        qd = np.zeros(NUM_JOINTS, np.float32) if qd_server is None else np.asarray(qd_server, np.float32)
        if applied_command_server is not None:
            self._prev_command = np.asarray(applied_command_server, np.float32).copy()
        vals = self._term_values(q, qd, self._prev_command)
        for t, buf in self._hist.items():
            buf[:-1] = buf[1:]
            buf[-1] = vals[t]
        obs = self.observation()
        feats = self.estimator.features(obs) if self.estimator is not None else None
        raw = self.actor(obs, feats).astype(np.float32)
        if self.spec.clip_actions:
            raw_used = np.clip(raw, -self.spec.clip_actions, self.spec.clip_actions)
        else:
            raw_used = raw
        a = np.tanh(raw_used) if self.spec.action_squash == "tanh" else np.clip(raw_used, -1.0, 1.0)
        if self.spec.action_mode == "delta":
            delta = self.spec.delta_min + 0.5 * (a + 1.0) * (self.spec.delta_max - self.spec.delta_min)
            target = self._prev_command + delta
        else:
            offset = self.default_joint_pos if self.spec.action_use_default_offset else 0.0
            target = offset + self.spec.action_scale * raw_used
        target = np.clip(target, self.lower_limits, self.upper_limits).astype(np.float32)
        self._prev_command = target.copy()
        self._last_action = raw.copy()
        return target, raw


def export_npz(policy: LeapPolicy, out: Path) -> None:
    a = policy.actor
    arrays = {f"w{i}": w for i, w in enumerate(a.weights)}
    arrays.update({f"b{i}": b for i, b in enumerate(a.biases)})
    if a.mean is not None:
        arrays["obs_mean"], arrays["obs_std"] = a.mean, a.std
    arrays["spec_json"] = np.array(json.dumps(policy.describe()))
    np.savez(out, **arrays)
