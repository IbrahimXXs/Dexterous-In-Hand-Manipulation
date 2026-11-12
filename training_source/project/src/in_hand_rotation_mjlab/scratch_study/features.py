"""The 19 cube-state features appended to the actor in Experiments C and D.

Identical definition, units, frame and normalisation for both experiments;
the only difference is the *source* of the physical state (C: frozen
estimator from proprioception; D: MuJoCo ground truth).

    pos_f     (3)  clamp((p - P_MEAN) / P_STD, +-5)   p: cube position in the palm frame [m]
    rot6d     (6)  first two columns of the canonical palm-frame rotation
    linvel_f  (3)  clamp(v / V_STD, +-5)              palm-frame linear velocity [m/s]
    angvel_f  (3)  clamp(w / W_STD, +-5)              palm-frame angular velocity [rad/s]
    contact   (4)  fingertip contact (C: probability, D: 0/1); index, middle, ring, thumb

Fixed scales (not re-fit per experiment, so C and D see exactly the same
mapping): taken from the palm-frame label statistics of the earlier
model_400 dataset (training split), rounded.

Cube symmetry: a cube's orientation is only defined up to its 24 proper
symmetries, and an estimator trained with a symmetry-aware loss may output
any equivalent rotation.  Both C and D therefore use the canonical
representative R* = R S*, S* = argmax_S trace(R S) (the equivalent rotation
closest to the palm frame).  It changes discontinuously only where two
representatives are equally close (e.g. every 45 deg of yaw), identically for
C and D.
"""

from __future__ import annotations

import torch

from in_hand_rotation_mjlab.estimation.rotations import cube_symmetry_matrices, quat_to_matrix

FEATURE_DIM = 19
CUBE_STATE_RAW_DIM = 17  # pos 3 | quat 4 | lin 3 | ang 3 | contact 4
P_MEAN = (0.0405, 0.0048, -0.0537)
P_STD = (0.0076, 0.0143, 0.0033)
V_STD = (0.0158, 0.0151, 0.0152)
W_STD = (0.480, 0.485, 0.435)
FEATURE_NAMES = (
  "pos_x", "pos_y", "pos_z",
  "rot_c1x", "rot_c1y", "rot_c1z", "rot_c2x", "rot_c2y", "rot_c2z",
  "linvel_x", "linvel_y", "linvel_z",
  "angvel_x", "angvel_y", "angvel_z",
  "contact_if", "contact_mf", "contact_rf", "contact_th",
)

_SYM_CACHE: dict = {}


def _sym(device, dtype) -> torch.Tensor:
  key = (str(device), dtype)
  if key not in _SYM_CACHE:
    _SYM_CACHE[key] = cube_symmetry_matrices(device=device, dtype=dtype)
  return _SYM_CACHE[key]


def canonical_rotation(R: torch.Tensor) -> torch.Tensor:
  """R S* with S* = argmax_S trace(R S) over the 24 cube symmetries."""
  S = _sym(R.device, R.dtype)  # [24,3,3]
  tr = torch.einsum("nij,kji->nk", R, S)
  best = tr.argmax(-1)
  return R @ S[best]


def cube_state_features(
  pos: torch.Tensor, R: torch.Tensor, lin_vel: torch.Tensor, ang_vel: torch.Tensor, contact: torch.Tensor
) -> torch.Tensor:
  dev, dt = pos.device, pos.dtype
  pm, ps = torch.tensor(P_MEAN, device=dev, dtype=dt), torch.tensor(P_STD, device=dev, dtype=dt)
  vs, ws = torch.tensor(V_STD, device=dev, dtype=dt), torch.tensor(W_STD, device=dev, dtype=dt)
  Rc = canonical_rotation(R)
  return torch.cat(
    [
      ((pos - pm) / ps).clamp(-5, 5),
      Rc[..., :, 0],
      Rc[..., :, 1],
      (lin_vel / vs).clamp(-5, 5),
      (ang_vel / ws).clamp(-5, 5),
      contact,
    ],
    dim=-1,
  )


def features_from_raw_state(raw: torch.Tensor) -> torch.Tensor:
  """``cube_state`` observation group [N,17] -> features [N,19]."""
  q = raw[:, 3:7]
  q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
  return cube_state_features(raw[:, 0:3], quat_to_matrix(q), raw[:, 7:10], raw[:, 10:13], raw[:, 13:17])


def features_from_estimator(estimator, obs: torch.Tensor) -> torch.Tensor:
  out = estimator(obs)
  return cube_state_features(out["pos"], out["rotmat"], out["lin_vel"], out["ang_vel"], out["contact_prob"])
