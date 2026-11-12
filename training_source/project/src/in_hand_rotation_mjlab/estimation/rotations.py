"""Rotation utilities (pure PyTorch, batched, no MJLab dependency).

Conventions: quaternions are ``wxyz`` (MuJoCo / MJLab convention); rotation
matrices act on column vectors; the 6D representation (Zhou et al., CVPR 2019)
is the first two *columns* of the rotation matrix, concatenated.

Cube symmetry
-------------
A uniform cube is invariant under the 24 proper rotations of the octahedral
group.  With only proprioception the estimator cannot tell those orientations
apart (the 2 mm COM randomisation is far below what joint angles reveal), so
the primary orientation loss and metric are symmetry-aware:
``min_S d(R_pred, R_gt S)``.  The raw (non-symmetric) error is reported too.
"""

from __future__ import annotations

import itertools
import math

import torch


def quat_normalize(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
  return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
  """wxyz quaternion [..., 4] -> rotation matrix [..., 3, 3]."""
  q = quat_normalize(q)
  w, x, y, z = q.unbind(-1)
  two = 2.0
  r00 = 1 - two * (y * y + z * z)
  r01 = two * (x * y - w * z)
  r02 = two * (x * z + w * y)
  r10 = two * (x * y + w * z)
  r11 = 1 - two * (x * x + z * z)
  r12 = two * (y * z - w * x)
  r20 = two * (x * z - w * y)
  r21 = two * (y * z + w * x)
  r22 = 1 - two * (x * x + y * y)
  return torch.stack(
    [
      torch.stack([r00, r01, r02], -1),
      torch.stack([r10, r11, r12], -1),
      torch.stack([r20, r21, r22], -1),
    ],
    -2,
  )


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
  """Rotation matrix [..., 3, 3] -> wxyz quaternion with w >= 0."""
  m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
  qw = 0.5 * torch.sqrt((1 + m00 + m11 + m22).clamp_min(0))
  qx = 0.5 * torch.sqrt((1 + m00 - m11 - m22).clamp_min(0))
  qy = 0.5 * torch.sqrt((1 - m00 + m11 - m22).clamp_min(0))
  qz = 0.5 * torch.sqrt((1 - m00 - m11 + m22).clamp_min(0))
  qx = torch.copysign(qx, m[..., 2, 1] - m[..., 1, 2])
  qy = torch.copysign(qy, m[..., 0, 2] - m[..., 2, 0])
  qz = torch.copysign(qz, m[..., 1, 0] - m[..., 0, 1])
  return quat_normalize(torch.stack([qw, qx, qy, qz], -1))


def matrix_to_rot6d(m: torch.Tensor) -> torch.Tensor:
  """[..., 3, 3] -> [..., 6] (first column, second column)."""
  return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def rot6d_to_matrix(d6: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  """Gram-Schmidt: [..., 6] -> proper rotation matrix [..., 3, 3]."""
  a1, a2 = d6[..., :3], d6[..., 3:6]
  b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(eps)
  a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
  b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(eps)
  b3 = torch.cross(b1, b2, dim=-1)
  return torch.stack([b1, b2, b3], dim=-1)


def geodesic_angle(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
  """Angle (rad) of r1^T r2, numerically safe. [..., 3, 3] -> [...]."""
  rel = r1.transpose(-1, -2) @ r2
  trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
  cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
  return torch.acos(cos)


def cube_symmetry_matrices(
  device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
  """The 24 proper rotations mapping a cube onto itself: [24, 3, 3]."""
  mats = []
  for perm in itertools.permutations(range(3)):
    for signs in itertools.product((-1.0, 1.0), repeat=3):
      m = torch.zeros(3, 3, dtype=torch.float64)
      for row, (col, s) in enumerate(zip(perm, signs)):
        m[row, col] = s
      if torch.linalg.det(m) > 0:
        mats.append(m)
  out = torch.stack(mats).to(dtype=dtype)
  assert out.shape[0] == 24
  return out.to(device) if device is not None else out


def symmetric_chordal_loss(
  r_pred: torch.Tensor, r_gt: torch.Tensor, sym: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """min_S ||R_pred - R_gt S||_F^2 per sample, plus the argmin index.

  ``||A - B||_F^2 = 6 - 2 tr(A^T B)`` for rotations, so the minimum is the
  maximum of ``tr(R_pred^T R_gt S)`` over the symmetry group.
  """
  # rel[..., i, j] = (R_pred^T R_gt)[i, j]; tr(rel S) = sum_ij rel_ij S_ji.
  rel = r_pred.transpose(-1, -2) @ r_gt
  traces = torch.einsum("...ij,kji->...k", rel, sym)
  best_trace, idx = traces.max(dim=-1)
  return 6.0 - 2.0 * best_trace, idx


def symmetric_geodesic_angle(
  r_pred: torch.Tensor, r_gt: torch.Tensor, sym: torch.Tensor
) -> torch.Tensor:
  rel = r_pred.transpose(-1, -2) @ r_gt
  traces = torch.einsum("...ij,kji->...k", rel, sym)
  best = traces.max(dim=-1).values
  cos = ((best - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
  return torch.acos(cos)


def tilt_angle(r: torch.Tensor) -> torch.Tensor:
  """Symmetry-invariant tilt: angle between the palm z-axis and the cube
  axis most aligned with it.  0 when a cube face lies flat on the palm."""
  # Column j of r is cube axis j expressed in the palm frame; its z-component
  # is the cosine to the palm normal.
  cos = r[..., 2, :].abs().max(dim=-1).values.clamp(max=1.0)
  return torch.acos(cos)


def yaw_mod_quarter_turn(r: torch.Tensor) -> torch.Tensor:
  """Yaw of the cube about the palm z-axis, reduced modulo pi/2.

  Uses the cube axis most aligned with the palm plane's x direction; for a
  near-upright cube this is the only yaw quantity that is observable.
  """
  # Project each cube axis onto the palm xy-plane and take the one with the
  # largest in-plane norm; its heading modulo 90 degrees is the yaw.
  xy = r[..., :2, :]  # [..., 2, 3]
  norms = xy.norm(dim=-2)
  idx = norms.argmax(dim=-1, keepdim=True)
  axis = torch.gather(xy, -1, idx.unsqueeze(-2).expand(*xy.shape[:-1], 1)).squeeze(-1)
  heading = torch.atan2(axis[..., 1], axis[..., 0])
  return torch.remainder(heading, math.pi / 2)
