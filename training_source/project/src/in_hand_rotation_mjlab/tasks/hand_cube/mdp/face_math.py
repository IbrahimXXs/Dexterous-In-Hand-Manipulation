"""Geometry helpers for the numbered-cube face-up task.

The face convention is intentionally shared with the earlier face-goal work:

  1: +Z, 2: +X, 3: +Y, 4: -Y, 5: -X, 6: -Z

Public tensor APIs use zero-based face IDs (0..5); user-facing tools may add one.
"""

from __future__ import annotations

import itertools

import torch

from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix


_FACE_NORMAL_VALUES = (
  (0.0, 0.0, 1.0),
  (1.0, 0.0, 0.0),
  (0.0, 1.0, 0.0),
  (0.0, -1.0, 0.0),
  (-1.0, 0.0, 0.0),
  (0.0, 0.0, -1.0),
)


def cube_face_normals(
  *, device: torch.device | str, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
  """Return the six labeled face normals in the cube body frame."""
  return torch.tensor(_FACE_NORMAL_VALUES, device=device, dtype=dtype)


def cube_face_normals_w(cube_quat_w: torch.Tensor) -> torch.Tensor:
  """Return all labeled face normals in world coordinates, shape (..., 6, 3)."""
  rotm_wb = matrix_from_quat(cube_quat_w)
  normals_b = cube_face_normals(device=cube_quat_w.device, dtype=cube_quat_w.dtype)
  return torch.einsum("...ij,fj->...fi", rotm_wb, normals_b)


def target_face_alignment(
  cube_quat_w: torch.Tensor, target_face_ids: torch.Tensor
) -> torch.Tensor:
  """Cosine alignment of the requested face normal with world up (+Z)."""
  batch_shape = torch.broadcast_shapes(
    cube_quat_w.shape[:-1], target_face_ids.shape
  )
  cube_quat_w = cube_quat_w.expand(batch_shape + (4,))
  target_face_ids = target_face_ids.expand(batch_shape)
  normals_w = cube_face_normals_w(cube_quat_w)
  target_face_ids = target_face_ids.to(device=cube_quat_w.device, dtype=torch.long)
  gather_shape = target_face_ids.shape + (1, 3)
  target_normal_w = torch.gather(
    normals_w,
    dim=-2,
    index=target_face_ids[..., None, None].expand(gather_shape),
  ).squeeze(-2)
  return torch.clamp(target_normal_w[..., 2], min=-1.0, max=1.0)


def target_face_angle(
  cube_quat_w: torch.Tensor, target_face_ids: torch.Tensor
) -> torch.Tensor:
  """Unsigned target-face error angle in radians, in [0, pi]."""
  return torch.acos(target_face_alignment(cube_quat_w, target_face_ids))


def top_face_ids(cube_quat_w: torch.Tensor) -> torch.Tensor:
  """Return the zero-based ID of the face most aligned with world up."""
  return cube_face_normals_w(cube_quat_w)[..., 2].argmax(dim=-1)


def rotation_6d_from_quat(quat: torch.Tensor) -> torch.Tensor:
  """Continuous, quaternion-sign-invariant 6D orientation representation.

  The output concatenates the first and second columns of the rotation matrix:
  ``[r00, r10, r20, r01, r11, r21]``.
  """
  rotm = matrix_from_quat(quat)
  return torch.cat([rotm[..., :, 0], rotm[..., :, 1]], dim=-1)


def cube_symmetry_quats(
  *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
  """Return the 24 proper axis-aligned rotations of a cube as WXYZ quaternions."""
  rotations: list[torch.Tensor] = []
  eye = torch.eye(3, dtype=torch.float64)
  for permutation in itertools.permutations(range(3)):
    permuted = eye[:, permutation]
    for signs in itertools.product((-1.0, 1.0), repeat=3):
      candidate = permuted * torch.tensor(signs, dtype=torch.float64)
      if torch.linalg.det(candidate) > 0.5:
        rotations.append(candidate)
  matrices = torch.stack(rotations, dim=0).to(device=device, dtype=dtype)
  return quat_from_matrix(matrices)
