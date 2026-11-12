"""Singularity-free cube-orientation geometry used by the scratch study.

Conventions: quaternions are (w, x, y, z); ``R(q)`` maps cube-body vectors to
the world frame; world +Z is the desired rotation axis (gravity is -Z).

Physical tilt
-------------
Let ``u(q) = R(q)^T e_z`` be the world vertical expressed in the cube body
frame.  The tilt of the cube relative to a reference orientation ``q0`` is

    tilt(q, q0) = arccos( clamp(u(q) . u(q0), -1, 1) )            (rad)

* Rotating the cube about the world vertical (the task) leaves it unchanged:
  ``(Rz R)^T e_z = R^T Rz^T e_z = R^T e_z``.
* It equals the angle through which the cube has been tipped away from its
  reference attitude, independent of yaw and of any Euler convention, so it
  has no singular orientations (unlike the original XYZ-Euler roll/pitch
  difference, which amplifies small tilts near |pitch| = 90 deg).

Genuine yaw increment
---------------------
Between consecutive control steps the world-frame incremental rotation is
``dq = q_t (x) q_{t-1}^{-1}``.  With ``dq`` taken in the w >= 0 hemisphere, its
rotation vector is ``theta * n``; the yaw increment is its world-z component

    dpsi = theta * n_z                                          (rad)

which is the angle turned about +Z (exact for rotations about Z, first-order
correct otherwise), continuous through all orientations, and never affected
by quaternion sign flips.  Increments are only accumulated for transitions
inside one episode (never across a reset).
"""

from __future__ import annotations

import torch


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    dim=-1,
  )


def world_up_in_body(q: torch.Tensor) -> torch.Tensor:
  """u = R(q)^T e_z, i.e. the third row of R(q)."""
  q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
  w, x, y, z = q.unbind(-1)
  return torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1)


def physical_tilt(q: torch.Tensor, q_ref: torch.Tensor | None = None, up_ref: torch.Tensor | None = None) -> torch.Tensor:
  """Tilt angle (rad) of ``q`` relative to ``q_ref`` (or a stored ``up_ref``)."""
  u = world_up_in_body(q)
  if up_ref is None:
    up_ref = world_up_in_body(q_ref)
  return torch.acos((u * up_ref).sum(-1).clamp(-1.0, 1.0))


def rotation_vector(q: torch.Tensor) -> torch.Tensor:
  """Rotation vector of a unit quaternion (shortest rotation, w >= 0)."""
  q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
  q = torch.where(q[..., :1] < 0, -q, q)
  v = q[..., 1:]
  s = v.norm(dim=-1, keepdim=True)
  angle = 2.0 * torch.atan2(s, q[..., :1])
  # angle / s -> 2 as s -> 0
  scale = torch.where(s > 1e-9, angle / s.clamp_min(1e-12), torch.full_like(s, 2.0))
  return v * scale


def yaw_increment(q_now: torch.Tensor, q_prev: torch.Tensor) -> torch.Tensor:
  """World-z component of the rotation vector of q_now * q_prev^-1 (rad)."""
  return rotation_vector(quat_mul(q_now, quat_conj(q_prev)))[..., 2]
