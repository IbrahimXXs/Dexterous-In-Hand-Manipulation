"""Combined supervised loss and metrics for the cube-state estimator.

    L = l_pos * L_pos + l_rot * L_rot + l_vel * L_vel + l_contact * L_contact

- ``L_pos``      Huber (smooth-L1) on standardised position.
- ``L_rot``      symmetry-aware chordal distance  min_S ||R_hat - R S||_F^2
                 (S over the cube's 24 rotations; see ``rotations.py``).
- ``L_vel``      Huber on standardised linear + angular velocity.
- ``L_contact``  binary cross-entropy on fingertip contact logits.

Samples flagged ``contact_valid == False`` (the first step after a reset,
whose contact sensor value predates the reset) are excluded from L_contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .layout import FINGERTIP_NAMES
from .rotations import (
  cube_symmetry_matrices,
  geodesic_angle,
  quat_to_matrix,
  symmetric_chordal_loss,
  symmetric_geodesic_angle,
)


@dataclass
class LossWeights:
  pos: float = 1.0
  rot: float = 1.0
  vel: float = 0.5
  contact: float = 0.5
  huber_delta: float = 1.0
  # Optional positive-class weight for contact BCE (None = unweighted).
  contact_pos_weight: float | None = None


class EstimatorLoss:
  def __init__(self, weights: LossWeights, device: torch.device | str):
    self.w = weights
    self.sym = cube_symmetry_matrices(device=device)
    self.pos_weight = (
      None
      if weights.contact_pos_weight is None
      else torch.tensor(weights.contact_pos_weight, device=device)
    )

  def __call__(self, model, pred: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    pos_n = (batch["pos"] - model.pos_mean) / model.pos_std
    lin_n = batch["lin_vel"] / model.lin_vel_std
    ang_n = batch["ang_vel"] / model.ang_vel_std
    d = self.w.huber_delta
    l_pos = F.huber_loss(pred["pos_n"], pos_n, delta=d)
    l_vel = 0.5 * (
      F.huber_loss(pred["lin_vel_n"], lin_n, delta=d)
      + F.huber_loss(pred["ang_vel_n"], ang_n, delta=d)
    )
    r_gt = quat_to_matrix(batch["quat"])
    l_rot = symmetric_chordal_loss(pred["rotmat"], r_gt, self.sym)[0].mean()

    valid = batch["contact_valid"].float().unsqueeze(-1).expand_as(batch["contact"])
    bce = F.binary_cross_entropy_with_logits(
      pred["contact_logits"],
      batch["contact"].float(),
      pos_weight=self.pos_weight,
      reduction="none",
    )
    l_con = (bce * valid).sum() / valid.sum().clamp_min(1.0)

    total = (
      self.w.pos * l_pos
      + self.w.rot * l_rot
      + self.w.vel * l_vel
      + self.w.contact * l_con
    )
    return total, {
      "loss/total": total.detach(),
      "loss/pos": l_pos.detach(),
      "loss/rot": l_rot.detach(),
      "loss/vel": l_vel.detach(),
      "loss/contact": l_con.detach(),
    }


class MetricAccumulator:
  """Streaming metrics over many batches (sums kept on device)."""

  def __init__(self, device: torch.device | str, threshold: float = 0.5):
    self.device = device
    self.thr = threshold
    self.sym = cube_symmetry_matrices(device=device)
    self.reset()

  def reset(self) -> None:
    z = lambda *s: torch.zeros(*s, device=self.device, dtype=torch.float64)  # noqa: E731
    self.n = z(())
    self.pos_se = z(3)
    self.lin_se = z(3)
    self.ang_se = z(3)
    self.rot_sym_deg = z(())
    self.rot_raw_deg = z(())
    self.rot_sym_sq = z(())
    self.rot_sym_all: list[torch.Tensor] = []
    self.tp = z(4)
    self.fp = z(4)
    self.fn = z(4)
    self.tn = z(4)

  @torch.no_grad()
  def update(self, pred: dict, batch: dict, keep_rot_errors: bool = False) -> None:
    n = batch["pos"].shape[0]
    self.n += n
    self.pos_se += ((pred["pos"] - batch["pos"]) ** 2).sum(0).double()
    self.lin_se += ((pred["lin_vel"] - batch["lin_vel"]) ** 2).sum(0).double()
    self.ang_se += ((pred["ang_vel"] - batch["ang_vel"]) ** 2).sum(0).double()
    r_gt = quat_to_matrix(batch["quat"])
    sym_deg = torch.rad2deg(symmetric_geodesic_angle(pred["rotmat"], r_gt, self.sym))
    raw_deg = torch.rad2deg(geodesic_angle(pred["rotmat"], r_gt))
    self.rot_sym_deg += sym_deg.sum().double()
    self.rot_sym_sq += (sym_deg**2).sum().double()
    self.rot_raw_deg += raw_deg.sum().double()
    if keep_rot_errors:
      self.rot_sym_all.append(sym_deg.float().cpu())
    valid = batch["contact_valid"].bool().unsqueeze(-1)
    p = (pred["contact_prob"] >= self.thr) & valid
    t = batch["contact"].bool() & valid
    nv = (~p) & (~t) & valid
    self.tp += (p & t).sum(0).double()
    self.fp += (p & ~t).sum(0).double()
    self.fn += (~p & t).sum(0).double()
    self.tn += nv.sum(0).double()

  def compute(self) -> dict[str, float]:
    n = self.n.clamp_min(1)
    out: dict[str, float] = {}
    pos_rmse_axis = torch.sqrt(self.pos_se / n)
    out["pos_rmse_mm"] = float(torch.sqrt(self.pos_se.sum() / n) * 1000)
    for i, ax in enumerate("xyz"):
      out[f"pos_rmse_{ax}_mm"] = float(pos_rmse_axis[i] * 1000)
    out["lin_vel_rmse_mps"] = float(torch.sqrt(self.lin_se.sum() / n))
    out["ang_vel_rmse_radps"] = float(torch.sqrt(self.ang_se.sum() / n))
    out["yaw_rate_rmse_radps"] = float(torch.sqrt(self.ang_se[2] / n))
    out["rot_err_sym_mean_deg"] = float(self.rot_sym_deg / n)
    out["rot_err_sym_rms_deg"] = float(torch.sqrt(self.rot_sym_sq / n))
    out["rot_err_raw_mean_deg"] = float(self.rot_raw_deg / n)
    if self.rot_sym_all:
      allerr = torch.cat(self.rot_sym_all)
      out["rot_err_sym_median_deg"] = float(allerr.median())
      out["rot_err_sym_p90_deg"] = float(torch.quantile(allerr[:1_000_000], 0.9))
    prec = self.tp / (self.tp + self.fp).clamp_min(1)
    rec = self.tp / (self.tp + self.fn).clamp_min(1)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-12)
    acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn).clamp_min(1)
    for i, name in enumerate(FINGERTIP_NAMES):
      out[f"contact_{name}_precision"] = float(prec[i])
      out[f"contact_{name}_recall"] = float(rec[i])
      out[f"contact_{name}_f1"] = float(f1[i])
      out[f"contact_{name}_accuracy"] = float(acc[i])
      out[f"contact_{name}_positive_rate"] = float(
        (self.tp[i] + self.fn[i])
        / (self.tp[i] + self.tn[i] + self.fp[i] + self.fn[i]).clamp_min(1)
      )
    tp, fp, fn = self.tp.sum(), self.fp.sum(), self.fn.sum()
    mp = tp / (tp + fp).clamp_min(1)
    mr = tp / (tp + fn).clamp_min(1)
    out["contact_micro_precision"] = float(mp)
    out["contact_micro_recall"] = float(mr)
    out["contact_micro_f1"] = float(2 * mp * mr / (mp + mr).clamp_min(1e-12))
    out["contact_macro_f1"] = float(f1.mean())
    out["num_samples"] = int(self.n)
    return out
