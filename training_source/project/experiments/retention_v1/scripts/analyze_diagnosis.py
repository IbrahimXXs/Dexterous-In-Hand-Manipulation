#!/usr/bin/env python3
"""Phase 1 analysis: why do the hardware-limited A policies lose the cube?

Reads data/diag_A{1,2,3}_nominal_seed9201.npz (record_diagnosis.py) and the 18 Nov
hardware logs, writes results/diagnosis.json and figures/diag_*.png.

Every per-episode quantity uses the first episode of each environment only; a failure is
characterised from its LAST LIVE step (on the termination step the environment has
already been reset).  Hypotheses: A gradual migration out of reach, B sudden slip after
loss of contact support, C joint limits without recovery, D poor initial grasp.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

E = Path("experiments/retention_v1")
RUNS = ("A1", "A2", "A3")
DT = 0.05
POS_THR, TILT_THR = 0.08, 0.8
HW_LOGS = Path("../../LEAP_Hand_RL_Deploy_Kit/logs")


def load(run):
  d = dict(np.load(E / "data" / f"diag_{run}_nominal_seed9201.npz"))
  return {k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in d.items()}


def episode_table(tr):
  T, n = tr["drift"].shape
  end = tr["end_step"]
  failed = end >= 0
  last = np.where(failed, np.maximum(end - 1, 0), T - 1)
  idx = np.arange(n)
  drift_last, tilt_last = tr["drift"][last, idx], tr["tilt"][last, idx]
  ftype = np.full(n, "survived", dtype=object)
  code = tr["end_code"]
  ftype[failed & (code == 4) & (drift_last / POS_THR >= tilt_last / TILT_THR)] = "pose:drift"
  ftype[failed & (code == 4) & (drift_last / POS_THR < tilt_last / TILT_THR)] = "pose:tilt"
  ftype[failed & (code == 2)] = "fell"
  ftype[failed & (code == 3)] = "too_fast"
  return {"end": end, "failed": failed, "last": last, "ftype": ftype, "drift_last": drift_last, "tilt_last": tilt_last}


def aligned(x, end, mask, window):
  """x [T,n,...] aligned to failure: rows k=0..window-1 = steps end-window..end-1."""
  out = []
  for i in np.flatnonzero(mask):
    e = end[i]
    if e >= window:
      out.append(x[e - window:e, i])
  return np.stack(out) if out else None


def main():
  res = {"runs": {}, "pooled": {}}
  tables, traces = {}, {}
  for run in RUNS:
    tr = load(run)
    traces[run] = tr
    et = episode_table(tr)
    tables[run] = et
    T, n = tr["drift"].shape
    turns = tr["yaw_inc"].sum(0) / (2 * math.pi)
    turns30 = tr["yaw_inc"][:600].sum(0) / (2 * math.pi)
    surv30 = ~et["failed"] | (et["end"] >= 600)
    types, counts = np.unique(et["ftype"], return_counts=True)
    res["runs"][run] = {
      "turns_30s": float(turns30.mean()), "turns_60s": float(turns.mean()),
      "survive_30s": float(surv30.mean()), "survive_60s": float((~et["failed"]).mean()),
      "mean_survival_60s_s": float(np.where(et["failed"], et["end"], T).mean() * DT),
      "end_types_60s": dict(zip(types.tolist(), (counts / n).round(4).tolist())),
      "yaw_rate_while_alive_rad_s": float(tr["yaw_inc"].sum() / (tr["alive"].sum() * DT)),
    }
  # ---------------------------------------------------------------- pooled per-episode arrays
  P = {}
  for k in ("end", "failed", "ftype", "drift_last", "tilt_last"):
    P[k] = np.concatenate([tables[r][k] for r in RUNS])
  P["run"] = np.concatenate([[r] * len(tables[r]["end"]) for r in RUNS])
  cat = lambda key: np.concatenate([traces[r][key] for r in RUNS], axis=1)  # noqa: E731
  pos = cat("cube_pos_palm")
  drift, tilt, alive = cat("drift"), cat("tilt"), cat("alive").astype(bool)
  dforce, oforce = cat("distal_force"), cat("other_force")
  atlim, head, act = cat("at_limit").astype(bool), cat("headroom"), cat("action")
  linv = np.linalg.norm(cat("lin_vel_palm"), axis=-1)
  yawi = cat("yaw_inc")
  T, N = drift.shape
  fail = P["failed"]
  pose_fail = np.isin(P["ftype"], ["pose:drift", "pose:tilt"])
  types, counts = np.unique(P["ftype"][fail], return_counts=True)
  res["pooled"]["failure_type_share"] = dict(zip(types.tolist(), (counts / fail.sum()).round(3).tolist()))
  res["pooled"]["n_episodes"] = int(N)
  res["pooled"]["failure_rate_60s"] = float(fail.mean())
  end_s = P["end"][fail] * DT
  res["pooled"]["failure_time_quantiles_s"] = np.quantile(end_s, [0.1, 0.25, 0.5, 0.75, 0.9]).round(1).tolist()
  res["pooled"]["failures_in_first_5s_share"] = float((end_s < 5).mean())
  # hazard per 10-s block (failures per alive episode)
  haz = []
  for b in range(0, T, 200):
    at_risk = (P["end"] < 0) | (P["end"] >= b)
    ev = fail & (P["end"] >= b) & (P["end"] < b + 200)
    haz.append(round(float(ev.sum() / max(1, at_risk.sum())), 4))
  res["pooled"]["failure_hazard_per_10s_block"] = haz

  # ---------------------------------------------------------------- A: gradual vs abrupt
  W = 100  # 5 s before failure
  mask = pose_fail & (P["end"] >= W)
  dA, tA = aligned(drift, P["end"], mask, W), aligned(tilt, P["end"], mask, W)
  ft = P["ftype"][mask]
  dd = mask.copy()
  gr = {}
  for name, X, thr, sel in (("drift", dA, POS_THR, ft == "pose:drift"), ("tilt", tA, TILT_THR, ft == "pose:tilt")):
    if X is None or sel.sum() == 0:
      continue
    Xs = X[sel] / thr  # fraction of the termination threshold
    gr[name] = {
      "n": int(sel.sum()),
      "fraction_of_threshold_at_t_minus": {f"{s}s": float(np.median(Xs[:, W - 1 - int(s / DT)])) for s in (0.25, 0.5, 1, 2, 3, 5)
                                          if W - 1 - int(s / DT) >= 0},
      "share_below_half_threshold_1s_before": float((Xs[:, W - 1 - 20] < 0.5).mean()),
      "share_below_half_threshold_0.5s_before": float((Xs[:, W - 1 - 10] < 0.5).mean()),
    }
  res["pooled"]["gradual_vs_abrupt"] = gr
  # baseline for comparison: survivors' drift / tilt distribution
  surv = ~fail
  res["pooled"]["survivor_drift_quantiles_m"] = np.quantile(drift[:, surv][alive[:, surv]], [0.5, 0.9, 0.99]).round(4).tolist()
  res["pooled"]["survivor_tilt_quantiles_rad"] = np.quantile(tilt[:, surv][alive[:, surv]], [0.5, 0.9, 0.99]).round(3).tolist()
  # survivors: is there a slow trend? drift at 10/30/60 s
  res["pooled"]["survivor_median_drift_m_at_10_30_60s"] = [float(np.median(drift[t, surv])) for t in (199, 599, T - 1)]
  res["pooled"]["survivor_median_tilt_rad_at_10_30_60s"] = [float(np.median(tilt[t, surv])) for t in (199, 599, T - 1)]

  # ---------------------------------------------------------------- palm-frame workspace
  sp = pos[:, surv][alive[:, surv]]
  mu, cov = sp.mean(0), np.cov(sp.T)
  ci = np.linalg.inv(cov)
  md = np.sqrt(np.einsum("tni,ij,tnj->tn", pos - mu, ci, pos - mu))
  res["pooled"]["palm_frame_survivor_mean_m"] = mu.round(4).tolist()
  res["pooled"]["palm_frame_survivor_sd_m"] = np.sqrt(np.diag(cov)).round(4).tolist()
  res["pooled"]["palm_frame_survivor_mean_full_m"] = mu.tolist()
  res["pooled"]["palm_frame_survivor_cov_m2"] = cov.tolist()
  # hazard of failing within the next 2 s as a function of Mahalanobis distance (alive steps)
  horizon = 40
  t_idx = np.arange(T)[:, None]
  fail_soon = fail[None, :] & (P["end"][None, :] > t_idx) & (P["end"][None, :] <= t_idx + horizon)
  valid = alive & (t_idx < T - horizon)
  bins = np.array([0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 100])
  h = []
  for lo_, hi_ in zip(bins[:-1], bins[1:]):
    m = valid & (md >= lo_) & (md < hi_)
    h.append({"md": [float(lo_), float(hi_)], "share_of_steps": float(m.sum() / valid.sum()),
              "p_fail_within_2s": float(fail_soon[m].mean()) if m.sum() > 50 else None})
  res["pooled"]["hazard_vs_palm_mahalanobis"] = h
  base_haz = float(fail_soon[valid].mean())
  res["pooled"]["base_p_fail_within_2s"] = base_haz
  # "practical workspace": smallest distance d* beyond which the 2-s failure probability >= 3x base
  dstar = next((b["md"][0] for b in h if b["p_fail_within_2s"] is not None and b["p_fail_within_2s"] >= 3 * base_haz), None)
  res["pooled"]["workspace_exit_mahalanobis"] = dstar
  if dstar is not None:
    out_ws = alive & (md >= dstar)
    first_exit = np.where(out_ws.any(0), out_ws.argmax(0), -1)
    pf = pose_fail
    lead = (P["end"] - first_exit)[pf & (first_exit >= 0)] * DT
    res["pooled"]["workspace_exit"] = {
      "share_of_pose_failures_preceded_by_exit": float((first_exit[pf] >= 0).mean()),
      "lead_time_exit_to_failure_s_quantiles": np.quantile(lead, [0.1, 0.25, 0.5, 0.75, 0.9]).round(2).tolist() if len(lead) else None,
      "share_of_survivors_ever_exiting": float((first_exit[surv] >= 0).mean()),
      "palm_position_at_exit_mean_m": pos[first_exit[pf & (first_exit >= 0)], np.flatnonzero(pf & (first_exit >= 0))].mean(0).round(4).tolist()
      if (pf & (first_exit >= 0)).any() else None,
    }
  # palm-frame displacement direction at failure (last live step - initial)
  idxs = np.flatnonzero(pose_fail)
  last = np.maximum(P["end"][idxs] - 1, 0)
  disp = pos[last, idxs] - pos[0, idxs]
  res["pooled"]["palm_displacement_at_pose_failure_mean_m"] = disp.mean(0).round(4).tolist()
  res["pooled"]["palm_displacement_at_pose_failure_sd_m"] = disp.std(0).round(4).tolist()
  sidx = np.flatnonzero(surv)
  sd = pos[T - 1, sidx] - pos[0, sidx]
  res["pooled"]["palm_displacement_survivors_60s_mean_m"] = sd.mean(0).round(4).tolist()

  # ---------------------------------------------------------------- B: contact support
  ncont = (dforce > 0.05).sum(-1)
  cA = aligned(ncont, P["end"], mask, W)
  fA = aligned(dforce.sum(-1) + oforce, P["end"], mask, W)
  cS = ncont[:, surv][alive[:, surv]]
  res["pooled"]["contacts"] = {
    "survivors_mean_distal_contacts": float(cS.mean()),
    "survivors_share_steps_le1_contact": float((cS <= 1).mean()),
    "pose_fail_mean_contacts_at_t_minus": {f"{s}s": float(cA[:, W - 1 - int(s / DT)].mean()) for s in (0.25, 0.5, 1, 2, 5) if W - 1 - int(s / DT) >= 0},
    "pose_fail_share_le1_contact_in_last_1s": float((cA[:, -20:] <= 1).mean()),
    "pose_fail_total_normal_force_N_at_t_minus": {f"{s}s": float(fA[:, W - 1 - int(s / DT)].mean()) for s in (0.25, 1, 5)},
    "survivors_mean_total_normal_force_N": float((dforce.sum(-1) + oforce)[:, surv][alive[:, surv]].mean()),
  }
  # ---------------------------------------------------------------- C: joint limits
  aL = aligned(atlim.mean(-1), P["end"], mask, W)
  hL = aligned(head.min(-1), P["end"], mask, W)
  res["pooled"]["joint_limits"] = {
    "survivors_cmd_at_limit_frac": float(atlim.mean(-1)[:, surv][alive[:, surv]].mean()),
    "pose_fail_cmd_at_limit_frac_at_t_minus": {f"{s}s": float(aL[:, W - 1 - int(s / DT)].mean()) for s in (0.25, 1, 5)},
    "per_joint_at_limit_survivors": atlim[:, surv][alive[:, surv]].mean(0).round(3).tolist(),
    "per_joint_at_limit_last1s_pose_fail": aligned(atlim, P["end"], mask, 20).mean((0, 1)).round(3).tolist(),
  }
  sat = (np.abs(act) > 0.99).mean(-1)
  sA = aligned(sat, P["end"], mask, W)
  res["pooled"]["action_saturation"] = {"survivors": float(sat[:, surv][alive[:, surv]].mean()),
                                        "pose_fail_last_1s": float(sA[:, -20:].mean()), "pose_fail_5_to_4s_before": float(sA[:, :20].mean())}
  # ---------------------------------------------------------------- D: initial grasp
  cid = np.concatenate([traces[r]["cache_id"] for r in RUNS])
  s30 = (~fail) | (P["end"] >= 600)
  ug = np.unique(cid)
  per = {int(g): (int((cid == g).sum()), float(s30[cid == g].mean())) for g in ug}
  k = np.array([per[g][0] for g in per]); p_ = np.array([per[g][1] for g in per])
  pbar = s30.mean()
  chi2 = float((k * (p_ - pbar) ** 2 / (pbar * (1 - pbar))).sum())
  dof = len(k) - 1
  # between-policy consistency: per-grasp 30-s survival in A1 vs mean of A2/A3
  cons = {}
  rr = []
  for r in RUNS:
    m = P["run"] == r
    rr.append(np.array([s30[m & (cid == g)].mean() if (m & (cid == g)).sum() >= 2 else np.nan for g in ug]))
  for i, j in ((0, 1), (0, 2), (1, 2)):
    ok = ~np.isnan(rr[i]) & ~np.isnan(rr[j])
    cons[f"{RUNS[i]}~{RUNS[j]}"] = float(np.corrcoef(rr[i][ok], rr[j][ok])[0, 1])
  from scipy.stats import chi2 as chi2d

  res["pooled"]["initial_grasp"] = {
    "grasps_seen": int(len(ug)), "mean_trials_per_grasp": float(k.mean()),
    "survive30_rate": float(pbar), "chi2_homogeneity": chi2, "dof": dof, "p_value": float(chi2d.sf(chi2, dof)),
    "between_policy_correlation_of_per_grasp_survival": cons,
    "share_grasps_survival_lt_0.5": float((p_[k >= 8] < 0.5).mean()), "share_grasps_survival_ge_0.9": float((p_[k >= 8] >= 0.9).mean()),
  }
  # initial-state predictors of 30-s failure
  p0 = pos[0]
  c0 = ncont[0]
  res["pooled"]["initial_state_vs_failure30"] = {
    "initial_palm_pos_survivors_mean": p0[s30].mean(0).round(4).tolist(), "initial_palm_pos_failures_mean": p0[~s30].mean(0).round(4).tolist(),
    "initial_mahalanobis_survivors_mean": float(md[0, s30].mean()), "initial_mahalanobis_failures_mean": float(md[0, ~s30].mean()),
    "initial_contacts_survivors_mean": float(c0[s30].mean()), "initial_contacts_failures_mean": float(c0[~s30].mean()),
  }
  # rotation while alive: do failing episodes rotate as fast before failing?
  ya = aligned(yawi, P["end"], mask, W)
  res["pooled"]["yaw_rate_rad_s"] = {"survivors": float(yawi[:, surv][alive[:, surv]].mean() / DT),
                                     "pose_fail_5_to_2s_before": float(ya[:, :60].mean() / DT), "pose_fail_last_1s": float(ya[:, -20:].mean() / DT)}
  res["pooled"]["linear_speed_m_s"] = {"survivors_median": float(np.median(linv[:, surv][alive[:, surv]])),
                                       "pose_fail_last_0.5s_median": float(np.median(aligned(linv, P["end"], mask, 10)))}
  # ---------------------------------------------------------------- sim vs hardware through estimator B
  res["hardware_comparison"] = hw_compare(traces)
  (E / "results" / "diagnosis.json").write_text(json.dumps(res, indent=1))
  figures(traces, tables, P, pos, md, drift, tilt, alive, ncont, atlim, sat, yawi, mask, res, cid, s30)
  print(json.dumps(res["pooled"], indent=1)[:6000])
  print(json.dumps(res["runs"], indent=1))
  print(json.dumps(res["hardware_comparison"], indent=1)[:3000])


def hw_compare(traces):
  """Compare hwlim A/C seed-1 hardware runs (18 Nov) with sim A1 through estimator B and joint data."""
  out = {}
  logs = sorted(HW_LOGS.glob("physical_20261118_*.npz"))
  sim = traces["A1"]
  al = sim["alive"].astype(bool)
  est = sim.get("est_pos_palm")
  for f in logs:
    d = np.load(f, allow_pickle=True)
    if "mon_cube_pos_palm" not in d.files or len(d["q_hw"]) < 300:
      continue
    meta = json.loads(str(d["run_meta_json"]))
    pol = meta.get("policy", meta.get("policy_name", "?"))
    p = d["mon_cube_pos_palm"]
    cp = d["mon_contact_prob_b"]
    out[f.stem] = {"policy": str(pol), "steps": int(len(p)),
                   "est_pos_palm_first2s_mean": p[:40].mean(0).round(4).tolist(), "est_pos_palm_last5s_mean": p[-100:].mean(0).round(4).tolist(),
                   "est_pos_change_m": float(np.linalg.norm(p[-100:].mean(0) - p[:40].mean(0))),
                   "est_contact_prob_mean": cp.mean(0).round(3).tolist(), "est_yaw_turns": float(d["mon_est_yaw_rad"][-1] / (2 * math.pi))}
  if est is not None:
    T = min(600, est.shape[0])
    surv = sim["end_step"] < 0
    e0, e1 = est[:40, surv].mean((0)), est[T - 100:T, surv].mean(0)
    out["sim_A1_survivors_30s"] = {"est_pos_palm_first2s_mean": e0.mean(0).round(4).tolist(), "est_pos_palm_last5s_mean": e1.mean(0).round(4).tolist(),
                                   "est_pos_change_m_median": float(np.median(np.linalg.norm(e1 - e0, axis=-1))),
                                   "true_pos_change_m_median": float(np.median(np.linalg.norm(sim["cube_pos_palm"][T - 100:T, surv].mean(0) - sim["cube_pos_palm"][:40, surv].mean(0), axis=-1))),
                                   "est_contact_prob_mean": sim["est_contact_prob"][:T][al[:T]].mean(0).round(3).tolist()}
  return out


def figures(traces, tables, P, pos, md, drift, tilt, alive, ncont, atlim, sat, yawi, mask, res, cid, s30):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                       "font.size": 8.5, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
  BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"
  T = drift.shape[0]
  t = np.arange(T) * DT
  # 1. example trajectories: 3 pose failures (late), 3 survivors, from A1
  tr, et = traces["A1"], tables["A1"]
  fails = [i for i in np.flatnonzero(np.isin(et["ftype"], ["pose:drift", "pose:tilt"])) if et["end"][i] > 300][:3]
  survs = list(np.flatnonzero(~et["failed"])[:3])
  fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
  for i, c in [(i, ORANGE) for i in fails] + [(i, BLUE) for i in survs]:
    a = tr["alive"][:, i].astype(bool)
    L = a.sum() if et["failed"][i] else T
    ax[0].plot(t[:L], tr["drift"][:L, i] * 1000, color=c, lw=1.2)
    ax[1].plot(t[:L], np.linalg.norm(tr["cube_pos_palm"][:L, i] - tr["cube_pos_palm"][0, i], axis=-1) * 1000, color=c, lw=1.2)
    ax[2].plot(t[:L], np.cumsum(tr["yaw_inc"][:L, i]) / (2 * math.pi), color=c, lw=1.2)
    ax[3].plot(t[:L], (tr["distal_force"][:L, i] > 0.05).sum(-1) + 0.05 * (c == ORANGE), color=c, lw=0.8)
    ax[4].plot(t[:L], tr["at_limit"][:L, i].sum(-1), color=c, lw=0.8)
  ax[0].axhline(80, color=GRAY, ls="--", lw=1); ax[0].set_ylabel("drift from start\n(world, mm)")
  ax[1].set_ylabel("palm-frame displacement\n(mm)"); ax[2].set_ylabel("cumulative turns")
  ax[3].set_ylabel("distal links\nin contact"); ax[4].set_ylabel("joints commanded\nat a limit"); ax[4].set_xlabel("time (s)")
  ax[0].set_title("A seed 1, 60 s: 3 pose failures (orange) vs 3 survivors (blue); dashed = termination threshold")
  fig.tight_layout(); fig.savefig(E / "figures" / "diag_example_trajectories.png", dpi=140); plt.close(fig)
  # 2. failure-aligned averages vs survivor baseline
  W = 100
  ta = (np.arange(W) - W) * DT
  al = aligned
  surv = ~P["failed"]
  fig, ax = plt.subplots(2, 3, figsize=(12, 6))
  items = [("drift (mm)", drift * 1000), ("tilt (deg)", np.degrees(tilt)), ("palm-frame Mahalanobis distance", md),
           ("distal links in contact", ncont.astype(float)), ("share of joints at a limit", atlim.mean(-1)), ("action saturation (|tanh u|>0.99)", sat)]
  for a_, (name, X) in zip(ax.ravel(), items):
    A = al(X, P["end"], mask, W)
    m, lo_, hi_ = A.mean(0), np.quantile(A, 0.25, 0), np.quantile(A, 0.75, 0)
    a_.fill_between(ta, lo_, hi_, color=ORANGE, alpha=0.2, lw=0)
    a_.plot(ta, m, color=ORANGE, lw=2, label="pose failures (mean, IQR)")
    base = X[:, surv][alive[:, surv]].mean()
    a_.axhline(base, color=BLUE, lw=2, ls="--", label="survivors (mean)")
    a_.set_title(name); a_.set_xlabel("time before failure (s)")
  ax[0, 0].legend(frameon=False)
  fig.suptitle(f"Pose failures aligned to the failure time (n = {mask.sum()}, pooled A seeds 1-3, 60 s)")
  fig.tight_layout(); fig.savefig(E / "figures" / "diag_failure_aligned.png", dpi=140); plt.close(fig)
  # 3. hazard vs palm-frame distance + palm-frame scatter
  fig, ax = plt.subplots(1, 2, figsize=(11, 4))
  h = [b for b in res["pooled"]["hazard_vs_palm_mahalanobis"] if b["p_fail_within_2s"] is not None]
  xs = [0.5 * (b["md"][0] + min(b["md"][1], 10)) for b in h]
  ax[0].plot(xs, [b["p_fail_within_2s"] for b in h], "o-", color=ORANGE, lw=2, ms=6)
  ax[0].axhline(res["pooled"]["base_p_fail_within_2s"], color=GRAY, ls="--", lw=1)
  ax[0].set_xlabel("Mahalanobis distance of cube centre from the survivors' palm-frame distribution")
  ax[0].set_ylabel("P(pose failure within 2 s)"); ax[0].set_title("Failure hazard vs cube position in the palm frame")
  rng = np.random.default_rng(0)
  si = rng.choice(np.flatnonzero(surv), 200, replace=False)
  fi = np.flatnonzero(mask)
  ax[1].scatter(pos[::20, si, 0][alive[::20, si]] * 1000, pos[::20, si, 1][alive[::20, si]] * 1000, s=2, color=BLUE, alpha=0.3, label="survivors (every 1 s)")
  lastp = pos[P["end"][fi] - 1, fi]
  ax[1].scatter(lastp[:, 0] * 1000, lastp[:, 1] * 1000, s=6, color=ORANGE, alpha=0.6, label="pose failures (last live step)")
  ax[1].set_xlabel("palm x (mm)"); ax[1].set_ylabel("palm y (mm)"); ax[1].legend(frameon=False, markerscale=3)
  ax[1].set_title("Cube centre in the palm frame")
  fig.tight_layout(); fig.savefig(E / "figures" / "diag_workspace.png", dpi=140); plt.close(fig)
  # 4. per-grasp 30-s survival
  ug = np.unique(cid)
  k = np.array([(cid == g).sum() for g in ug]); p_ = np.array([s30[cid == g].mean() for g in ug])
  order = np.argsort(p_)
  fig, ax = plt.subplots(figsize=(10, 3.2))
  ax.bar(np.arange(len(ug)), p_[order], color=BLUE, width=0.8)
  ax.axhline(s30.mean(), color=GRAY, ls="--", lw=1)
  ax.set_xlabel(f"reset-cache grasp (sorted; {k.mean():.0f} trials each on average)"); ax.set_ylabel("30-s survival")
  ax.set_title("30-s survival per initial grasp (pooled A seeds 1-3); dashed = overall")
  fig.tight_layout(); fig.savefig(E / "figures" / "diag_initial_grasps.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
  import sys

  sys.path.insert(0, str(Path(__file__).parent))
  main()
