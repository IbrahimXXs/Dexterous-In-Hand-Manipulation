#!/usr/bin/env python3
"""C5-HW-EFFORT-V1: held-out comparison and the pre-registered decision (PROTOCOL_EFFORT_V1.md).

Reads results/items/<key>.json and data/<key>.npz written by eval_items.py.  Writes
results/effort_v1_comparison.{json,md} and figures/effort_v1_*.png.

Single-seed exploratory experiment: every CI here is a paired bootstrap over evaluation
environments (same seeds -> same reset draws for all policies) and describes evaluation variation
for this one trained model, NOT variation across independent training runs.

    python experiments/c5_hw_v1/scripts/compare_effort.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

E = Path("experiments/c5_hw_v1")
sys.path.insert(0, str(E / "scripts"))
A_POL, C5_POL, NEW = ("A1", "A2", "A3"), ("C5_1", "C5_2", "C5_3"), "EFFORT_1"
ALL = A_POL + ("R1",) + C5_POL + (NEW,)
SEEDS = (6201, 6202)
PRESETS = ("heavy_slippery", "high_delay", "obs_noise_x3")
NBOOT = 10000
KEYS30 = ("turns", "failure", "yaw_rate", "survival_s", "p_held", "turns_if_survived", "util_mean", "util_p95", "util_p99",
          "sat_frac_all", "sat_if_mcp", "sat_if_pip", "sat_th_axl", "index_blocked_frac", "jam_proxy", "tracking_err_mean",
          "joint_speed_mean", "cmd_limit_occupancy")


def item(setting, pol, act, preset, seed):
  p = E / "results" / "items" / f"{setting}_{pol}_{act}_{preset}_s{seed}.json"
  return json.loads(p.read_text()) if p.exists() else None


def pooled(setting, pol, act, preset="nominal", seeds=SEEDS):
  """Mean over the seeds' item means (equal environment counts = pooled mean)."""
  its = [item(setting, pol, act, preset, s) for s in seeds]
  if any(i is None for i in its):
    return None
  out = {}
  for tag in ("30s", "60s"):
    for k in KEYS30:
      v = [i["summary"][tag][k] for i in its]
      out[f"{k}_{tag}"] = float(np.mean(v))
  out["sat_per_joint_60s"] = {j: float(np.mean([i["summary"]["60s"]["sat_frac_per_joint"][j] for i in its]))
                              for j in its[0]["summary"]["60s"]["sat_frac_per_joint"]}
  out["util_mean_per_joint_60s"] = {j: float(np.mean([i["summary"]["60s"]["util_mean_per_joint"][j] for i in its]))
                                    for j in its[0]["summary"]["60s"]["util_mean_per_joint"]}
  out["util_p99_per_joint_60s"] = {j: float(np.mean([i["summary"]["60s"]["util_p99_per_joint"][j] for i in its]))
                                   for j in its[0]["summary"]["60s"]["util_p99_per_joint"]}
  out["tracking_per_joint_60s"] = {j: float(np.mean([i["summary"]["60s"]["tracking_err_per_joint"][j] for i in its]))
                                   for j in its[0]["summary"]["60s"]["tracking_err_per_joint"]}
  out["checkpoint"] = its[0]["checkpoint"]
  out["sha256"] = its[0]["sha256"]
  return out


def per_env(setting, pol, act, preset="nominal", seed=6201):
  """Per-environment turns30, fail30, jam (60 s), index blocked indicator, cache ids."""
  import eval_items as EI
  from in_hand_rotation_mjlab.grasp_rotation_v2.rollout import summary_per_env

  tr = dict(np.load(E / "data" / f"{setting}_{pol}_{act}_{preset}_s{seed}.npz"))
  s30 = summary_per_env(tr, 600)
  T = int(tr["T"])
  blocked = np.unpackbits(tr["blocked_idx_packed"], axis=0, count=T).astype(bool)
  jam = EI.jam_per_env(blocked, tr["end_step"], float(tr["dt"]))
  s60 = summary_per_env(tr, T)
  return {"turns30": s30["turns"], "fail30": s30["failure"], "fail60": s60["failure"], "jam60": jam["jam_60s"].astype(float),
          "jam30": jam["jam_30s"].astype(float), "onset": jam["first_jam_onset_s"], "longest": jam["longest_jam_s"],
          "end_step": tr["end_step"], "dt": float(tr["dt"]), "cache_id": tr["cache_id"]}


def jam_breakdown(pe: dict) -> dict:
  """Descriptive: where the >= 1 s index jams occur (both seeds pooled)."""
  f60 = np.concatenate([pe[s]["fail60"] for s in SEEDS]).astype(bool)
  j = np.concatenate([pe[s]["jam60"] for s in SEEDS]).astype(bool)
  on = np.concatenate([pe[s]["onset"] for s in SEEDS])
  end = np.concatenate([pe[s]["end_step"] for s in SEEDS]) * pe[SEEDS[0]]["dt"]
  jf = j & f60
  lag = (end - on)[jf]
  return {"jam_rate_held_60s": float(j[~f60].mean()) if (~f60).any() else math.nan,
          "jam_rate_failed_60s": float(j[f60].mean()) if f60.any() else math.nan,
          "failure_rate_given_jam": float(f60[j].mean()) if j.any() else math.nan,
          "failure_rate_given_no_jam": float(f60[~j].mean()) if (~j).any() else math.nan,
          "onset_lt_3s": float(np.mean(on[j] < 3)) if j.any() else math.nan,
          "onset_3_10s": float(np.mean((on[j] >= 3) & (on[j] < 10))) if j.any() else math.nan,
          "onset_ge_10s": float(np.mean(on[j] >= 10)) if j.any() else math.nan,
          "failed_within_2s_after_onset_of_jam_envs": float(np.mean(lag <= 2.0) * jf.sum() / max(j.sum(), 1)) if jf.any() else 0.0,
          "n_envs": int(len(j))}


def paired_bootstrap(new: dict, base: list[dict], key: str, rng) -> dict:
  """Difference new - mean(base) of the pooled mean over both seeds; environments resampled
  jointly (same index for every policy) within each seed."""
  diffs = []
  for s in SEEDS:
    d = new[s][key] - np.mean([b[s][key] for b in base], axis=0)
    diffs.append(d)
  point = float(np.mean(np.concatenate(diffs)))
  boots = np.zeros(NBOOT)
  for b in range(NBOOT):
    boots[b] = np.mean(np.concatenate([d[rng.integers(0, len(d), len(d))] for d in diffs]))
  lo, hi = np.percentile(boots, [2.5, 97.5])
  return {"diff": point, "ci95": [float(lo), float(hi)], "excludes_zero": bool(lo > 0 or hi < 0)}


def rel_bootstrap(new: dict, base: list[dict], key: str, rng) -> dict:
  """Relative reduction 1 - new/mean(base), same paired resampling."""
  n_all = [new[s][key] for s in SEEDS]
  b_all = [np.mean([b[s][key] for b in base], axis=0) for s in SEEDS]
  nb = float(np.mean(np.concatenate(b_all)))
  point = 1 - float(np.mean(np.concatenate(n_all))) / nb if nb > 0 else math.nan
  boots = np.zeros(NBOOT)
  for b in range(NBOOT):
    idx = [rng.integers(0, len(x), len(x)) for x in n_all]
    num = np.mean(np.concatenate([x[i] for x, i in zip(n_all, idx)]))
    den = np.mean(np.concatenate([x[i] for x, i in zip(b_all, idx)]))
    boots[b] = 1 - num / den if den > 0 else np.nan
  lo, hi = np.nanpercentile(boots, [2.5, 97.5])
  return {"reduction": point, "ci95": [float(lo), float(hi)]}


def main():
  rng = np.random.default_rng(20261124)
  res = {"label": "SINGLE-SEED EXPLORATORY EXPERIMENT (PPO seed 1 only). CIs are paired bootstraps over evaluation "
                  "environments: evaluation variation for this one model, not training-seed variation.",
         "decision_setting": "A (full 239-grasp distribution), nominal, seeds 6201+6202, effort-matched actuator (350 mA-equivalent; "
                             "response NOT fitted)", "tables": {}}
  # ---------- tables for every setting / actuator / preset
  for st in ("A", "B", "C"):
    for act in ("effort", "default"):
      presets = ("nominal",) + (PRESETS if st == "A" else ())
      for pr in presets:
        seeds = SEEDS if pr == "nominal" else (6201,)
        T = {p: pooled(st, p, act, pr, seeds) for p in ALL}
        T = {p: v for p, v in T.items() if v is not None}
        if not T:
          continue
        for grp, mem in (("A_mean", A_POL), ("C5_mean", C5_POL)):
          if all(m in T for m in mem):
            T[grp] = {k: float(np.mean([T[m][k] for m in mem])) for k in T[mem[0]] if isinstance(T[mem[0]][k], float)}
            T[grp]["sd_over_training_seeds"] = {k: float(np.std([T[m][k] for m in mem], ddof=1))
                                                for k in ("turns_30s", "failure_30s", "jam_proxy_60s", "sat_if_mcp_60s")}
        res["tables"][f"{st}/{act}/{pr}"] = T
  # ---------- decision
  key = "A/effort/nominal"
  T = res["tables"].get(key, {})
  dec = {"available": NEW in T and "C5_mean" in T and "A_mean" in T}
  if dec["available"]:
    n, c, a = T[NEW], T["C5_mean"], T["A_mean"]
    new_e = {s: per_env("A", NEW, "effort", "nominal", s) for s in SEEDS}
    c5_e = [{s: per_env("A", p, "effort", "nominal", s) for s in SEEDS} for p in C5_POL]
    for s in SEEDS:  # pairing check: same reset draws for every policy
      assert all((x[s]["cache_id"] == new_e[s]["cache_id"]).all() for x in c5_e)
    jam_d = paired_bootstrap(new_e, c5_e, "jam60", rng)
    jam_r = rel_bootstrap(new_e, c5_e, "jam60", rng)
    t_d = paired_bootstrap(new_e, c5_e, "turns30", rng)
    f_d = paired_bootstrap(new_e, c5_e, "fail30", rng)
    red = 1 - n["jam_proxy_60s"] / c["jam_proxy_60s"] if c["jam_proxy_60s"] > 0 else math.nan
    crit = {
      "1_jam_reduction_ge_50pct": bool(red >= 0.5), "1_ci_of_difference_excludes_0": jam_d["excludes_zero"] and jam_d["diff"] < 0,
      "2_turns30_ge_0.90_C5": bool(n["turns_30s"] >= 0.90 * c["turns_30s"]),
      "2_turns30_ge_1.20_A": bool(n["turns_30s"] >= 1.20 * a["turns_30s"]),
      "3_fail30_le_C5_plus_0.03": bool(n["failure_30s"] <= c["failure_30s"] + 0.03),
    }
    rej = {"turns30_lt_0.80_C5": bool(n["turns_30s"] < 0.80 * c["turns_30s"]), "jam_reduction_lt_25pct": bool(red < 0.25),
           "fail30_gt_C5_plus_0.08": bool(n["failure_30s"] > c["failure_30s"] + 0.08)}
    success = all(crit.values())
    rejected = any(rej.values())
    verdict = ("REJECTED" if rejected else "SINGLE-SEED PASS (exploratory; not confirmed)" if success else "INCONCLUSIVE")
    per_c5 = {}
    for p, ce in zip(C5_POL, c5_e):
      cp = T[p]
      per_c5[p] = {"jam_reduction": 1 - n["jam_proxy_60s"] / cp["jam_proxy_60s"] if cp["jam_proxy_60s"] > 0 else math.nan,
                   "jam_diff": paired_bootstrap(new_e, [ce], "jam60", rng),
                   "turns30_ratio": n["turns_30s"] / cp["turns_30s"], "fail30_diff": n["failure_30s"] - cp["failure_30s"]}
    dec.update({"values": {"new": {k: n[k] for k in ("turns_30s", "failure_30s", "jam_proxy_60s", "jam_proxy_30s")},
                           "C5_mean": {k: c[k] for k in ("turns_30s", "failure_30s", "jam_proxy_60s", "jam_proxy_30s")},
                           "A_mean": {k: a[k] for k in ("turns_30s", "failure_30s")}},
                "jam_reduction_point": red, "jam_reduction_bootstrap": jam_r, "jam_diff_bootstrap": jam_d,
                "turns30_diff_bootstrap": t_d, "fail30_diff_bootstrap": f_d,
                "turns30_ratio_C5": n["turns_30s"] / c["turns_30s"], "turns30_ratio_A": n["turns_30s"] / a["turns_30s"],
                "criteria": crit, "rejection": rej, "verdict": verdict, "per_C5_seed_descriptive": per_c5,
                "note": "Seed-level Welch tests (SPEC Section 7) are impossible with one training seed; criterion 1's CI is the "
                        "paired environment bootstrap (evaluation variation only)."})
    # default-actuator and secondary settings: descriptive
    for k2 in ("A/default/nominal", "B/effort/nominal", "C/effort/nominal", "B/default/nominal", "C/default/nominal"):
      t2 = res["tables"].get(k2, {})
      if NEW in t2 and "C5_mean" in t2:
        dec.setdefault("descriptive", {})[k2] = {
          "turns30_ratio_C5": t2[NEW]["turns_30s"] / t2["C5_mean"]["turns_30s"],
          "fail30_diff": t2[NEW]["failure_30s"] - t2["C5_mean"]["failure_30s"],
          "jam_new": t2[NEW]["jam_proxy_60s"], "jam_C5": t2["C5_mean"]["jam_proxy_60s"]}
    rob = {}
    for pr in PRESETS:
      t2 = res["tables"].get(f"A/effort/{pr}", {})
      if NEW in t2:
        rob[pr] = {"turns30_new": t2[NEW]["turns_30s"], "turns30_C5": t2["C5_mean"]["turns_30s"], "turns30_A": t2["A_mean"]["turns_30s"],
                   "ge_0.90_A": t2[NEW]["turns_30s"] >= 0.9 * t2["A_mean"]["turns_30s"],
                   "fail30_new": t2[NEW]["failure_30s"], "fail30_C5": t2["C5_mean"]["failure_30s"],
                   "jam_new": t2[NEW]["jam_proxy_60s"], "jam_C5": t2["C5_mean"]["jam_proxy_60s"]}
    dec["robustness_presets_effort"] = rob
    br = {NEW: jam_breakdown(new_e)}
    for p, ce in zip(C5_POL, c5_e):
      br[p] = jam_breakdown(ce)
    for p in A_POL + ("R1",):
      try:
        br[p] = jam_breakdown({s: per_env("A", p, "effort", "nominal", s) for s in SEEDS})
      except FileNotFoundError:
        pass
    dec["jam_breakdown_descriptive"] = br
  res["decision"] = dec
  (E / "results" / "effort_v1_comparison.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
  write_md(res)
  try:
    figures(res)
  except Exception as ex:  # noqa: BLE001
    print("figures skipped:", ex)
  print(json.dumps({k: v for k, v in dec.items() if k in ("verdict", "criteria", "rejection", "jam_reduction_point",
                                                          "jam_diff_bootstrap", "turns30_ratio_C5", "turns30_ratio_A")}, indent=1, default=float))


def f3(x):
  return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def f4(x):
  return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


ROWS = (("Turns / 30 s", "turns_30s", f3), ("Turns / 60 s", "turns_60s", f3), ("Yaw rate held (rad/s, 30 s)", "yaw_rate_30s", f3),
        ("Failure rate (30 s)", "failure_30s", f3), ("Failure rate (60 s)", "failure_60s", f3), ("P(held 60 s)", "p_held_60s", f3),
        ("Survival (s, 60-s rollout)", "survival_s_60s", f3), ("Turns / 60 s if held 60 s", "turns_if_survived_60s", f3),
        ("Index MCP saturation", "sat_if_mcp_60s", f4), ("Index PIP saturation", "sat_if_pip_60s", f4),
        ("Thumb abduction saturation", "sat_th_axl_60s", f4), ("All joints at ≥ 98% of limit", "sat_frac_all_60s", f4),
        ("Index blocked fraction", "index_blocked_frac_60s", f4), ("Index jam proxy (≥ 1 s, 60 s)", "jam_proxy_60s", f4),
        ("Index jam proxy (≥ 1 s, 30 s)", "jam_proxy_30s", f4),
        ("Effort utilisation mean", "util_mean_60s", f3), ("Utilisation p95", "util_p95_60s", f3), ("Utilisation p99", "util_p99_60s", f3),
        ("Tracking error mean (rad)", "tracking_err_mean_60s", f3), ("Joint speed mean (rad/s)", "joint_speed_mean_60s", f3),
        ("Command-limit occupancy", "cmd_limit_occupancy_60s", f3))


def table(T, cols):
  L = ["| Metric | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
  for name, k, f in ROWS:
    L.append(f"| {name} | " + " | ".join(f(T[c].get(k)) if c in T else "—" for c in cols) + " |")
  return L


def write_md(res):
  L = ["# C5-HW-EFFORT-V1: held-out comparison", "", f"**{res['label']}**", "",
       f"Decision setting: {res['decision_setting']}.", ""]
  d = res["decision"]
  if d.get("available"):
    L += [f"## Verdict: {d['verdict']}", ""]
    c = d["criteria"]
    L += ["| Pre-registered criterion (SPEC §7) | Value | Met |", "|---|---|---|",
          f"| 1a jam proxy reduced ≥ 50% vs C5 | {d['jam_reduction_point']:.1%} (CI {d['jam_reduction_bootstrap']['ci95'][0]:.1%} to {d['jam_reduction_bootstrap']['ci95'][1]:.1%}) | {'yes' if c['1_jam_reduction_ge_50pct'] else 'no'} |",
          f"| 1b 95% CI of the difference excludes 0 | {d['jam_diff_bootstrap']['diff']:+.4f} [{d['jam_diff_bootstrap']['ci95'][0]:+.4f}, {d['jam_diff_bootstrap']['ci95'][1]:+.4f}] | {'yes' if c['1_ci_of_difference_excludes_0'] else 'no'} |",
          f"| 2a turns30 ≥ 0.90 × C5 | ratio {d['turns30_ratio_C5']:.3f} | {'yes' if c['2_turns30_ge_0.90_C5'] else 'no'} |",
          f"| 2b turns30 ≥ 1.20 × A mean | ratio {d['turns30_ratio_A']:.3f} | {'yes' if c['2_turns30_ge_1.20_A'] else 'no'} |",
          f"| 3 failure30 ≤ C5 + 0.03 | {d['values']['new']['failure_30s']:.3f} vs {d['values']['C5_mean']['failure_30s']:.3f} | {'yes' if c['3_fail30_le_C5_plus_0.03'] else 'no'} |",
          "", "Rejection checks: " + ", ".join(f"{k} = {v}" for k, v in d["rejection"].items()), "",
          f"_{d['note']}_", "",
          f"Paired differences vs the C5 mean (environment bootstrap): turns30 {d['turns30_diff_bootstrap']['diff']:+.3f} "
          f"[{d['turns30_diff_bootstrap']['ci95'][0]:+.3f}, {d['turns30_diff_bootstrap']['ci95'][1]:+.3f}], failure30 "
          f"{d['fail30_diff_bootstrap']['diff']:+.3f} [{d['fail30_diff_bootstrap']['ci95'][0]:+.3f}, {d['fail30_diff_bootstrap']['ci95'][1]:+.3f}].", "",
          "Where the ≥ 1 s index jams occur (full distribution, effort-matched; descriptive):", "",
          "| policy | jam rate, held 60 s | jam rate, failed | P(fail \\| jam) | P(fail \\| no jam) | onset < 3 s | 3–10 s | ≥ 10 s |",
          "|---|---|---|---|---|---|---|---|"] + [
          f"| {p} | {b['jam_rate_held_60s']:.3f} | {b['jam_rate_failed_60s']:.3f} | {b['failure_rate_given_jam']:.3f} | "
          f"{b['failure_rate_given_no_jam']:.3f} | {b['onset_lt_3s']:.2f} | {b['onset_3_10s']:.2f} | {b['onset_ge_10s']:.2f} |"
          for p, b in d.get("jam_breakdown_descriptive", {}).items()] + ["",
          "Against each C5 seed (descriptive):", "", "| C5 seed | jam reduction | jam diff [CI] | turns30 ratio | failure30 diff |", "|---|---|---|---|---|"]
    for p, v in d["per_C5_seed_descriptive"].items():
      L.append(f"| {p} | {v['jam_reduction']:.1%} | {v['jam_diff']['diff']:+.4f} [{v['jam_diff']['ci95'][0]:+.4f}, {v['jam_diff']['ci95'][1]:+.4f}] | "
               f"{v['turns30_ratio']:.3f} | {v['fail30_diff']:+.3f} |")
    L.append("")
  for key, T in res["tables"].items():
    st, act, pr = key.split("/")
    name = {"A": "full 239-grasp distribution", "B": "grasp 49", "C": "robust 10-grasp subset"}[st]
    L += [f"## {name} · {act} actuator · {pr}" + (" · seeds 6201+6202" if pr == "nominal" else " · seed 6201"), ""]
    L += table(T, ["A_mean", "R1", "C5_mean", NEW])
    L += ["", "Per training seed:", ""]
    L += table(T, list(A_POL) + list(C5_POL) + [NEW])
    L.append("")
  (E / "results" / "effort_v1_comparison.md").write_text("\n".join(L) + "\n")


def figures(res):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  (E / "figures").mkdir(exist_ok=True)
  T = res["tables"].get("A/effort/nominal", {})
  if NEW not in T:
    return
  pols = list(A_POL) + ["R1"] + list(C5_POL) + [NEW]
  col = {p: ("#7f7f7f" if p in A_POL else "#9467bd" if p == "R1" else "#1f77b4" if p in C5_POL else "#d62728") for p in pols}
  fig, ax = plt.subplots(1, 4, figsize=(17, 4))
  for a, k, t in ((ax[0], "turns_30s", "turns / 30 s"), (ax[1], "failure_30s", "failure (30 s)"),
                  (ax[2], "jam_proxy_60s", "index jam proxy (envs, ≥1 s)"), (ax[3], "sat_if_mcp_60s", "index MCP saturation")):
    a.bar(range(len(pols)), [T[p][k] for p in pols], color=[col[p] for p in pols])
    a.set_xticks(range(len(pols)), pols, rotation=45)
    a.set_title(t)
  fig.suptitle("Full distribution, effort-matched actuator (350 mA-eq.), seeds 6201+6202 — EFFORT_1 is a single training seed")
  fig.tight_layout()
  fig.savefig(E / "figures" / "effort_v1_summary.png", dpi=120)
  plt.close(fig)
  jn = list(T[NEW]["sat_per_joint_60s"])
  fig, ax = plt.subplots(figsize=(12, 4))
  w = 0.27
  for i, (lab, vals) in enumerate((("A mean", [np.mean([T[p]["sat_per_joint_60s"][j] for p in A_POL]) for j in jn]),
                                   ("C5 mean", [np.mean([T[p]["sat_per_joint_60s"][j] for p in C5_POL]) for j in jn]),
                                   ("EFFORT_1", [T[NEW]["sat_per_joint_60s"][j] for j in jn]))):
    ax.bar(np.arange(16) + (i - 1) * w, vals, w, label=lab, color=("#7f7f7f", "#1f77b4", "#d62728")[i])
  ax.set_xticks(range(16), jn, rotation=45)
  ax.set_ylabel("share of samples at ≥ 98% of effort limit")
  ax.legend()
  ax.set_title("Per-joint saturation, full distribution, effort-matched actuator")
  fig.tight_layout()
  fig.savefig(E / "figures" / "effort_v1_saturation_per_joint.png", dpi=120)
  plt.close(fig)


if __name__ == "__main__":
  main()
