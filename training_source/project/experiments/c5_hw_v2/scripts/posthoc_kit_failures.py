#!/usr/bin/env python3
"""C5-HW-V2 POST-HOC, DESCRIPTIVE analysis (added after the pre-registered decision; not part of it, changes nothing).

Question: are the ~4-6% 30-s failures from simulated kit starts (setting K) a property of the starts, shared by
every policy, or policy-specific?  Uses the held-out K rollouts already on disk (effort actuator, nominal, seeds
6201 + 6202; the same per-environment start and randomisation draws for every policy, checked here via cache_id).
For each environment it counts how many of the eight baseline policies (A1-A3, R1, C5_1-C5_3, EFFORT_1) fail
from it, and reports how C5HW2_1's failures overlap with the C5 seeds'.  Writes results/posthoc_kit_failures.json.

    python experiments/c5_hw_v2/scripts/posthoc_kit_failures.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compare_v2", E / "scripts" / "compare_v2.py")
CV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CV)

BASE = ("A1", "A2", "A3", "R1", "C5_1", "C5_2", "C5_3", "EFFORT_1")
NEW = "C5HW2_1"


def main() -> None:
  fails, ids = {}, None
  for p in BASE + (NEW,):
    arrs = [CV.per_env(CV.key("K", p, "effort", "nominal", s)) for s in (6201, 6202)]
    cid = np.concatenate([a["cache_id"] for a in arrs])
    if ids is None:
      ids = cid
    assert np.array_equal(ids, cid), f"start draws differ for {p}"
    fails[p] = 1.0 - np.concatenate([a["survived30"] for a in arrs])
  n = len(ids)
  nbase = sum(fails[p] for p in BASE)  # 0..8 baseline policies failing from each environment's start
  new_f = fails[NEW].astype(bool)
  c5_all = np.all([fails[p].astype(bool) for p in ("C5_1", "C5_2", "C5_3")], axis=0)
  c5_any = np.any([fails[p].astype(bool) for p in ("C5_1", "C5_2", "C5_3")], axis=0)
  hist = {str(k): int((nbase == k).sum()) for k in range(len(BASE) + 1)}
  res = {
    "label": "POST-HOC, DESCRIPTIVE (not pre-registered; single-seed exploratory; does not change the verdict)",
    "setting": "K (simulated kit starts), effort actuator, nominal, held-out seeds 6201+6202 pooled",
    "n_environments": int(n),
    "paired_start_draws_checked": True,
    "failure30": {p: float(fails[p].mean()) for p in BASE + (NEW,)},
    "environments_by_number_of_baseline_policies_failing": hist,
    "share_failing_for_all_8_baselines": float((nbase == len(BASE)).mean()),
    "share_failing_for_at_least_6_of_8_baselines": float((nbase >= 6).mean()),
    "C5HW2_1_failures": int(new_f.sum()),
    "C5HW2_1_failures_also_failed_by_all_3_C5_seeds": float(c5_all[new_f].mean()),
    "C5HW2_1_failures_also_failed_by_any_C5_seed": float(c5_any[new_f].mean()),
    "C5HW2_1_failures_with_at_least_6_of_8_baselines_failing": float((nbase[new_f] >= 6).mean()),
    "C5HW2_1_failure_rate_on_starts_no_baseline_fails": float(new_f[nbase == 0].mean()),
    "C5HW2_1_failure_rate_on_starts_all_8_baselines_fail": float(new_f[nbase == len(BASE)].mean()) if (nbase == len(BASE)).any() else None,
  }
  c1 = json.loads((E / "results" / "v2_comparison.json").read_text())["criteria"]["1_kit_start_survival30"]
  res["criterion1_failure_threshold"] = 0.5 * min(c1["failure_C5_mean"], c1["failure_EFFORT_1"])
  res["criterion1_threshold_below_all_8_fail_share"] = bool(res["criterion1_failure_threshold"] < res["share_failing_for_all_8_baselines"])
  (E / "results" / "posthoc_kit_failures.json").write_text(json.dumps(res, indent=1) + "\n")
  print(json.dumps(res, indent=1))


if __name__ == "__main__":
  main()
