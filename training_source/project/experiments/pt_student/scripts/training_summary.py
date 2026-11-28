#!/usr/bin/env python3
"""PT-student training summary (CPU; reads logs only): teacher vs C5 training curves, final distillation losses,
GPU hours.

* Teacher vs C5: mean of "Mean reward" and "Episode_Metrics/genuine_yaw_rate" over updates 2800-3000 of each run's
  training log (the same environment and reward; the teacher's actor also sees the privileged critic group).
* Students: mean of each loss term and of the per-step episode-end rate over the last 100 DAgger updates.
* GPU hours: Slurm jobs from their start/exit lines (an externally killed job: its log's last "Time elapsed");
  evaluation from the per-item "seconds" in results/items; short smoke runs are not counted.

    python experiments/pt_student/scripts/training_summary.py
Writes results/training.json and results/training.md.
"""

from __future__ import annotations

import glob
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

E = Path(__file__).resolve().parents[1]
PROJECT = E.parents[1]
GRV2 = PROJECT / "experiments" / "grasp_rotation_v2" / "logs"
WINDOW = (2800, 3000)
STUDENTS = ("S-TF", "S-TF-noaux", "S-MLP", "S-TF-AR")
# Slurm jobs of this study: (label, job id, kind); a job whose slurm log is missing is skipped.
JOBS = (("teacher (killed externally at ~990 updates; not used)", 338614, "train"), ("teacher", 339167, "train"),
        ("S-TF", 339499, "distill"), ("S-TF-noaux", 339962, "distill"), ("S-MLP", 339963, "distill"),
        ("teacher + action-rate (variant)", 339968, "train"))


def curve(log: Path, lo: int, hi: int) -> dict:
  it, rew, yaw = None, [], []
  for line in log.read_text(errors="replace").splitlines():
    m = re.search(r"Learning iteration (\d+)/", line)
    if m:
      it = int(m.group(1))
    elif it is not None and lo <= it <= hi:
      if "Mean reward:" in line:
        rew.append(float(line.split()[-1]))
      elif "genuine_yaw_rate" in line:
        yaw.append(float(line.split()[-1]))
  return {"updates": [lo, hi], "n": len(rew), "mean_reward": float(np.mean(rew)) if rew else None,
          "genuine_yaw_rate": float(np.mean(yaw)) if yaw else None}


def stamp(line: str) -> datetime:
  """`date` output, e.g. 'Fri Nov 27 19:48:36 +04 2026' (the zone has no minutes, which %z needs)."""
  p = line.split()[:6]
  if re.fullmatch(r"[+-]\d\d", p[4]):
    p[4] += "00"
  return datetime.strptime(" ".join(p), "%a %b %d %H:%M:%S %z %Y")


def job_hours(job: int) -> dict | None:
  f = E / "logs" / f"slurm_{job}.out"
  if not f.exists():
    return None
  lines = f.read_text(errors="replace").splitlines()
  start = next((stamp(x) for x in lines if " start " in x), None)
  end = next((stamp(x) for x in reversed(lines) if re.search(r" exit \d+", x)), None)
  if start is not None and end is not None:
    return {"hours": (end - start).total_seconds() / 3600, "source": "start/exit lines", "finished": True}
  # no exit line: killed externally (or still running) -> the training log's last "Time elapsed"
  logs = sorted((E / "logs").glob(f"*job{job}.log"))
  if logs:
    el = re.findall(r"Time elapsed: (\d+):(\d+):(\d+)", logs[0].read_text(errors="replace"))
    if el:
      h, m, s = map(int, el[-1])
      return {"hours": h + m / 60 + s / 3600, "source": f"last 'Time elapsed' in {logs[0].name}", "finished": False}
  return {"hours": None, "source": "running", "finished": False}


def main() -> None:
  out = {"label": "training logs (descriptive); every teacher and student is ONE training seed", "teacher_vs_c5": {}, "students": {}, "gpu_hours": {}}
  out["teacher_vs_c5"]["T"] = curve(E / "logs" / "train_teacher_seed1.log", *WINDOW)
  if (E / "logs" / "train_teacher_ar_seed1.log").exists():
    c = curve(E / "logs" / "train_teacher_ar_seed1.log", *WINDOW)
    if c["n"]:
      out["teacher_vs_c5"]["T-AR"] = c
  for s in (1, 2, 3):
    out["teacher_vs_c5"][f"C5s{s}"] = curve(GRV2 / f"train_C5_seed{s}.log", *WINDOW)
  for s in STUDENTS:
    f = E / "logs" / f"distill_{s}.jsonl"
    if not f.exists():
      continue
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    meta, recs = rows[0]["meta"], rows[1:]
    if len(recs) < meta["updates"]:
      continue  # still training
    last = recs[-100:]
    out["students"][s] = {"updates": len(recs), "n_params": meta["n_params"], "teacher_sha256": meta["teacher_sha256"][:16],
                          **{k: float(np.mean([r[k] for r in last])) for k in ("loss_action", "loss_pos", "loss_ori", "loss_joint", "done_rate_per_step")},
                          "seconds_per_update": float(np.mean([r["seconds"] for r in recs]))}
  for label, job, kind in JOBS:
    h = job_hours(job)
    if h is not None:
      out["gpu_hours"][f"{job} {label}"] = h
  ev = sum(json.loads(Path(f).read_text())["seconds"] for f in glob.glob(str(E / "results" / "items" / "*.json")))
  out["gpu_hours"]["evaluation (sum of item seconds)"] = {"hours": ev / 3600, "source": "results/items/*.json", "finished": True}
  out["gpu_hours_total"] = float(sum(v["hours"] for v in out["gpu_hours"].values() if v["hours"] is not None))
  (E / "results" / "training.json").write_text(json.dumps(out, indent=1) + "\n")
  f = lambda x, d=3: "–" if x is None else f"{x:.{d}f}"  # noqa: E731
  L = ["# PT-student: training summary", "", f"Generated by `scripts/training_summary.py`. **{out['label']}.**", "",
       f"## Teacher vs C5 in training (same environment and reward), mean over updates {WINDOW[0]}–{WINDOW[1]}", "",
       "| Run | Mean reward | Genuine yaw rate |", "|---|---|---|"]
  for k, v in out["teacher_vs_c5"].items():
    L.append(f"| {k} | {f(v['mean_reward'], 2)} | {f(v['genuine_yaw_rate'])} |")
  L += ["", "## Students: mean over the last 100 DAgger updates (β = 0)", "",
        "| Student | Parameters | Action loss | Position loss | Orientation loss | Clean-joint loss | Episode ends per step | s / update |",
        "|---|---|---|---|---|---|---|---|"]
  for k, v in out["students"].items():
    L.append(f"| {k} | {v['n_params']:,} | {f(v['loss_action'], 4)} | {f(v['loss_pos'])} | {f(v['loss_ori'])} | {f(v['loss_joint'], 4)} | "
             f"{f(v['done_rate_per_step'], 4)} | {f(v['seconds_per_update'], 1)} |")
  L += ["", "S-TF-noaux trains its auxiliary heads with weight 0, so its auxiliary losses are untrained-head values.", "",
        "## GPU hours (one RTX 5000 Ada per job)", "", "| Job | Hours | Source |", "|---|---|---|"]
  for k, v in out["gpu_hours"].items():
    L.append(f"| {k} | {f(v['hours'], 2)} | {v['source']} |")
  L += [f"| **Total** | **{out['gpu_hours_total']:.1f}** | |"]
  (E / "results" / "training.md").write_text("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  main()
