#!/usr/bin/env python3
"""PT-student distillation by DAgger (PLAN.md section 2).  Training seed 1, environment = the teacher's training
environment (= C5's: Mjlab-Leap-GRV2-C5 env, 4096 envs, domain randomisation, resets, noise, curriculum;
for the optional S-TF-AR, the action-rate teacher's environment, which differs only by that reward term).

Each update: 32-step rollout; per environment and step the executed action is the teacher's with probability beta,
else the student's; the teacher labels every visited state; the rollout joins a FIFO buffer of the last 4 rollouts;
one pass over the buffer in shuffled minibatches of 16384.  beta = 1 -> 0 linearly over the first 20% of updates.
AdamW lr 3e-4, wd 0.01, 50-update linear warm-up, grad-norm clip 1.0.  Final checkpoint = the one used.

    python experiments/pt_student/scripts/distill.py --student S-TF --teacher <model_3000.pt> [--updates 1500]
Writes experiments/pt_student/runs/<student>/ckpt_*.pt (git-ignored) and logs/distill_<student>.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
E = PROJECT / "experiments" / "pt_student"
TEACHER_TASK = "Mjlab-Leap-PT-Teacher"
TEACHER_TASK_AR = "Mjlab-Leap-PT-Teacher-AR"  # optional action-rate variant (PLAN 4.3)
STUDENTS = {"S-TF": ("transformer", True), "S-TF-noaux": ("transformer", False), "S-MLP": ("mlp", True), "S-TF-AR": ("transformer", True)}
TEACHER_TASK_OF = {s: (TEACHER_TASK_AR if s == "S-TF-AR" else TEACHER_TASK) for s in STUDENTS}
N_ENVS, STEPS, BUF, MB = 4096, 32, 4, 16384
LR, WD, WARMUP, CLIP = 3e-4, 0.01, 50, 1.0


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


M = _load("pt_models", E / "scripts" / "models.py")


def make_train_env(task: str, n: int, seed: int, device: str):
  from in_hand_rotation_mjlab.estimation.sim import _import_tasks
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  _import_tasks()
  cfg = load_env_cfg(task)
  cfg.scene.num_envs, cfg.seed = int(n), int(seed)
  agent_cfg = load_rl_cfg(task)
  raw = ManagerBasedRlEnv(cfg=cfg, device=device)
  return RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions), raw, agent_cfg


def load_teacher(env, ck: str, device: str, task: str = TEACHER_TASK):
  from in_hand_rotation_mjlab.scratch_study.evaluation import load_study_policy
  from mjlab.tasks.registry import load_rl_cfg

  return load_study_policy(env, load_rl_cfg(task), ck, device)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--student", required=True, choices=list(STUDENTS))
  ap.add_argument("--teacher", required=True)
  ap.add_argument("--updates", type=int, default=1500)
  ap.add_argument("--num-envs", type=int, default=N_ENVS)
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--minibatch", type=int, default=MB, help="smoke tests only; the plan fixes 16384")
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", default=None, help="run directory (default runs/<student>)")
  a = ap.parse_args()
  torch.manual_seed(a.seed)
  np.random.seed(a.seed)
  dev = a.device
  kind, aux = STUDENTS[a.student]
  out = Path(a.out) if a.out else E / "runs" / a.student
  out.mkdir(parents=True, exist_ok=True)
  logf = (E / "logs" / f"distill_{a.student}.jsonl") if a.out is None else out / "distill.jsonl"
  w_aux = M.AUX_WEIGHTS if aux else {k: 0.0 for k in M.AUX_WEIGHTS}

  task = TEACHER_TASK_OF[a.student]  # the teacher's own training environment
  env, raw, _ = make_train_env(task, a.num_envs, a.seed, dev)
  teacher = load_teacher(env, a.teacher, dev, task)
  student = (M.TransformerStudent() if kind == "transformer" else M.MLPStudent()).to(dev)
  opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WD)
  sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / WARMUP))
  meta = {"student": a.student, "teacher_task": task, "kind": kind, "aux": aux, "aux_weights": w_aux, "teacher": a.teacher,
          "teacher_sha256": hashlib.sha256(Path(a.teacher).read_bytes()).hexdigest(), "updates": a.updates, "num_envs": a.num_envs,
          "seed": a.seed, "n_params": M.n_params(student), "buffer_rollouts": BUF, "minibatch": a.minibatch, "lr": LR}
  print(json.dumps(meta), flush=True)
  buf: list[dict] = []
  obs = env.get_observations()
  n = a.num_envs
  beta_end = 0.2 * a.updates
  t_start = time.time()
  with open(logf, "w") as lf:
    lf.write(json.dumps({"meta": meta}) + "\n")
    for it in range(a.updates):
      t0 = time.time()
      beta = max(0.0, 1.0 - it / beta_end)
      student.eval()
      R = {"obs": [], "ta": [], "pos": [], "ori": [], "joint": []}
      ep_done, rew = 0.0, 0.0
      log_acc: dict[str, list] = {}
      with torch.no_grad():
        for _ in range(STEPS):
          a_t = teacher(obs)
          o = obs["actor"]
          tg = M.aux_targets(obs["cube_state"], obs["critic"])
          if bool(student.norm.fitted):
            a_s = student(o)["action"]
            mix = torch.rand(n, device=dev) < beta
            act = torch.where(mix[:, None], a_t, a_s)
          else:
            act = a_t
          R["obs"].append(o.clone())
          R["ta"].append(torch.tanh(a_t))
          for k in ("pos", "ori", "joint"):
            R[k].append(tg[k])
          obs, r, dones, extras = env.step(act)
          ep_done += float(dones.float().mean())
          rew += float(r.mean())
          for k, v in (extras.get("log") or {}).items():
            if "yaw" in k.lower() or "Termination" in k or "episode_length" in k.lower():
              log_acc.setdefault(k, []).append(float(v) if not torch.is_tensor(v) else float(v.float().mean()))
      roll = {k: torch.cat(v) for k, v in R.items()}
      if not bool(student.norm.fitted):
        student.norm.fit(roll["obs"])
      buf.append(roll)
      if len(buf) > BUF:
        buf.pop(0)
      data = {k: torch.cat([b[k] for b in buf]) for k in roll}
      N = data["obs"].shape[0]
      perm = torch.randperm(N, device=dev)
      student.train()
      L = {k: 0.0 for k in ("action", "pos", "ori", "joint", "total")}
      nb = 0
      for s in range(0, N - a.minibatch + 1, a.minibatch):
        idx = perm[s:s + a.minibatch]
        o = student(data["obs"][idx])
        lt = M.loss_terms(o, data["ta"][idx], {k: data[k][idx] for k in ("pos", "ori", "joint")})
        loss = lt["action"] + sum(w_aux[k] * lt[k] for k in w_aux)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), CLIP)
        opt.step()
        for k in lt:
          L[k] += float(lt[k])
        L["total"] += float(loss)
        nb += 1
      sched.step()
      rec = {"update": it, "beta": beta, "lr": opt.param_groups[0]["lr"], "n_batches": nb, **{f"loss_{k}": v / max(nb, 1) for k, v in L.items()},
             "done_rate_per_step": ep_done / STEPS, "reward_mean": rew / STEPS,
             **{k: float(np.mean(v)) for k, v in log_acc.items()}, "seconds": round(time.time() - t0, 2)}
      lf.write(json.dumps(rec) + "\n")
      lf.flush()
      if it % 10 == 0 or it == a.updates - 1:
        print(f"[{a.student}] update {it} beta {beta:.3f} loss {rec['loss_total']:.4f} act {rec['loss_action']:.4f} "
              f"pos {rec['loss_pos']:.3f} ori {rec['loss_ori']:.3f} joint {rec['loss_joint']:.4f} done/step {rec['done_rate_per_step']:.4f} "
              f"({rec['seconds']}s)", flush=True)
      if (it + 1) % 100 == 0 or it == a.updates - 1:
        M.save(student, out / f"ckpt_{it + 1:04d}.pt", meta=meta, update=it + 1)
  M.save(student, out / "final.pt", meta=meta, update=a.updates)
  print(f"done {a.updates} updates in {(time.time() - t_start) / 3600:.2f} h; final {out / 'final.pt'}", flush=True)


if __name__ == "__main__":
  main()
