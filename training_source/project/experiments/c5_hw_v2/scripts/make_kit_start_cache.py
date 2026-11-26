#!/usr/bin/env python3
"""C5-HW-V2 step B: kit-start reset cache (FIT_PLAN.md section B; simulation only, no policy involved).

Reproduces the deploy kit's start sequence in simulation, at the deployed 350 mA-equivalent effort range
(task Mjlab-Leap-C5HW-EFFORT; all other randomisation the task's own):

  1. hand at the kit's home pose (0 rad, clipped into the approved command limits), held by the motors
  2. cube flat at grasp 49's position + 10 mm, grasp 49's yaw; noise U(+-4 mm) in x, y and U(+-15 deg) yaw
  3. settle 1.0 s under gravity
  4. 5.0 s linear closing ramp of the joint targets from home to the policy default pose (kit INIT_DURATION_S)
  5. record measured joints + cube pose; the stored joint target IS the measured pose (the kit resets its
     previous command to the measured post-ramp pose: deploy_direct.py / mode kit_reset_measured)
  6. keep starts with the cube within 25 mm (xy) of grasp 49, not more than 15 mm below it, >= 2 fingertip contacts

Also writes the comparison of the simulated post-ramp joint pose with the hardware post-ramp pose
(q_hw[0] of every C5-family run log).

    python experiments/c5_hw_v2/scripts/make_kit_start_cache.py [--num-envs 4096 --seed 8501]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
REPO = PROJECT.parents[1]
E = PROJECT / "experiments" / "c5_hw_v2"
KIT = REPO / "LEAP_Hand_RL_Deploy_Kit"
GRASP_CACHE = PROJECT / "experiments" / "hwlimit_ac" / "cache" / "hwlimit_grasp_cache.npz"
TASK = "Mjlab-Leap-C5HW-EFFORT"
GRASP = 49
SETTLE_S, RAMP_S, DROP_Z = 1.0, 5.0, 0.010
NOISE_XY, NOISE_YAW = 0.004, math.radians(15.0)
KEEP_XY, KEEP_BELOW, MIN_CONTACTS = 0.025, 0.015, 2


def audit():
  spec = importlib.util.spec_from_file_location("audit_hardware_logs", KIT / "kit" / "audit_hardware_logs.py")
  mod = importlib.util.module_from_spec(spec)
  sys.modules["audit_hardware_logs"] = mod
  spec.loader.exec_module(mod)
  return mod


def quat_wxyz_to_yaw(q: np.ndarray) -> float:
  w, x, y, z = q
  return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  h = 0.5 * yaw
  return torch.stack([torch.cos(h), torch.zeros_like(h), torch.zeros_like(h), torch.sin(h)], -1)


def hand_contact_pairs(raw, cube, robot) -> torch.Tensor:
  """Per environment: number of active cube-hand contact pairs (any robot geom, palm included)."""
  n = raw.num_envs
  out = torch.zeros(n, dtype=torch.long, device=raw.device)
  nacon = int(raw.sim.data.nacon[0])
  if nacon == 0:
    return out
  cg, _ = cube.find_geoms("cube_geom", preserve_order=True)
  cid = int(cube.indexing.geom_ids[cg[0]].item())
  rids = robot.indexing.geom_ids.flatten().to(raw.device, dtype=torch.long)
  geom = raw.sim.data.contact.geom[:nacon].to(torch.long)
  world = raw.sim.data.contact.worldid[:nacon].to(torch.long)
  a, b = geom[:, 0] == cid, geom[:, 1] == cid
  other = torch.where(a, geom[:, 1], geom[:, 0])
  rel = (a | b) & torch.isin(other, rids)
  return torch.bincount(world[rel], minlength=n)


def hardware_post_ramp(A) -> dict:
  """q_hw[0] (first policy-loop read = measured post-ramp pose) of every C5-family run log, server order."""
  rows, files = [], []
  for f in sorted(glob.glob(str(KIT / "logs" / "physical_*.npz"))):
    if "_part" in f:
      continue
    d = np.load(f, allow_pickle=False)
    if "run_meta_json" not in d.files or "q_hw" not in d.files or len(d["q_hw"]) == 0:
      continue
    pol = Path(json.loads(str(d["run_meta_json"])).get("policy_dir", "")).name
    if pol not in ("C5", "C5s2", "C5s3"):
      continue
    rows.append(d["q_hw"][0][A.HW_TO_SERVER].astype(np.float64))
    files.append(Path(f).name)
  Q = np.stack(rows)
  return {"files": files, "n": len(files), "median": np.median(Q, 0), "iqr": np.percentile(Q, [25, 75], axis=0)}


@torch.no_grad()
def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--seed", type=int, default=8501)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", type=Path, default=E / "cache" / "kit_start_cache.npz")
  a = ap.parse_args()

  from in_hand_rotation_mjlab.estimation.sim import _import_tasks
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R
  from mjlab.tasks.registry import load_env_cfg

  A = audit()
  _import_tasks()
  terms = tuple(load_env_cfg(TASK).terminations)
  n, dev = a.num_envs, a.device
  g = np.load(GRASP_CACHE, allow_pickle=False)
  p49 = g["cube_pose_rel"][GRASP][:3].astype(np.float64)
  yaw49 = quat_wxyz_to_yaw(g["cube_pose_rel"][GRASP][3:].astype(np.float64))
  steps_settle = None
  env, raw, _ = R.make_env(TASK, n, a.seed, dev, num_steps=400, terminations_off=terms)
  dt = float(raw.step_dt)
  steps_settle, steps_ramp = round(SETTLE_S / dt), round(RAMP_S / dt)
  raw._gr2_assignment = torch.full((n,), GRASP, dtype=torch.long, device=dev)
  robot, cube = raw.scene["robot"], raw.scene["cube"]
  sensor = raw.scene["fingertip_cube_contact"]
  term = raw.action_manager.get_term("joint_pos")
  alo, ahi = term.approved_limits()
  jn = [str(x) for x in robot.joint_names]
  assert tuple(jn) == A.SERVER_JOINTS, jn
  targets = {"x": None}

  def process_actions(actions):  # motors get exactly the scripted targets (as KitRunner does)
    term._raw_actions[:] = actions
    term._prev_target[:] = term._target
    term._substep_counter = 0
    term._target = targets["x"].clone()
    term._processed_actions = term._target

  term.process_actions = process_actions
  env.reset()
  assert torch.equal(raw._hand_cube_reset_cache_ids, raw._gr2_assignment)
  origin = raw.scene.env_origins
  gen = torch.Generator(device=dev).manual_seed(a.seed)
  u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, device=dev, generator=gen)  # noqa: E731
  dx, dy, dyaw = u(-NOISE_XY, NOISE_XY), u(-NOISE_XY, NOISE_XY), u(-NOISE_YAW, NOISE_YAW)
  pos = torch.as_tensor(p49, dtype=torch.float32, device=dev).expand(n, -1).clone()
  pos[:, 0] += dx
  pos[:, 1] += dy
  pos[:, 2] += DROP_Z
  pose = torch.cat([pos + origin, yaw_quat(yaw49 + dyaw)], -1)
  q_home = torch.clamp(torch.zeros(16, device=dev), alo, ahi).expand(n, -1).clone()
  q_default = robot.data.default_joint_pos.clone()
  robot.write_joint_state_to_sim(q_home, torch.zeros_like(q_home))
  cube.write_root_link_pose_to_sim(pose)
  cube.write_root_link_velocity_to_sim(torch.zeros((n, 6), device=dev))
  zero = torch.zeros((n, 16), device=dev)
  targets["x"] = q_home
  for _ in range(steps_settle):
    env.step(zero)
  settled_z = (cube.data.root_link_pos_w[:, 2] - origin[:, 2]).clone()
  for k in range(1, steps_ramp + 1):
    targets["x"] = q_home + (k / steps_ramp) * (q_default - q_home)
    env.step(zero)
  q = robot.data.joint_pos.clone()
  cpose = torch.cat([cube.data.root_link_pos_w - origin, cube.data.root_link_quat_w], -1).clone()
  contact = (sensor.data.found > 0).clone()
  hand_pairs = hand_contact_pairs(raw, cube, robot)
  F = robot.data.actuator_force.abs()
  L = raw.sim.model.actuator_forcerange[:, robot.indexing.ctrl_ids, 1]
  sat_end = (F / L.clamp(min=1e-9) >= 0.98).float().mean(0).cpu().numpy()
  env.close()

  dxy = torch.linalg.vector_norm(cpose[:, :2] - torch.as_tensor(p49[:2], dtype=torch.float32, device=dev), dim=-1)
  low = cpose[:, 2] < float(p49[2]) - KEEP_BELOW
  far = dxy > KEEP_XY
  few = contact.sum(-1) < MIN_CONTACTS
  finite = torch.isfinite(cpose).all(-1) & torch.isfinite(q).all(-1)
  # DEVIATION from FIT_PLAN.md B.6 (recorded before any policy evaluation): the ">= 2 fingertip contacts" filter is
  # not applied.  The reference grasp 49 itself is held by the finger bases / proximal links and the palm, with 0-1
  # fingertip-sensor contacts per step (checked 26 Nov on the unmodified cache state), so the filter rejected 100% of
  # starts in the smoke test.  The kit does not reject starts on contacts either.  Fingertip and all-hand contact
  # counts are reported for information.
  ok = ~low & ~far & finite
  sel = torch.nonzero(ok).flatten()
  target = torch.clamp(q, alo, ahi)  # the action clamps every command into the approved limits
  out = {
    "cube_size": np.full(len(sel), float(g["cube_size"][GRASP]), np.float32),
    "joint_pos": q[sel].cpu().numpy().astype(np.float32),
    "joint_pos_target": target[sel].cpu().numpy().astype(np.float32),
    "cube_pose_rel": cpose[sel].cpu().numpy().astype(np.float32),
    "joint_names": np.array(jn),
    "fingertip_contact": contact[sel].cpu().numpy().astype(np.uint8),
    "source_checkpoint": np.array(f"kit-start simulation of grasp {GRASP} (no policy): experiments/c5_hw_v2/scripts/make_kit_start_cache.py seed {a.seed}"),
    "source_index": sel.cpu().numpy().astype(np.int32),
  }
  a.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(a.out, **out)
  hw = hardware_post_ramp(A)
  sim_med = np.median(out["joint_pos"], 0)
  diff = sim_med - hw["median"]
  summary = {
    "plan_sha256": (E / "logs" / "fit_plan.sha256").read_text().split()[0], "task": TASK, "seed": a.seed, "num_envs": n,
    "dt_s": dt, "settle_steps": steps_settle, "ramp_steps": steps_ramp, "grasp": GRASP, "grasp49_pos": p49.tolist(),
    "grasp49_yaw_deg": math.degrees(yaw49), "cube_size": float(g["cube_size"][GRASP]),
    "kept": int(len(sel)), "kept_fraction": float(ok.float().mean()),
    "reject_breakdown_not_exclusive": {"cube_low": float(low.float().mean()), "cube_off_xy": float(far.float().mean()),
                                       "fewer_than_2_tip_contacts": float(few.float().mean()), "non_finite": float((~finite).float().mean())},
    "settled_cube_z_minus_grasp49_mm": {"median": float(1000 * (settled_z.median() - float(p49[2]))),
                                        "p5": float(1000 * (torch.quantile(settled_z, 0.05) - float(p49[2]))),
                                        "p95": float(1000 * (torch.quantile(settled_z, 0.95) - float(p49[2])))},
    "post_ramp_cube_offset_mm_all_starts": {"xy_median": float(1000 * dxy.median()), "z_median": float(1000 * (cpose[:, 2].median() - float(p49[2]))),
                                         "z_p10": float(1000 * (torch.quantile(cpose[:, 2], 0.1) - float(p49[2]))), "tip_contacts_mean": float(contact.float().sum(-1).mean())},
    "deviation_from_plan": "B.6 '>= 2 fingertip contacts' not applied (reference grasp 49 shows 0-1 fingertip-sensor contacts while held; the filter rejected 100% in the smoke test); position filters kept. Recorded before any policy evaluation.",
    "info_fingertip_contacts_ge2_fraction_all": float((~few).float().mean()),
    "info_hand_contact_pairs_kept": {"median": float(hand_pairs[ok].float().median()) if ok.any() else None,
                                     "zero_fraction": float((hand_pairs[ok] == 0).float().mean()) if ok.any() else None},
    "mean_fingertip_contacts_kept": float(out["fingertip_contact"].sum(1).mean()),
    "per_fingertip_contact_fraction_kept": out["fingertip_contact"].mean(0).tolist(),
    "actuator_saturation_frac_at_ramp_end_per_joint": dict(zip(jn, sat_end.round(4).tolist())),
    "target_clamped_max_rad": float((target - q)[sel].abs().max()) if len(sel) else None,
    "hardware_post_ramp": {"n_logs": hw["n"], "files": hw["files"], "median": dict(zip(jn, hw["median"].round(4).tolist())),
                           "iqr": {j: hw["iqr"][:, i].round(4).tolist() for i, j in enumerate(jn)}},
    "sim_post_ramp_median": dict(zip(jn, sim_med.round(4).tolist())),
    "sim_minus_hw_post_ramp_rad": dict(zip(jn, diff.round(4).tolist())),
    "sim_minus_hw_abs_median_rad": float(np.median(np.abs(diff))), "sim_minus_hw_abs_max_rad": float(np.abs(diff).max()),
    "grasp49_cache_joint_pos": dict(zip(jn, g["joint_pos"][GRASP].round(4).tolist())),
    "cache_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
  }
  a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
  print(json.dumps({k: v for k, v in summary.items() if k not in ("hardware_post_ramp", "grasp49_cache_joint_pos")}, indent=1))


if __name__ == "__main__":
  main()
