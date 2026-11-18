#!/usr/bin/env python3
"""Build the reset-state (grasp) cache of the hardware-limit study.

The original cache (``tasks/hand_cube/cache/leap_right_mbzuai_58mm_policy_grasp_cache.npz``,
449 grasps, read-only) has only ~2% of its grasps inside the deployment kit's
approved joint limits.  For each original grasp:

1. Projection: joint state and joint command are clipped into the approved
   command limits (kit limits with the 0.01 rad margin, intersected with the
   model's soft limits; ``hwlimit_study.limits``).  Cube pose and size are kept.
2. Settling: the projected grasps are loaded into the hardware-limit
   environment (training domain randomisation, 4096 environments, every grasp
   drawn ~9 times) and the command is held (zero delta action) for
   ``--settle-s``.  The settled joint state, cube pose and fingertip contacts
   form the new reset state; the command stays the projected command.
3. Acceptance.  The retention criterion of ``scripts/validate_grasp_cache.py``
   (the property the original cache was validated for: 95% retention there)
   over a ``--hold-s`` hold after settling: height >= 0.20 m, cube-palm
   distance <= 0.15 m, no termination, finite state.  In addition the settled
   state itself must be clean: >= ``--min-contacts`` fingertip contacts,
   cube-hand penetration <= ``--penetration-mm`` and a joint state no more
   than ``--state-tol`` outside the approved limits.  (The original cache does
   not meet a stricter ">= 2 contacts" or "penetration during the whole hold"
   criterion: 58% and 66% in the original environment.)  A grasp is kept when
   at least ``--min-pass-rate`` of its randomised instances pass; its first
   passing instance becomes the cache entry.

Only simulator state is used (this is reset-state generation, not policy input).

    python experiments/hwlimit_ac/scripts/build_grasp_cache.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

E = Path("experiments/hwlimit_ac")
ORIG = Path("src/in_hand_rotation_mjlab/tasks/hand_cube/cache/leap_right_mbzuai_58mm_policy_grasp_cache.npz")


def penetration(env, cube_geom_id: int, robot_geom_ids: torch.Tensor) -> torch.Tensor:
  result = torch.zeros(env.num_envs, device=env.device)
  nacon = int(env.sim.data.nacon[0])
  if nacon == 0:
    return result
  geom = env.sim.data.contact.geom[:nacon].to(torch.long)
  dist = env.sim.data.contact.dist[:nacon]
  world = env.sim.data.contact.worldid[:nacon].to(torch.long)
  a, b = geom[:, 0] == cube_geom_id, geom[:, 1] == cube_geom_id
  other = torch.where(a, geom[:, 1], geom[:, 0])
  rel = (a | b) & torch.isin(other, robot_geom_ids)
  if torch.any(rel):
    result.scatter_reduce_(0, world[rel], torch.clamp(-dist[rel], min=0.0), reduce="amax", include_self=True)
  return result


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--seed", type=int, default=31)
  p.add_argument("--settle-s", type=float, default=1.0)
  p.add_argument("--hold-s", type=float, default=10.0)
  p.add_argument("--min-pass-rate", type=float, default=0.5)
  p.add_argument("--state-tol", type=float, default=0.02)
  p.add_argument("--penetration-mm", type=float, default=2.0)
  p.add_argument("--min-contacts", type=int, default=1)
  p.add_argument("--out", default=str(E / "cache" / "hwlimit_grasp_cache.npz"))
  p.add_argument("--device", default="cuda:0")
  a = p.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.utils.torch import configure_torch_backends

  import in_hand_rotation_mjlab.tasks  # noqa: F401
  from in_hand_rotation_mjlab.hwlimit_study.limits import approved_command_limits
  from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_hwlimit_study.env_cfgs import hwlimit_env_cfg

  configure_torch_backends()
  orig = dict(np.load(ORIG, allow_pickle=False))
  names = [str(x) for x in orig["joint_names"]]
  n_orig = len(orig["cube_size"])

  # soft limits from the model (same for all envs)
  probe_cfg = hwlimit_env_cfg(grasp_cache=ORIG)
  probe_cfg.scene.num_envs = 1
  probe = ManagerBasedRlEnv(cfg=probe_cfg, device=a.device)
  rob = probe.scene["robot"]
  assert tuple(rob.joint_names) == tuple(names), (rob.joint_names, names)
  soft = rob.data.soft_joint_pos_limits[0].cpu().numpy().astype(np.float64)
  probe.close()
  lo, hi = approved_command_limits(soft[:, 0], soft[:, 1])

  cand = dict(orig)
  cand["joint_pos"] = np.clip(orig["joint_pos"], lo, hi).astype(np.float32)
  cand["joint_pos_target"] = np.clip(orig["joint_pos_target"], lo, hi).astype(np.float32)
  cand_path = Path(a.out).with_name("_candidate_projected.npz")
  cand_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(cand_path, **cand)
  proj_shift = np.abs(cand["joint_pos_target"] - orig["joint_pos_target"])

  cfg = hwlimit_env_cfg(grasp_cache=cand_path)
  cfg.scene.num_envs = a.num_envs
  cfg.seed = a.seed
  cfg.curriculum = {}
  dt = cfg.sim.mujoco.timestep * cfg.decimation
  n_settle, n_hold = round(a.settle_s / dt), math.ceil(a.hold_s / dt)
  cfg.episode_length_s = (n_settle + n_hold + 5) * dt
  env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
  robot, cube = env.scene["robot"], env.scene["cube"]
  sensor = env.scene["fingertip_cube_contact"]
  palm_ids, _ = robot.find_geoms("palm_collision_.*", preserve_order=True)
  cg, _ = cube.find_geoms("cube_geom", preserve_order=True)
  cube_geom_id = int(cube.indexing.geom_ids[cg[0]].item())
  robot_geom_ids = robot.indexing.geom_ids.flatten().to(env.device, dtype=torch.long)
  term = env.action_manager.get_term("joint_pos")
  alo, ahi = term.approved_limits()
  assert np.allclose(alo.cpu().numpy(), lo, atol=1e-6) and np.allclose(ahi.cpu().numpy(), hi, atol=1e-6)

  env.reset()
  ids = env._hand_cube_reset_cache_ids.clone()
  N = a.num_envs
  zero = torch.zeros((N, env.action_manager.total_action_dim), device=env.device)
  ok = torch.ones(N, dtype=torch.bool, device=env.device)
  settled = {}
  drop = torch.zeros(N, dtype=torch.bool, device=env.device)
  origin = env.scene.env_origins
  with torch.inference_mode():
    for t in range(n_settle + n_hold):
      _, _, terminated, time_out, _ = env.step(zero)
      done = terminated.bool() | time_out.bool()
      palm = robot.data.geom_pos_w[:, palm_ids].mean(dim=1)
      pdist = torch.linalg.vector_norm(cube.data.root_link_pos_w - palm, dim=-1)
      finite = torch.isfinite(cube.data.root_link_pos_w).all(-1) & torch.isfinite(robot.data.joint_pos).all(-1)
      bad = done | (cube.data.root_link_pos_w[:, 2] < 0.20) | (pdist > 0.15) | ~finite
      ok &= ~bad
      drop |= bad
      if t == n_settle - 1:
        pen_settled = penetration(env, cube_geom_id, robot_geom_ids)
        found = (sensor.data.found > 0)
        settled = {
          "joint_pos": robot.data.joint_pos.clone(),
          "cube_pose_rel": torch.cat([cube.data.root_link_pos_w - origin, cube.data.root_link_quat_w], -1).clone(),
          "contact": found.clone(),
          "command": term._target.clone(),
        }
  q = settled["joint_pos"]
  excess = torch.maximum(alo - q, q - ahi).clamp_min(0).amax(-1)
  few_contacts = settled["contact"].sum(-1) < a.min_contacts
  deep = pen_settled > a.penetration_mm / 1000
  outside = excess > a.state_tol
  ok &= ~few_contacts & ~deep & ~outside
  fail_breakdown = {"dropped_or_terminated": float(drop.float().mean()), "few_contacts_at_settle": float(few_contacts.float().mean()),
                    "penetration_at_settle": float(deep.float().mean()), "state_outside_limits": float(outside.float().mean())}
  ok_np, ids_np = ok.cpu().numpy(), ids.cpu().numpy()

  # command after settling must equal the projected command (zero delta)
  cmd_err = (settled["command"] - torch.as_tensor(cand["joint_pos_target"], device=env.device)[ids])[ok].abs().max().item()

  keep, first_inst, rates, counts = [], [], np.zeros(n_orig), np.zeros(n_orig, int)
  for k in range(n_orig):
    inst = np.flatnonzero(ids_np == k)
    counts[k] = len(inst)
    if len(inst) == 0:
      continue
    rates[k] = ok_np[inst].mean()
    if len(inst) >= 3 and rates[k] >= a.min_pass_rate:
      keep.append(k)
      first_inst.append(inst[ok_np[inst]][0])
  keep, first_inst = np.array(keep, int), np.array(first_inst, int)
  sel = torch.as_tensor(first_inst, device=env.device)
  out = {
    "cube_size": orig["cube_size"][keep].astype(np.float32),
    "joint_pos": settled["joint_pos"][sel].cpu().numpy().astype(np.float32),
    "joint_pos_target": cand["joint_pos_target"][keep].astype(np.float32),
    "cube_pose_rel": settled["cube_pose_rel"][sel].cpu().numpy().astype(np.float32),
    "joint_names": orig["joint_names"],
    "fingertip_contact": settled["contact"][sel].cpu().numpy().astype(np.uint8),
    "source_checkpoint": np.array(f"projected+settled from {ORIG.name} (source: {orig['source_checkpoint']})"),
    "source_index": keep.astype(np.int32),
  }
  np.savez(a.out, **out)
  env.close()
  cand_path.unlink()

  q_in = out["joint_pos"]
  summary = {
    "original_entries": n_orig,
    "original_entries_fully_inside_limits_state": int(((orig["joint_pos"] >= lo) & (orig["joint_pos"] <= hi)).all(1).sum()),
    "original_entries_fully_inside_limits_command": int(((orig["joint_pos_target"] >= lo) & (orig["joint_pos_target"] <= hi)).all(1).sum()),
    "projection_command_shift_rad": {"mean_of_max_per_grasp": float(proj_shift.max(1).mean()), "max": float(proj_shift.max()),
                                     "per_joint_mean": dict(zip(names, proj_shift.mean(0).round(4).tolist()))},
    "settle_envs": N, "instances_per_entry_min_mean": [int(counts.min()), float(counts.mean())],
    "instance_pass_rate": float(ok_np.mean()),
    "instance_fail_breakdown (not exclusive)": fail_breakdown,
    "entry_pass_rate_mean": float(rates.mean()),
    "kept_entries": int(len(keep)),
    "kept_fraction": float(len(keep) / n_orig),
    "settled_command_equals_projection_max_err": cmd_err,
    "new_cache_state_max_excess_rad": float(np.maximum(lo - q_in, q_in - hi).clip(min=0).max()),
    "new_cache_mean_fingertip_contacts": float(out["fingertip_contact"].sum(1).mean()),
    "new_cache_per_fingertip_contact_fraction": out["fingertip_contact"].mean(0).tolist(),
    "approved_command_limits": {n: [round(float(l), 6), round(float(h), 6)] for n, l, h in zip(names, lo, hi)},
    "args": vars(a),
  }
  Path(a.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
  print(json.dumps({k: v for k, v in summary.items() if k != "approved_command_limits"}, indent=2))


if __name__ == "__main__":
  main()
