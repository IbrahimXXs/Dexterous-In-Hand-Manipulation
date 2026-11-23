#!/usr/bin/env python3
"""C5-HW-V1 design support (simulation only, evaluation only, no training).

Runs the frozen hardware-trial checkpoints (A3, C5, R1, P0; same sha256 as on the hand) in the
training environment from the hardware start grasp (cache grasp 49) and records, per 20 Hz
control step and environment, in the hardware log's step convention:

  q[k]       joint positions read before step k          (hardware: q_hw[k], read at loop start)
  cmd[k]     the policy's position target for step k      (hardware: command_hw[k], written after the read)
  force[k]   actuator force during step k (end of step)   (no hardware equivalent; current is the analogue)
  limit      per-environment, per-actuator effort limit   (domain-randomised at reset)

so that the hardware audit's lag, tracking-error and saturation metrics can be computed on
simulation with the *same code* (LEAP_Hand_RL_Deploy_Kit/kit/audit_hardware_logs.py).  Joint order
is the simulation (server) order; the hardware audit maps hardware order to it.

Uses the unchanged environment builder of grasp_rotation_v2 (rollout.make_env) and the task
Mjlab-Leap-RetV1-A (identical physics, observations, actions and randomisation for all four
policies; only the reward differs, which plays no part in an evaluation rollout).

    python experiments/c5_hw_v1/scripts/sim_actuator_reference.py --num-envs 256 --steps 600 --seed 8301
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "experiments" / "c5_hw_v1" / "sim_reference"
POLICIES = {  # the checkpoints that ran on the hand on 20 Nov (sha256 checked below)
  "A3": ("experiments/hwlimit_ac/checkpoints/A_seed3_selected_model_2600.pt", "38a2922b7d00ec77"),
  "C5": ("experiments/grasp_rotation_v2/checkpoints/C5_1_selected_model_2400.pt", "2a932dfe37a5540d"),
  "R1": ("experiments/retention_v1/checkpoints/R1_selected_model_2600.pt", "3b8ad077ff434157"),
  "P0": ("experiments/grasp_rotation_v2/checkpoints/P0_1_selected_model_1500.pt", "2bc2ce11d551e8bd"),
}
HARDWARE_GRASP = 49  # the cube placement used on the hand (docs/hardware_trials_A3_C5_R1_P0.md)


@torch.no_grad()
def record(policy: str, seed: int, num_envs: int, steps: int, device: str,
           effort_limit: tuple[float, float] | None = None) -> dict[str, np.ndarray]:
  """``effort_limit`` (diagnostic only) overrides the reset-time effort-limit randomisation range
  of the evaluation environment (training's range is 0.262-0.321 Nm); nothing is trained."""
  from in_hand_rotation_mjlab.grasp_rotation_v2 import rollout as R
  from in_hand_rotation_mjlab.hwlimit_study.evaluation import load_study_policy
  from in_hand_rotation_mjlab.scratch_study.geometry import yaw_increment

  ck, sha16 = POLICIES[policy]
  ck_path = PROJECT / ck
  assert hashlib.sha256(ck_path.read_bytes()).hexdigest().startswith(sha16), f"{policy}: checkpoint is not the hardware one"
  env, raw, agent_cfg = _make_env(R, num_envs, seed, device, steps, effort_limit)
  raw._gr2_assignment = torch.full((num_envs,), HARDWARE_GRASP, dtype=torch.long, device=device)
  actor = load_study_policy(env, agent_cfg, str(ck_path), device)
  robot = raw.scene["robot"]
  term = raw.action_manager.get_term("joint_pos")
  joint_names = list(robot.joint_names)
  act_names = list(robot.actuator_names) if hasattr(robot, "actuator_names") else joint_names
  ctrl_ids = robot.indexing.ctrl_ids
  T, n = steps, num_envs
  q = np.zeros((T, n, 16), np.float32)
  cmd = np.zeros((T, n, 16), np.float32)
  force = np.zeros((T, n, 16), np.float32)
  alive = np.zeros((T, n), bool)
  yaw = np.zeros((T, n), np.float32)
  cube = raw.scene["cube"]
  env.reset()
  obs = env.get_observations()
  fr = raw.sim.model.actuator_forcerange
  limit = (fr[:, ctrl_ids, 1] if fr.ndim == 3 else fr[ctrl_ids, 1][None].expand(n, -1)).clone().cpu().numpy()
  kp = None
  active = torch.ones(n, dtype=torch.bool, device=device)
  for t in range(T):
    q[t] = robot.data.joint_pos.cpu().numpy()  # read before the step (hardware q_hw[k])
    qc = cube.data.root_link_quat_w.clone()
    obs, _, dones, _ = env.step(actor(obs))
    yaw[t] = torch.where(active & ~dones.bool(), yaw_increment(cube.data.root_link_quat_w, qc), 0.0).cpu().numpy()
    cmd[t] = term._target.cpu().numpy()  # target applied during step t (hardware command_hw[k])
    force[t] = robot.data.actuator_force.cpu().numpy()
    alive[t] = active.cpu().numpy()
    active &= ~dones.bool()
  env.close()
  return {"q": q, "cmd": cmd, "force": force, "alive": alive, "effort_limit": limit, "yaw_inc": yaw,
          "effort_limit_override": np.array(effort_limit if effort_limit else (np.nan, np.nan)), "dt": np.float32(raw.step_dt),
          "joint_names": np.array(joint_names), "actuator_names": np.array(act_names), "policy": np.array(policy),
          "seed": np.int64(seed), "grasp": np.int64(HARDWARE_GRASP)}


def _make_env(R, num_envs, seed, device, steps, effort_limit):
  """rollout.make_env, optionally with the effort-limit randomisation range replaced (diagnostic)."""
  if effort_limit is None:
    return R.make_env("Mjlab-Leap-RetV1-A", num_envs, seed, device, num_steps=steps)
  import mjlab.tasks.registry as reg  # rollout.make_env imports load_env_cfg from here at call time
  base = reg.load_env_cfg

  def patched(task, *a, **k):
    cfg = base(task, *a, **k)
    cfg.events["dr_effort_limits"].params["effort_limit"] = tuple(effort_limit)
    return cfg

  reg.load_env_cfg = patched
  try:
    return R.make_env("Mjlab-Leap-RetV1-A", num_envs, seed, device, num_steps=steps)
  finally:
    reg.load_env_cfg = base


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--policies", default="A3,C5,R1,P0")
  ap.add_argument("--seed", type=int, default=8301)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=600)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--effort-limit", default=None,
                  help="lo,hi Nm: diagnostic override of the effort-limit randomisation (e.g. 0.167,0.204 = 350 mA x 0.53 Nm/A +-10%%)")
  ap.add_argument("--tag", default="")
  a = ap.parse_args()
  el = tuple(float(x) for x in a.effort_limit.split(",")) if a.effort_limit else None
  OUT.mkdir(parents=True, exist_ok=True)
  for p in a.policies.split(","):
    f = OUT / f"sim_{p}{a.tag}_grasp{HARDWARE_GRASP}_seed{a.seed}_n{a.num_envs}.npz"
    if f.exists():
      print("exists:", f)
      continue
    r = record(p, a.seed, a.num_envs, a.steps, a.device, el)
    np.savez_compressed(f, **r)
    print(f"{p}: saved {f}  joint order {list(r['joint_names'])[:4]}... actuators {list(r['actuator_names'])[:4]}...", flush=True)
  (OUT / "README.txt").write_text(json.dumps({"generator": "experiments/c5_hw_v1/scripts/sim_actuator_reference.py",
                                              "task": "Mjlab-Leap-RetV1-A", "grasp": HARDWARE_GRASP, "seed": a.seed,
                                              "num_envs": a.num_envs, "steps": a.steps,
                                              "policies": {k: v[0] for k, v in POLICIES.items()}}, indent=1) + "\n")


if __name__ == "__main__":
  main()
