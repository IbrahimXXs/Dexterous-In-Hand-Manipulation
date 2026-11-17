# LEAP Hand hardware session, 17 Nov 2026: model_400, A and C on the real hand

**Status:** the hand was connected, function-tested, and model_400, A (seed 2) and C (seed 2) each ran on it. The A vs C comparison is **not done yet**. Cube rotation was not measured on the hand, and one deployment constraint (the controller's joint limits) was found to hurt C far more than A in simulation. That constraint has to be settled before a fair hardware comparison.

## Summary

- **Setup checks out.** The deploy kit (`LEAP_Hand_RL_Deploy_Kit/`) drives all 16 motors. A new function test passed every stage, and the operator confirmed each of the 16 joints by eye.
- **The kit now runs the study policies.** It gained a tanh action mapping (used by A and C), a fix for `clip_actions: null`, and a NumPy port of estimator B for C. The kit's outputs match the training networks to about 1e-5 for all six checkpoints.
- **Hardware runs completed.** model_400 ×2, A_seed2 ×3 and C_seed2 ×3 all finished with no guard trips. Current stayed ≤ 390 mA (limit 350 mA set on the motors, abort at 450), temperature ≤ 24 °C.
- **Main finding (simulation).** The kit clamps the sideways (rot) joints to ±0.52 rad, while the policies were trained with ±1.05 rad. In a 30 s closed-loop sim rehearsal of the kit runtime, this costs A 0.14 turns but costs C 0.59 turns. Under the current limits, C (0.53) falls below A (0.62). Without them, C reaches 1.01 and A 0.72.
- **The limits are active on the real hand.** In 87–98% of the steps of every hardware run, at least one joint was commanded at the controller limit, for all three policies.

## 1. What was built

| File | Purpose |
|---|---|
| `LEAP_Hand_RL_Deploy_Kit/` | Professor's torch-free deploy kit, added to the repo with the changes below |
| `LEAP_Hand_RL_Deploy_Kit/kit/function_test.py`, `./leap.sh functest [sim]` | Basic hardware test with no policy: read-only check, hold at the measured pose, flat pose, one-motor-at-a-time sweep confirmed by the operator, policy start pose, 20 Hz tracking. Always returns home with torque off. |
| `LEAP_Hand_RL_Deploy_Kit/kit/policy_runtime.py` | Adds `squash: tanh` support (read from `env.yaml`) and fixes `clip_actions: null`, which the kit had treated as clipping to 1.0. model_400 behaviour is unchanged. |
| `LEAP_Hand_RL_Deploy_Kit/kit/cube_estimator.py` | NumPy estimator B and the 19 C features (canonical cube rotation over the 24 symmetries). It re-checks itself against training-side features every time it loads. |
| `LEAP_Hand_RL_Deploy_Kit/policies/{A,C}_seed{1,2,3}/` | Ready-to-run folders: checkpoint, regenerated `params/env.yaml` + `agent.yaml`, and `cube_estimator.npz` for C |
| `training_source/project/scripts/estimation/export_kit_estimator.py` | Writes `cube_estimator.npz` from `estimator_B.pt` |
| `training_source/project/experiments/scratch_state_estimation/scripts/rehearse_deploy.py` | Sim rehearsal: the kit runtime in closed loop in the study environment, with motors executing exactly its clipped targets |
| `training_source/project/src/in_hand_rotation_mjlab/scratch_study/deploy.py`, `scripts/estimation/export_scratch_policy.py`, `checkpoint/scratch_study/`, `policy/scratch_policy.py`, `scripts/serve_scratch_policy.py` | Alternative ZeroMQ deployment path: TorchScript bundles of A and C that also output the cube-state and fingertip-force estimates. Not used on the hand this session. |
| `training_source/project/tests/test_scratch_deploy.py` | The bundles equal the rsl_rl training actors (7 tests pass) |

## 2. Hardware session (team-observed)

Logs are in `LEAP_Hand_RL_Deploy_Kit/logs/`. The kit does not record which policy a run used. The policy column is inferred from the shadow-pass markers: a powered run only unlocks after a shadow pass of that exact policy folder (model_400 09:52, A_seed2 10:05, C_seed2 10:19).

**Function test** (`functest_20261117_094855.npz`): all stages passed.
- Sweep: reach 88–99% of the commanded move, crosstalk ≤ 0.029 rad, ≤ 157 mA, all 16 joints confirmed by the operator.
- Policy start pose (no cube): error ≤ 0.042 rad, holding current ≤ 63 mA.
- 20 Hz tracking: loop period 50.2 ms, RMS error 0.025 rad, response lag about 2 steps (~100 ms).

**Policy runs** (all completed, `stop_reason = max_steps`, no stall recoveries):

| Log `physical_20261117_*` | Policy (inferred) | Length | Max current | Max tracking error | Steps with a joint at the controller limit |
|---|---|---|---|---|---|
| 095231, 100410 | model_400 | 6 s each | 117 mA | 0.08 rad | 98% (if_rot, if_dip, mf_mcp, th_ipl) |
| 100529 | A_seed2 | 6 s | 379 mA | 0.49 rad | 88% |
| 100557, 100745 | A_seed2 | 30 s each | 389 mA | 0.51 rad | 87–93% (rot joints, dips, th_cmc) |
| 102021, 102122, 102226 | C_seed2 | 6 s each | 390 mA | 0.38 rad | 94–97% (if_rot up to 97%, mf_rot, rf_rot) |

**Not measured:** cube rotation (turns), drops and failures on the hand. There is no camera or motion capture, and the outcomes were not written down. Count turns from video in future runs.

## 3. Problems found and how they were handled

1. **Serial port.** The user was not in the `dialout` group, the FTDI latency was 16 ms, and `brltty` was installed. Fixed for the session with `chmod` plus a 1 ms latency timer. Run `./install.sh` once to make this permanent.
2. **Resting pose outside the controller limits.** With torque off, the PIPs hang back to about −25°, outside the kit's −15° limit, and the first function test refused to start. Fix: the test now fails only on mechanically impossible readings (a 90° horn or homing error), holds the resting pose at torque-on, and only moves out-of-limit joints back toward the range.
3. **Action mapping.** A and C were trained with `tanh(u)`. The kit hard-clipped the action, and read `clip_actions: null` as 1.0. Both are fixed and verified (kit equals training to 4e-6 for A).
4. **First rehearsal mismatch (0.6 rad) was an artifact.** In that first version, the simulator integrated its own targets instead of executing the runtime's. With the motors executing the runtime's targets, the kit's observation equals the training observation to 1e-7.
5. **Reset semantics.** The kit resets the previous command to the measured pose. In training it was the grasp target (fingers rest short of it on the cube). In simulation the two resets differ by at most 0.03 turns for A and for C (table in Section 4).
6. **Controller joint limits (open).** The kit uses the `leap_hand_controller` limits: rot ±0.52 rad, thumb cmc/axl 0 to 1.571, mcp/pip/dip ≥ −0.26. Training used the model's mechanical range: rot ±1.05, pip down to −0.51, thumb cmc down to −0.35. 65% of the training grasp poses already have `if_rot` below −0.52. This is the dominant sim-to-real factor found (Section 4). It is a safety setting in the course kit, so widening it needs the TA's approval and a check that the fingers really reach the range without colliding.
7. **Actuator lag (unconfirmed).** The hand tracks about 2 control steps (~100 ms) behind. Training randomized the actuator delay over 25–75 ms. The measured lag includes tracking dynamics, so it is only an indication.
8. **Not fixed: legacy runner normalization.** `policy/yaw_policy.py` (the repo's older robot runner) normalizes with `std + 1e-8`, while training used `+ 1e-2`. The kit is correct, so all hardware runs used the kit.

## 4. Simulation rehearsal of the deployment path

Kit runtime in closed loop, nominal preset, test seed 5001, 64 environments × 30 s, against the training actor in the unmodified environment (`results/deploy_rehearsal.json`, `results/deploy_rehearsal_nohwclip.json`). The ± values are 95% CIs of the paired per-environment difference against the reference evaluated in the same invocation. The reference is a range because the GPU simulation is not bit-reproducible between invocations.

| Turns in 30 s | A_seed2 | C_seed2 |
|---|---|---|
| Training actor, normal sim (reference) | 0.74–0.76 | 1.09–1.12 |
| Kit, controller limits, reset to measured pose (as on the hand) | 0.62 (−0.14 ± 0.10) | 0.53 (−0.59 ± 0.11) |
| Kit, controller limits, reset to grasp target | 0.59 (−0.17 ± 0.08) | 0.56 (−0.56 ± 0.10) |
| Kit, training joint limits, reset to measured pose | 0.72 (−0.02 ± 0.09) | 1.01 (−0.08 ± 0.09) |
| Kit, training joint limits, reset to grasp target | — | 1.03 (−0.07 ± 0.07) |

**Reading.** The joint limits explain the loss, and they hit C four times harder than A. A hardware A vs C comparison run under the current limits would mostly measure this constraint, not the value of the estimator. One seed per condition, one test seed, simulation only.

## 5. Next steps

1. **Decide the joint limits with the TA.** With torque off, check whether each finger reaches ±60° sideways without collisions. If it does, add a `--joint-limits training` option to the kit that widens only the joint limits (current, temperature, velocity and tracking guards unchanged). If it does not, keep the limits and report them as a limitation. Optionally, fine-tune A and C in simulation with the controller limits.
2. **Run the A vs C hardware comparison:** the same seeds, equal numbers of 30 s trials alternating A and C, the same cube placement, and video for turn counts. Record the policy folder for every log; the kit does not store it.
3. **Optionally log the estimators on the hand.** Write cube-state and force estimates to each log (`kit/cube_estimator.py` already keeps `last`), or reconstruct them offline from `q_hw` and `command_hw`.

## 6. Reproduce

```bash
cd LEAP_Hand_RL_Deploy_Kit
./install.sh                                  # once per machine (dialout, 1 ms latency, deps)
./leap.sh functest sim                        # function test on the simulated bus
./leap.sh check && ./leap.sh functest         # hardware
./leap.sh inspect policies/C_seed2            # shows action_squash: tanh and the cube_estimator line
./leap.sh shadow policies/A_seed2 && ./leap.sh run policies/A_seed2 -- --max-steps 600

cd ../training_source/project                 # sim rehearsal (MJLab stack, `uv sync --frozen`)
python scripts/estimation/export_kit_estimator.py --out ../../LEAP_Hand_RL_Deploy_Kit/policies/C_seed2/cube_estimator.npz
python experiments/scratch_state_estimation/scripts/rehearse_deploy.py \
    --policy ../../LEAP_Hand_RL_Deploy_Kit/policies/C_seed2 --cond C --reset measured --reset target
```
