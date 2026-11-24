# Real-hand trials, 24 Nov 2026: C5 (seed 1) vs C5 seed 3 (and optionally seed 2)

**Why:** on 20 Nov, C5 (seed 1) rotated fastest but completed only 4 of 11 attempts. Six attempts ended in safety stops: 5 sustained index-finger jams at the 350 mA limit, and one more trip at 3.1 s (`docs/hardware_policy_actuator_audit_2026-11-20.md`).

**No "low-jam" C5 has been trained yet.** C5-HW-V1 (training with the real 350 mA limit) is a specification only (`experiments/c5_hw_v1/SPEC.md`). The new checkpoints are **C5 seeds 2 and 3**, trained with the same recipe as seed 1. This session tests whether a different seed avoids the jam.

## Candidates

| Kit name | Checkpoint | Sim, 30 s turns / failure (full distribution) | Kit rehearsal (sim, as on the hand) |
|---|---|---|---|
| C5 | `C5_1_selected_model_2400` (tested 20 Nov) | 1.87 / 20% | −0.03 ± 0.11 turns |
| **C5s3** | `C5_3_selected_model_3000` | 1.69 / 20% | **−0.17 ± 0.12** turns (the PIP-limit gap; disappears with training limits) |
| C5s2 | `C5_2_selected_model_1500` | 1.83 / 15% | +0.01 ± 0.08 turns |

- **Kit folders:** `grasp_rotation_v2/deploy/policies/C5_seed{2,3}`, reached through the shortcuts `policies/C5s2` and `policies/C5s3`.
- **Verification:** kit = training to ≤ 1.9e-5; checkpoint sha256 values match `checkpoints/MANIFEST.txt`.

**Jam screen in simulation.** Same procedure as the audit: motors capped at the deployed 350 mA, grasp 49, 256 environments, 30 s (`experiments/c5_hw_v1/results/c5_seed_screen.json`).

| | C5 seed 1 | C5 seed 2 | C5 seed 3 |
|---|---|---|---|
| Index MCP/PIP saturated (share of time) | 7.5% | **1.7%** | 3.6% |
| Environments with the index saturated ≥ 1 s | 2.7% | 0% | 0.4% |
| Environments with **any** joint saturated ≥ 1 s | 3.5% (index MCP) | **42% (middle MCP)** | 5.1% (ring MCP) |
| Turns per 30 s, capped | 2.45 | 2.36 | 2.23 |

**Reading:**
- **Seed 3:** halves seed 1's index loading and nearly removes sustained index saturation, at about 9% less rotation. It is the main candidate.
- **Seed 2:** moves the load to the middle finger, with sustained saturation in 42% of environments. That is the same failure mechanism on another finger, so it is tested only as an optional extra.
- **Limit of the screen:** the simulation reproduces where the hand saturates but not how long a real jam persists (audit E5). Only the hand can answer this.

## Design

- **Plan `c5seeds`:** blocks C5s3, C5, C5, C5s3 (ABBA), repeated 5 times = **10 attempts each**. This is the hardware confirmation size in SPEC §7.
- **Primary metric:** safety-stop rate (C5 on 20 Nov: 6/11).
- **Secondary metrics:** completed 60 s runs, drops, rotations per trial.
- **Rotation count:** quarter turns, as on 20 Nov; full revolutions = count / 4.
- **Unchanged from 20 Nov:** same placement (`docs/figures/cube_placement.png`), kit settings, guards, 60 s trials and USB latency timer.
- **Record EVERY attempt,** including safety stops, answering the questions after each one. Discard only for a documented reason, such as the cube being misplaced before the policy started. Write the reason in the notes and keep the record.
- **Optional, after the 20:** a few `C5s2` runs, recorded in the same file.

## Commands

```bash
cd ~/Documents/AnyDesk/Dexterous-In-Hand-Manipulation/LEAP_Hand_RL_Deploy_Kit
sudo chmod 666 /dev/ttyUSB0
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
./leap.sh check
export LEAP_TRIAL_PLAN=c5seeds LEAP_TRIALS_CSV=$PWD/logs/trials_20261124.csv   # in every new terminal
./leap.sh shadow policies/C5s3          # once (torque off) to unlock it; C5 is already unlocked
./leap.sh trial-plan                    # the 20-attempt order
./leap.sh trial next                    # repeat 20 times
./leap.sh compare                       # Q5 (C5 vs C5s3) and Q6 (C5 vs C5s2) -> logs/trial_comparison.md
./leap.sh trial C5s2                    # optional extra attempts of seed 2 (shadow it first)
```
