# Retention-V1: pre-registered protocol (seed-1 pilot)

Written on 2026-11-19, before any Retention-V1 policy was trained. Its sha256 is recorded in `logs/protocol.sha256` at launch. Deviations are listed in the README with their reasons.

**Question.** Does a retention-aware reward help a proprioception-only PPO policy, trained from scratch under the approved joint limits, rotate the cube for longer without losing it?

## 1. Policy and training

- **Task:** `Mjlab-Leap-RetV1-A`.
- **What differs from hardware-limited A:** only two reward terms, `retention_penalty` (−0.05) and `grasp_loss` (−10). Their design is documented in `README.md` Section 3.
- **Everything else is identical to hardware-limited A:**
  - the 320-D actor observation;
  - the action and the approved limits;
  - the reset cache and 20-s episodes;
  - physics and domain randomisation;
  - the curriculum;
  - PPO: 4096 environments, 3001 updates, checkpoints every 50 updates, zero-initialised actor output.
- **Initialisation:** from scratch. No weights or optimiser state come from any existing policy.
- **Seed:** 1, the same seed as hardware-limited A seed 1. Its environment draws therefore start identically.
- **Run handling:** one run only. It is never resumed; if interrupted, it is retrained from scratch.
- **Stage 3A check at about 500 updates.** I stop and fix the run only if there is an implementation error or a reward exploit:
  - rotation collapses: validation yaw rate < 50% of A1's at the same update;
  - the retention penalty dominates the return;
  - training becomes unstable (NaNs, an exploding loss).

  Otherwise the same run continues to 3000 updates.

## 2. Validation and checkpoint selection

- **Validation:** every 100 updates, on simulation seed 7001, the nominal preset, 1024 environments, 30 s, deterministic actions. This is exactly the hardware-limited A validation protocol, so the A curves (`../hwlimit_ac/results/curves/`) are directly comparable.
- **Long-horizon validation:** also at updates 500, 1000, …, 3000, for 60 s, recording uninterrupted turns and survival.
- **Selected checkpoint:** the highest 30-s validation mean uninterrupted turns among updates 100–3000, with ties going to the later checkpoint. This is the same rule as A.
- **Final checkpoint:** `model_3000.pt` is kept and reported separately.
- **Convergence:** assessed with the hardware-limited A criteria C1–C4 (`../hwlimit_ac/PROTOCOL.md` Section 3).

## 3. Test (never used for selection)

**Baseline:** the three selected hardware-limited A policies.

**Held-out 30-s test:** the same as hardware-limited A:
- simulation seeds 5001 and 5002;
- presets nominal, heavy_slippery, light_grippy, high_delay, weak_motors and obs_noise_x3;
- 1024 environments × 30 s.

The A results are taken from `../hwlimit_ac/results/test/`. They used the same code path, seeds and environments; the only difference is the reward terms, which do not change evaluation.

**60-s retention test:** for Retention-V1 and the three A policies, re-run here with identical settings:
- seed 5001;
- presets nominal, high_delay and heavy_slippery;
- 1024 environments × 60 s;
- uninterrupted first episode only, with the timeout beyond the horizon.

**Metrics:**
- **Primary:** uninterrupted turns in 30 s; pose-failure rate; mean uninterrupted survival.
- **Long-horizon:**
  - turns in 60 s;
  - P(retain 30 s) and P(retain 60 s);
  - time to failure;
  - palm-frame position and drift;
  - time to leave the useful region (Mahalanobis ≥ 3, as in the diagnosis);
  - revolutions before failure.
- **Additional:**
  - distal-contact continuity;
  - joint-limit saturation;
  - yaw rate while alive and rotating fraction;
  - per-grasp survival;
  - the high_delay and heavy_slippery presets.

## 4. Pre-registered success targets (exploratory, 1 seed)

Reference values are hardware-limited A, selected checkpoints, nominal, test seeds 5001 and 5002:
- turns 1.21 (seeds 1.16 / 1.14 / 1.33);
- pose failures 24.1%;
- survival 24.7 s of 30.

Retention-V1 seed 1, selected checkpoint, is called **promising** only if all of these hold:
1. turns in 30 s ≥ 1.39 (+15% over the A mean);
2. pose-failure rate < 24.1% × 0.8 = 19.3%;
3. mean survival > 24.7 s;
4. 60-s turns and P(retain 60 s) above the A mean;
5. yaw rate while alive ≥ 90% of A's (no collapse of rotation).

**Outcomes:**
- **Outcome A (promising):** all five hold. Recommend seeds 2 and 3.
- **Outcome B (retention up, rotation down):** 2–3 hold and 1 or 5 fails.
- **Outcome C (no improvement):** otherwise.

A single training seed cannot establish an improvement. A's own seed spread (1.14–1.33 turns) is reported alongside, together with Retention-V1's position within that range.
