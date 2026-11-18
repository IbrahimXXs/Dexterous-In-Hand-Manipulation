# Pre-registered protocol: A vs C under the approved joint limits

Written 2026-11-17, before any training run of this study started. The sha256 of this file is recorded in `logs/protocol.sha256` at launch. Any later deviation is listed in the README under "Deviations", with its reason.

**Question.** Under the deployment kit's approved joint limits and a 3000-update PPO budget, does adding estimator B's cube-state and fingertip-contact features to the 320-D proprioceptive input (C) improve continuous in-hand cube rotation compared with proprioception only (A)?

## 1. Runs

- **Conditions:** A and C only, each with training seeds 1, 2 and 3, for 6 runs. Seeds are passed as `--agent.seed`, which is also the environment seed.
- **Tasks:** `Mjlab-Leap-HwLim-A` and `Mjlab-Leap-HwLim-C`. Both use the same environment and the same PPO configuration.
- **Budget:** 3001 PPO updates per run (iterations 0–3000).
  - Checkpoints are saved every 50 updates; all are kept.
  - Every run is trained from scratch and is never resumed. An interrupted run is renamed `*_INTERRUPTED` and retrained from scratch.
- **Estimator B:** frozen, sha256 `6e333d60…`, and never retrained in this study (the Phase-2 decision rule is in the README, Section 3).

## 2. Validation curves: every 100 updates

For each run, checkpoints 100, 200, …, 3000 are evaluated with:
- validation simulation seed 7001 (never used for testing);
- the nominal preset;
- 1024 environments × 30 s;
- the deterministic policy mean;
- `hwlimit_study.evaluation.evaluate`.

Recorded per checkpoint:
- mean uninterrupted turns `V` (primary), with its 95% CI over environments;
- mean yaw rate;
- survival;
- failure, drop and pose-failure rates;
- joint-limit saturation.

The PPO training reward is taken from the TensorBoard log (`Train/mean_reward`).

## 3. Convergence assessment

A fixed number of updates is never treated as evidence of convergence.

- **Smoothed curve:** a 3-point centred moving average of `V`; the end points use 2 points.
- **Final window:** the last 1000 updates, checkpoints 2000–3000 (11 points).
- **Comparison windows:**
  - W1 = checkpoints 2100–2500;
  - W2 = checkpoints 2600–3000 (5 points each).

A run is **approximately converged** only if all four criteria hold:

| # | Criterion | Threshold |
|---|---|---|
| C1 | No remaining trend: projected gain 1000·b, where b is the OLS slope of `V` over the final window | ≤ 0.05 · mean `V` of the final window. Also reported: the slope's 95% CI. |
| C2 | Sustained plateau: mean `V`(W2) − mean `V`(W1) | ≤ max(0.03 turns, 0.03 · mean `V`(W1)) |
| C3 | Training reward: relative change of mean `Train/mean_reward` over updates 2751–3000 vs 2501–2750 | ≤ 2% |
| C4 | Survival is stable: validation failure rate, W2 vs W1 | absolute change ≤ 0.03 |

**Labels:**
- If C1–C4 all hold: **approximately converged**.
- Otherwise, if any failed criterion shows improvement (a positive slope, a W2 > W1 gain, a reward increase or a falling failure rate): **not converged (still improving)**.
- Otherwise: **not converged (declining or unstable)**.

**Condition level:**
- A condition is called approximately converged only if all three of its seeds are.
- The validation curves and the window-to-window variation (s.d. of `V` across the 11 final-window evaluations) are reported for every run.

## 4. Checkpoint selection

- **Selected checkpoint:** the highest validation `V` (seed 7001, nominal) among checkpoints 100–3000; ties go to the later checkpoint. This is the same rule as the completed study.
- The final checkpoint (`model_3000.pt`) is kept and tested separately.
- Test seeds are never used for selection.

## 5. Held-out test

- **Policies tested:** the selected and the final checkpoint of all 6 runs.
- **Simulation seeds:** 5001 and 5002.
- **Presets:**
  - nominal;
  - heavy_slippery;
  - light_grippy;
  - high_delay (70–100 ms, which covers the ~100 ms lag measured on the hand);
  - weak_motors;
  - obs_noise_x3.
- **Rollouts:** 1024 environments × 30 s, deterministic, all under the approved limits.

**Primary metric:** mean uninterrupted turns in 30 s (nominal, pooled over both test seeds), selected checkpoints.

**Secondary metrics:**
- turns with resets, yaw rate, rotation direction;
- survival, completion;
- drops, pose failures, "too fast" terminations;
- drift and tilt;
- joint-limit saturation (per joint and any-joint);
- kit clipping (expected to be 0);
- maximum per-step command change (the kit aborts above 0.05 rad);
- action change and executed-action saturation;
- learning speed (validation `V` averaged over updates 100–3000, and the first update reaching 0.5 turns);
- convergence status.

**Statistics:**
- Per-seed values are reported, along with condition means.
- C − A differences get:
  - 95% CIs from a two-level bootstrap (training seeds, then environments);
  - a t-interval across the 3 vs 3 seed means;
  - the exact one-sided permutation p-value over the 20 seed splits. The minimum attainable p is 0.05.
- With n = 3 per condition, no difference is called statistically significant unless both CIs exclude 0. Effect sizes are reported with their uncertainty, not as a pass/fail result.

## 6. Estimator audit after training

Estimator B is scored closed-loop on the trained HwLim-C and HwLim-A policies' test rollouts (nominal, seed 5001). This is reported only; B is not changed after the fact.
