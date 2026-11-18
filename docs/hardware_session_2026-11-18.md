# LEAP Hand hardware session, 18 Nov 2026: HwLim A vs C with live cube/force monitor

**Status:** first hardware runs of the policies retrained inside the kit's joint limits (`experiments/hwlimit_ac`), now with the live monitor. It logs the estimated cube state and fingertip forces every step, for both policies. This is a pilot: 2 full 30 s trials per policy, one seed each. **Cube rotation was not measured independently.** The team compared the live viewer with the real cube by eye (Section 2.1): the cube estimate is rough, and the force estimates look plausible but are not verified.

## 1. What was run (team-observed)

All logs are in `LEAP_Hand_RL_Deploy_Kit/logs/`. Each log now records its own policy, checkpoint sha256 and monitor (`run_meta_json`), so nothing below is inferred.

| Log `physical_20261118_*` | Policy | Length | Ended | Max current | Max temp | Max tracking error |
|---|---|---|---|---|---|---|
| 110351 | hwlim_C_seed1 | 6.0 s | completed | 386 mA | 22 °C | 0.28 rad |
| 110745 | hwlim_C_seed1 | 22.5 s | **stopped by operator (Ctrl+C)** | 382 mA | 22 °C | 0.37 rad |
| 111019 | hwlim_C_seed1 | 30.3 s | completed | 400 mA | 22 °C | 0.41 rad |
| 111155 | hwlim_A_seed1 | 30.3 s | completed | 385 mA | 23 °C | 0.32 rad |
| 111713 | hwlim_A_seed1 | 30.2 s | completed | 393 mA | 23 °C | 0.59 rad |

- **Shadow passes** (torque off, 30 s each): 4 for C seed 1, 2 for A seed 1, all completed.
- **Simulated-bus runs** (`sim_*`): 5, used to check the viewer. They contain no hand or cube data.
- **Guards:** no guard tripped and no stall recovery occurred. 350 mA is the limit set on the motors and 450 mA the abort threshold; the logged peaks of 382–400 mA are brief excursions above the configured limit.

## 2. Monitor results: estimated forces vs motor current

Produced by `./leap.sh analyze`; per-run files are `logs/physical_20261118_*_analysis.{json,png}`.
- r(F, I) = correlation between the estimated fingertip force and the summed |current| of that finger's 4 motors.
- "I contact / no" = mean finger current when the force estimator's contact probability is above 0.5 vs below 0.2.

| Run | Policy | Est. turns | Thumb r(F, I) | Ring r(F, I) | Index r(F, I) | Ring I contact / no (mA) | Thumb I contact / no (mA) |
|---|---|---|---|---|---|---|---|
| 110351 | C1 | +0.01 | +0.34 | +0.48 | −0.02 | 327 / 186 | 247 / 183 |
| 110745 | C1 | −0.25 | +0.39 | +0.57 | −0.18 | 346 / 172 | 268 / 238 |
| 111019 | C1 | −0.02 | +0.26 | +0.40 | +0.15 | 299 / 187 | 222 / 190 |
| 111155 | A1 | −0.05 | +0.23 | −0.05 | +0.17 | 148 / 161 | 204 / 194 |
| 111713 | A1 | +0.04 | +0.44 | +0.51 | −0.02 | 369 / 171 | 263 / 189 |

**Reading:**
- **Thumb:** the estimated force rises and falls with the motor current in all 5 runs (r +0.23 to +0.44). Whenever the ring finger is estimated to be in contact often enough to measure, it does too (r +0.40 to +0.57), and in contact it draws about twice the current.
- **Index:** the estimate does not track current (r −0.18 to +0.17).
- **Middle:** it is rarely estimated in contact (3–17% of steps).
- **Interpretation:** the force estimator carries real contact information on the thumb and ring finger. The comparison cannot validate absolute newtons, because current also includes gravity, friction and tracking effort.
- **Current-implied force:** F(I), from J^T F = Kt·I, agrees less well (r ≤ 0.42). It is noisy, it is limited by the 350 mA current limit, and Kt is approximate.

**Estimated cube rotation is near zero in every run:** −0.25 to +0.04 turns, while simulation predicts about 1.1–1.2 turns in 30 s for both policies. Two explanations remain open:
1. The real cube did not turn much (a sim-to-real gap).
2. Estimator B cannot follow the real cube. It is already about 1.5× worse in rotation on these new gaits in simulation (`experiments/hwlimit_ac/README.md` §8.4), and the real hand adds more shift.

Only the team's observation or a video can separate the two.

**A vs C:** with 2 full trials each and only estimated rotation, no comparison is possible yet. Both policies ran safely and similarly by every logged measure.

### 2.1 Team visual check (live viewer vs real hand)

During the runs the team watched the live viewer next to the real hand.
- **Cube state:** the estimated cube agreed only roughly with the real cube, by eye.
- **Forces:** the estimated fingertip forces looked plausible: arrows appeared on the fingers that were visibly pressing.
- **Not verified:** there is no force sensor, camera or motion capture, so neither estimate is verified. The correlation with motor current above is the only quantitative support, and it is indirect.

## 3. Problem found: control-loop jitter (USB latency timer)

| | 17 Nov (latency timer set to 1 ms) | 18 Nov (adapter replugged at 10:54) |
|---|---|---|
| Mean compute per step | 4–7 ms | 18–20 ms |
| Steps with a period > 60 ms | 3–7% | 19–22% |
| Mean step period | 50.2–50.5 ms | 50.3–50.6 ms |
| Simulated bus, viewer running | | 5 ms, 0% > 60 ms |

- **Likely cause:** replugging the FTDI adapter resets its latency timer to the Linux default of 16 ms, and every motor read then waits about 16 ms. The timer could not be read afterwards (the adapter was unplugged), so this is not directly confirmed. The monitor and viewer are ruled out: on the simulated bus they add less than 1 ms.
- **Effect:** the policy was trained at a steady 50 ms. About one step in five arriving late adds delay the policy never saw.
- **Fix:** `echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer` after each replug, or `./install.sh` once, which installs a udev rule. `leap.sh` now warns before every run when the timer is not 1 ms.

## 4. Still missing

- **Quantitative ground truth.** The team's check is qualitative ("rough" cube estimate, "plausible" forces). Still needed:
  - video or camera tracking of the cube, for turns and pose error;
  - a fingertip force sensor or a calibrated load test, for the force estimator.
- **Repeat the comparison with the latency timer at 1 ms:** at least 5 × 30 s per policy, alternating A and C, the same cube placement, and video.
- **Possibly retrain estimator B** on the HwLim policies' rollouts (proposed in `experiments/hwlimit_ac/README.md` §8.4).
