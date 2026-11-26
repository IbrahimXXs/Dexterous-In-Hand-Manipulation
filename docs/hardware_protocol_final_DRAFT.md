# DRAFT / NOT APPROVED: final real-hand comparison protocol (M400 vs A3 vs C5s3)

> **Status: DRAFT. NOT APPROVED. Do not run until the supervisor has signed section 10.**
> Written 26 Nov 2026 by the project (not by the lab). Nothing here changes the kit's guards, limits, `configs/hand.env` or policies.
> All hardware steps are for people at the bench. Steps that need a person at the hand are marked **[BENCH]**.

## 1. Question and arms

**Question:** on the real hand, under the same placement and kit settings, does the project's best deployable policy rotate the cube faster than the course baseline and the project's own PPO baseline, and how long does it hold it?

| Arm | Kit folder | What it is | Status |
|---|---|---|---|
| **M400** | `policies/M400` | course baseline `model_400` | run on 20 Nov (n = 4) |
| **A3** | `policies/A3` | proprioceptive PPO baseline, real joint limits, seed 3 | run on 20 Nov (n = 4) |
| **C5s3** | `policies/C5s3` | C5 (grasp-49 reset + yaw clip ±0.5 rad/s), seed 3 | run on 24/25 Nov: 0 guard stops in 18 attempts (13 on 24 Nov, 5 on 25 Nov) |
| *C5-EFFORT* (optional) | not in `policies/` | 350 mA-effort-trained C5, 1 seed | **only after** supervisor approval to copy it into the kit **and** a passing torque-off shadow run |
| *C5-HW-V2* (optional) | not in `policies/` | C5-EFFORT + response fit + kit-start resets, 1 seed | same condition as C5-EFFORT, and only if its simulation report passes |

**Why C5s3 and not C5 seed 1:** seed 1 tripped a safety guard in 9 of 16 attempts (20 + 24 Nov), seed 3 in 0 of 13 on 24 Nov (Fisher p = 0.001; `docs/hardware_session_2026-11-24.md`). The 25 Nov session added 5 more C5s3 runs, also without a guard stop (`LEAP_Hand_RL_Deploy_Kit/logs/trials_20261125.csv`).

**Prior real-hand numbers (for planning only, not part of this test):**
- 20 Nov operator counts, full revolutions per 60-s trial: M400 0.50 (n = 4), A3 0.50 (n = 4), C5 seed 1 3.48 (n = 5) (`docs/hardware_session_2026-11-20.md`).
- 24 Nov, C5s3: 17.6 quarter turns/min while held ≥ 20 s (n = 5). 6 of 12 valid trials dropped at 5–7 s, attributed to the start placement (`docs/hardware_session_2026-11-24.md`).

## 2. Design

- **Attempts:** **12 per arm** (36 in total), at least the 10 required.
- **Order:** a balanced (Williams) Latin square for three arms. Six sequences, each used twice:
  `M400 A3 C5s3 | A3 C5s3 M400 | C5s3 M400 A3 | M400 C5s3 A3 | A3 M400 C5s3 | C5s3 A3 M400`.
  Every arm appears equally often in each position and after each other arm. Time-of-session effects (temperature, cube wear, operator fatigue) are therefore balanced.
- **Run length:** 60 s (`TRIAL_STEPS = 1200`, the default). The placement countdown (5 s) and closing ramp (5 s) are unchanged.
- **Camera on every attempt:** `LEAP_USE_VISION=1`. An attempt without a camera recording is still recorded, but is excluded from the primary metric (section 6). If the camera fails, stop the session until it works (section 8).
- **Optional arms** are tested only in a **separate block** after the 36 attempts, as C5s3 vs the new arm in ABBA order, ×5 (10 each). They never interleave with the main square.
- **Expected duration:** about 3 min per attempt (countdown, ramp, 60 s, the operator's questions, reset). That is about 2 h for 36 attempts, plus setup and calibration.

**Software to run this plan (prepared, not merged):** branch `task3-hw-protocol` adds the plan `final3` to `LEAP_Hand_RL_Deploy_Kit/kit/trials.py`: the six sequences above, and the questions Q7 (C5s3 vs M400) and Q8 (C5s3 vs A3). It is scheduling code only; no guard, limit or `hand.env` value changes. It needs `TRIAL_BLOCKS=12` (12 sequences × 3 = 36), which `leap.sh` reads from the environment (`TRIAL_BLOCKS=${TRIAL_BLOCKS:-5}` in `hand.env`).

## 3. Before the session

1. **[BENCH] Station check, unchanged from 24/25 Nov:**

   ```bash
   cd ~/Documents/AnyDesk/Dexterous-In-Hand-Manipulation/LEAP_Hand_RL_Deploy_Kit
   sudo chmod 666 /dev/ttyUSB0
   echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
   ./leap.sh check
   ```

2. **Kit integrity, no hand needed:**
   - `sha256sum configs/hand.env` must print `68cc0663e9cb200f…` (the value at commit `8bdcbee`).
   - The checkpoint sha256 values of the three arms must match the earlier trial CSVs. C5s3 = `c221af89199cdeab…` (`logs/trials_20261125.csv`).
3. **[BENCH] Camera:**
   - Only one cube in view. Remove the second MBZUAI cube from the table: on 25 Nov the tracker followed it.
   - Run `./leap.sh vision` once and confirm about 30 fps in the printout. The 25 Nov fix (10 Hz panel, background recorder) has not yet been confirmed on the camera. More frames mean shorter hidden gaps.
   - If the camera has moved since 25 Nov, re-measure `--roi` before the first trial (`docs/hardware_session_2026-11-25.md`, "Files").
4. **[BENCH] Placement template** (section 4) mounted and checked against `docs/figures/cube_placement.png`.
5. **Shadow runs:** M400, A3 and C5s3 are already unlocked. An optional arm needs `./leap.sh shadow <folder>` (torque off) first, and only after its approval (section 10).
6. Use a **new trial file:**

   ```bash
   export LEAP_TRIAL_PLAN=final3 TRIAL_BLOCKS=12 LEAP_USE_VISION=1 \
          LEAP_TRIALS_CSV=$PWD/logs/trials_<YYYYMMDD>.csv
   ```

## 4. Placement: template and photo

**Problem being addressed:** C5s3's early drops at 5–7 s in 6 of 12 trials came from where the cube sat at the start. Training starts from grasp 49 tolerate about ±2 mm and ±10°, but hand placement is not that repeatable.

- **Template (to be made; design decision for the supervisor):** a thin removable guide that registers on the palm (on the palm edge or the finger bases) and has an L-shaped stop. The stop puts the cube centre over the knuckle line, in line with the middle finger, edges parallel to the knuckle line, logo / taped face up: exactly as `docs/figures/cube_placement.png`.
  - It must be **out of the fingers' path** before the closing ramp starts, or flat enough that the fingers cannot touch it.
  - Until the template exists, place by eye as on 24/25 Nov, and say so in the notes of every attempt.
- **Photo before each run:**
  - The trial recording starts before the countdown, so the frame at the **end of the closing ramp** (policy start) is the placement photo. Extract it after the session with `scripts/retime_recording.py`: it prints the policy start inside the re-timed video.
  - **[BENCH]** Additionally, take a phone photo from the operator's side after placing and before the ramp. Name it with the trial number.
- **Repeat rule (as before):** if the cube is pushed more than about 5 mm or tilts while the fingers close, the operator notes it. The attempt is still recorded (section 5) and the next attempt is a repeat of the same arm.

## 5. During each attempt

```bash
./leap.sh trial next     # 36 times; the plan picks the arm
./leap.sh trial-plan     # progress
```

- **[BENCH]** Place the cube during the countdown and hold it lightly until the fingers have closed, then let go. Do not touch the hand or the cube during the 60 s.
- **[BENCH] After each run, answer the kit's questions:**
  - quarter turns counted;
  - dropped (yes/no) and drop time;
  - intervened;
  - notes, including placement quality.
- **Record every attempt, including safety stops.** Exclude an attempt only for a documented reason that is independent of the policy, for example the cube was misplaced before the policy started or the camera was not recording. The row stays in the CSV with the reason in its notes.
- **Never edit a trial CSV by hand.** Corrections go in a separate note, as on 24 Nov (trial 15).

## 6. Metrics

**Primary (per attempt): camera revolutions per minute while held.**
- **Numerator:** camera quarter turns from policy start to the drop, or to the end of the run if no drop (`vision_qt_until_drop`, or `vision_quarter_turns` for no drop, from `scripts/vision_trials.py`), ÷ 4.
- **Denominator:** held time in minutes, defined as the operator's drop time, or the run time if no drop.
- **Inclusion:**
  - Defined for attempts held ≥ 10 s. Shorter holds are reported in a separate column; their rate is too noisy.
  - Attempts whose camera range (`vision_qt_lo`–`vision_qt_hi`) spans more than ±4 quarter turns are reported, and the primary analysis is repeated without them (sensitivity).
- **Sign:** fixed with `--axis-sign-from-trial` naming one attempt the operator saw turning the trained way. **Rotation in the wrong direction counts as negative.**

**Secondary:**
1. **Time to drop.** Operator drop time; 60 s if held. A safety stop ends the observation and is censored at its time in the drop analysis (it is also counted in item 2).
2. **Safety-stop rate:** stops by any kit guard ÷ attempts.
3. **Operator quarter turns per attempt.** The 20/24 Nov metric: comparable with earlier sessions.
4. **60-s completion:** held and no stop.

## 7. Analysis plan

Pre-specified contrasts, in order:
- **Q7:** C5s3 vs M400. Does the project beat the course baseline?
- **Q8:** C5s3 vs A3. Does C5 beat the project's PPO baseline?

Holm correction over the two contrasts for the primary metric.

| Metric | Test | Reported |
|---|---|---|
| Rev/min while held (primary) | Difference of means. 95% bootstrap CI: 10,000 resamples of attempts within each arm, percentile interval. Two-sided permutation p (20,000 permutations, as in `kit/trials.py`). | Per-arm mean, median and n; the difference with its CI and p |
| Time to drop | Kaplan–Meier survival curve per arm, safety stops censored. Log-rank test for Q7/Q8. | Figure; median time to drop; share held at 10, 30 and 60 s |
| Safety stops, 60-s completion | Fisher exact, two-sided | k/n per arm, p |
| Operator quarter turns | Bootstrap CI of the difference (as above) | Per arm |
| Camera vs operator | Per-attempt difference and Spearman correlation | Agreement table (validates the camera count) |

**Sensitivity of 12 vs 12 (Fisher, two-sided, α = 0.05):** the smallest differences that can reach significance are 0/12 vs 5/12, 1/12 vs 7/12, and 2/12 vs 8/12. Smaller differences in stop or completion rates will not be resolvable in this session. That is a limitation to state, not a reason to add attempts after looking at the data.

**No optional stopping:** the 36 attempts are run as scheduled, whatever the interim results.

## 8. Stopping rules

**The kit's guards stay exactly as configured:** current 450 mA, temperature 60 °C, velocity 8 rad/s, tracking 0.65 rad, command step 0.05 rad, motor current limit 350 mA. A guard stop ends that attempt; it is recorded as a safety stop.

**Pause the session (and tell the supervisor) if:**
1. two consecutive attempts of the same arm end in a safety stop;
2. any motor reads ≥ 55 °C at the end of an attempt. Let it cool to below 45 °C; the 60 °C guard is a last resort;
3. a joint reads outside its limits, a cable or screw is loose, or there is an unusual noise or smell;
4. the camera stops recording or shows < 15 fps;
5. the cube is damaged or its markers are peeling.

**Stop the session for good if:**
- the same hardware fault happens twice;
- the supervisor says so.

## 9. Camera-to-palm calibration, hand unpowered

**Why:** without it, rotation is measured about the motion's own axis (sign set by one trial), and the camera's cube pose cannot be compared with the kit's palm-frame estimate. They disagreed in sign in the 25 Nov dashboard plot. Calibration is **optional for the primary metric**, but required for any palm-frame claim.

**Requirement:** two or more (better three) cube placements whose palm-frame pose is physically known. Guessed poses are not acceptable (`docs/live_vision_digital_twin.md` §6.2). This needs a jig, for example a 3D-printed part that bolts to the palm and has CAD-known pockets.

**Procedure: the hand stays unpowered throughout.** No kit command that enables torque is run. The hand's USB/power can stay disconnected.

1. **[BENCH]** Fix the camera where it will be for the trials. Do not move it afterwards. If it moves, the calibration is void.
2. Copy the reference template:

   ```bash
   cp training_source/project/real_world/camera_configs/camera_to_palm_reference.example.json my_reference.json
   ```

   Enter the jig poses derived from the CAD in the palm frame of `kit/hand_scene.py` (origin = mean of the 10 `palm_collision_*` geoms; axes of the `palm` body). For each pose, write how it was derived. Set `is_example: false`.
3. **[BENCH]** Place the jig on the palm; the hand is unpowered. Make sure it sits flat and fully seated.
4. **[BENCH]** Start the capture. For each placement: put the cube in the jig pocket, step back so that no hand is in view, and press SPACE. 60 frames with two or more faces visible are averaged.

   ```bash
   uv run --extra vision --extra realsense python scripts/cube_vision.py camera-to-palm capture --camera realsense \
       --reference my_reference.json --captures c2p_captures.json \
       --layout training_source/project/assets/mbzuai_cube/measured/layout.measured_rgb.json \
       --print-layout training_source/project/assets/mbzuai_cube/measured/marker_print_layout.measured_perface_20261125.json
   ```

5. Solve; no hand needed:

   ```bash
   uv run --extra vision python scripts/cube_vision.py camera-to-palm solve \
       --reference my_reference.json --captures c2p_captures.json --out training_source/project/real_world/camera_to_palm_realsense.json
   ```

   The file is marked `calibrated: true` only if the maximum centre error is ≤ 3 mm and the rotation error ≤ 2°, over two or more placements.
6. **[BENCH] Check:** put the cube in one placement that was **not** used in the solve. The dashboard's palm-frame pose must match the jig pose within the same tolerance. Record the residual.
7. Commit the calibration file and the reference JSON, never the recordings. Note the camera position (a photo) in the session report.

## 10. Approval (to be filled in by the supervisor)

| Item | Decision | Signature / date |
|---|---|---|
| Protocol sections 1–8 as written (36 attempts: M400, A3, C5s3) | | |
| Placement template design (section 4) | | |
| Merge the `final3` trial plan into the kit (`kit/trials.py` only) | | |
| Optional: copy C5-EFFORT into `policies/` + shadow run | | |
| Optional: copy C5-HW-V2 into `policies/` + shadow run (only if its simulation report passes) | | |
| Camera-to-palm jig and calibration (section 9) | | |
