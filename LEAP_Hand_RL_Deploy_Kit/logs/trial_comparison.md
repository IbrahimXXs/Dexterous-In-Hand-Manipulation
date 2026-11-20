# Real-hand comparison (24 trials, 2026-11-20 10:45)

Observed by the operator (turns by eye, drops, interventions); trials where someone helped the cube are excluded.
Small samples: treat p-values and CIs as indicative.

Failure = cube dropped OR the run was ended by a safety guard (e.g. a finger jammed). Speed: rotations counted per trial (counting ends at a drop); rate per minute only for trials held >= 20 s.

| Policy | trials | rotations per trial | rate while held (>= 20 s) /min | drops | safety stops | failures | mean time held [s] | est. turns/min (monitor) | sim ref: turns/30 s, failure |
|---|---|---|---|---|---|---|---|---|---|
| A3 | 5 | 2.0 ± 1.4 (n=4) | 1.5 (n=2) | 2/5 | 0/5 | 40% | 40.6 | +0.34 | 1.33, 24% |
| C5 | 5 | 13.9 ± 2.2 (n=5) | 14.6 (n=5) | 1/5 | 0/5 | 20% | 57.3 | -1.70 | 1.87, 20% |
| R1 | 5 | 2.6 ± 1.3 (n=5) | nan (n=0) | 5/5 | 0/5 | 100% | 9.2 | +0.30 | 1.16, 19% |
| P0 | 5 | 0.0 ± 0.0 (n=5) | nan (n=0) | 5/5 | 2/5 | 100% | 10.8 | +0.08 | 1.27, 12% |
| M400 | 4 | 2.0 ± 0.8 (n=4) | 4.8 (n=1) | 4/4 | 0/4 | 100% | 13.5 | +1.52 | nan, nan% |

| Question | Comparison | Measure | First | Second | Second − first [95% CI] | p (permutation) |
|---|---|---|---|---|---|---|
| Q1: which rotates faster? | A3 vs C5 | rotations per trial | 2.00 | 13.90 | +11.90 [+9.85, +14.10] | 0.009 |
| Q2: which keeps the cube more reliably? | R1 vs P0 | failure rate | 1.00 | 1.00 | +0.00 [+0.00, +0.00] | 1.000 |
| Q2: which keeps the cube more reliably? | R1 vs P0 | drop rate | 1.00 | 1.00 | +0.00 [+0.00, +0.00] | 1.000 |
| Q2: which keeps the cube more reliably? | R1 vs P0 | time held [s] | 9.20 | 10.80 | +1.60 [-6.00, +11.80] | 0.903 |
| Q3: did the retention reward improve real-world stability? | A3 vs R1 | failure rate | 0.40 | 1.00 | +0.60 [+0.20, +1.00] | 0.163 |
| Q3: did the retention reward improve real-world stability? | A3 vs R1 | drop rate | 0.40 | 1.00 | +0.60 [+0.20, +1.00] | 0.163 |
| Q3: did the retention reward improve real-world stability? | A3 vs R1 | time held [s] | 40.61 | 9.20 | -31.41 [-51.76, -9.66] | 0.069 |
| Q4: speed vs retention trade-off | C5 vs P0 | rotations per trial | 13.90 | 0.00 | -13.90 [-15.80, -12.20] | 0.007 |
| Q4: speed vs retention trade-off | C5 vs P0 | failure rate | 0.20 | 1.00 | +0.80 [+0.40, +1.00] | 0.047 |
| Q4: speed vs retention trade-off | C5 vs P0 | time held [s] | 57.26 | 10.80 | -46.46 [-55.30, -35.31] | 0.007 |
