# Real-hand comparison (18 trials, 2026-11-24 10:12)

Observed by the operator (turns by eye, drops, interventions); trials where someone helped the cube are excluded.
Small samples: treat p-values and CIs as indicative.

Failure = cube dropped OR the run was ended by a safety guard (e.g. a finger jammed). Speed: rotations counted per trial (counting ends at a drop); rate per minute only for trials held >= 20 s.

| Policy | trials | rotations per trial | rate while held (>= 20 s) /min | drops | safety stops | failures | mean time held [s] | est. turns/min (monitor) | sim ref: turns/30 s, failure |
|---|---|---|---|---|---|---|---|---|---|
| C5 | 5 | 5.2 ± 5.3 (n=4) | 13.0 (n=2) | 4/5 | 3/5 | 80% | 21.8 | -0.57 | 1.87, 20% |
| C5s3 | 12 | 6.8 ± 6.2 (n=12) | 17.6 (n=5) | 11/12 | 0/12 | 92% | 23.0 | +0.02 | 1.69, 20% |

| Question | Comparison | Measure | First | Second | Second − first [95% CI] | p (permutation) |
|---|---|---|---|---|---|---|
| Q5: does C5 seed 3 trip the guards less, at what speed? | C5 vs C5s3 | safety-stop rate | 0.60 | 0.00 | -0.60 [-1.00, -0.20] | 0.014 |
| Q5: does C5 seed 3 trip the guards less, at what speed? | C5 vs C5s3 | failure rate | 0.80 | 0.92 | +0.12 [-0.17, +0.60] | 1.000 |
| Q5: does C5 seed 3 trip the guards less, at what speed? | C5 vs C5s3 | rotations per trial | 5.25 | 6.83 | +1.58 [-4.08, +7.08] | 0.663 |
| Q5: does C5 seed 3 trip the guards less, at what speed? | C5 vs C5s3 | time held [s] | 21.83 | 23.03 | +1.21 [-22.01, +21.88] | 0.931 |
| Q6: does C5 seed 2 trip the guards less, at what speed? | C5 vs C5s2 | safety-stop rate | 0.60 | nan | +nan [+nan, +nan] | nan |
| Q6: does C5 seed 2 trip the guards less, at what speed? | C5 vs C5s2 | failure rate | 0.80 | nan | +nan [+nan, +nan] | nan |
| Q6: does C5 seed 2 trip the guards less, at what speed? | C5 vs C5s2 | rotations per trial | 5.25 | nan | +nan [+nan, +nan] | nan |
| Q6: does C5 seed 2 trip the guards less, at what speed? | C5 vs C5s2 | time held [s] | 21.83 | nan | +nan [+nan, +nan] | nan |
