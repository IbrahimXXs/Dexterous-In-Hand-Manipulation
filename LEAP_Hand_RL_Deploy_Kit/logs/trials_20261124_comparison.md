# Real-hand comparison (10 trials, 2026-11-24 09:53)

Observed by the operator (turns by eye, drops, interventions); trials where someone helped the cube are excluded.
Small samples: treat p-values and CIs as indicative.

Failure = cube dropped OR the run was ended by a safety guard (e.g. a finger jammed). Speed: rotations counted per trial (counting ends at a drop); rate per minute only for trials held >= 20 s.

| Policy | trials | rotations per trial | rate while held (>= 20 s) /min | drops | safety stops | failures | mean time held [s] | est. turns/min (monitor) | sim ref: turns/30 s, failure |
|---|---|---|---|---|---|---|---|---|---|
| C5 | 5 | 5.2 ± 5.3 (n=4) | 13.0 (n=2) | 4/5 | 3/5 | 80% | 21.8 | -0.57 | 1.87, 20% |
| C5s3 | 4 | 7.0 ± 8.7 (n=4) | 19.9 (n=1) | 3/4 | 0/4 | 75% | 22.4 | -0.49 | 1.69, 20% |

| Question | Comparison | Measure | First | Second | Second − first [95% CI] | p (permutation) |
|---|---|---|---|---|---|---|
| Q5: does C5 seed 3 trip the guards less, at what speed? | C5 vs C5s3 | safety-stop rate | 0.60 | 0.00 | -0.60 [-1.00, -0.20] | 0.167 |
| Q5: does C5 seed 3 trip the guards less, at what speed? | C5 vs C5s3 | failure rate | 0.80 | 0.75 | -0.05 [-0.55, +0.40] | 1.000 |
| Q5: does C5 seed 3 trip the guards less, at what speed? | C5 vs C5s3 | rotations per trial | 5.25 | 7.00 | +1.75 [-6.00, +11.00] | 0.741 |
| Q5: does C5 seed 3 trip the guards less, at what speed? | C5 vs C5s3 | time held [s] | 21.83 | 22.35 | +0.52 [-27.55, +30.25] | 0.992 |
| Q6: does C5 seed 2 trip the guards less, at what speed? | C5 vs C5s2 | safety-stop rate | 0.60 | nan | +nan [+nan, +nan] | nan |
| Q6: does C5 seed 2 trip the guards less, at what speed? | C5 vs C5s2 | failure rate | 0.80 | nan | +nan [+nan, +nan] | nan |
| Q6: does C5 seed 2 trip the guards less, at what speed? | C5 vs C5s2 | rotations per trial | 5.25 | nan | +nan [+nan, +nan] | nan |
| Q6: does C5 seed 2 trip the guards less, at what speed? | C5 vs C5s2 | time held [s] | 21.83 | nan | +nan [+nan, +nan] | nan |
