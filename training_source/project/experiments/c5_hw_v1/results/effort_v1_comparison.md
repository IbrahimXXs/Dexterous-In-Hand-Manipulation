# C5-HW-EFFORT-V1: held-out comparison

**SINGLE-SEED EXPLORATORY EXPERIMENT (PPO seed 1 only). CIs are paired bootstraps over evaluation environments: evaluation variation for this one model, not training-seed variation.**

Decision setting: A (full 239-grasp distribution), nominal, seeds 6201+6202, effort-matched actuator (350 mA-equivalent; response NOT fitted).

## Verdict: SINGLE-SEED PASS (exploratory; not confirmed)

| Pre-registered criterion (SPEC §7) | Value | Met |
|---|---|---|
| 1a jam proxy reduced ≥ 50% vs C5 | 96.5% (CI 95.0% to 97.9%) | yes |
| 1b 95% CI of the difference excludes 0 | -0.1409 [-0.1485, -0.1329] | yes |
| 2a turns30 ≥ 0.90 × C5 | ratio 0.911 | yes |
| 2b turns30 ≥ 1.20 × A mean | ratio 1.362 | yes |
| 3 failure30 ≤ C5 + 0.03 | 0.178 vs 0.180 | yes |

Rejection checks: turns30_lt_0.80_C5 = False, jam_reduction_lt_25pct = False, fail30_gt_C5_plus_0.08 = False

_Seed-level Welch tests (SPEC Section 7) are impossible with one training seed; criterion 1's CI is the paired environment bootstrap (evaluation variation only)._

Paired differences vs the C5 mean (environment bootstrap): turns30 -0.159 [-0.174, -0.144], failure30 -0.003 [-0.011, +0.006].

Where the ≥ 1 s index jams occur (full distribution, effort-matched; descriptive):

| policy | jam rate, held 60 s | jam rate, failed | P(fail \| jam) | P(fail \| no jam) | onset < 3 s | 3–10 s | ≥ 10 s |
|---|---|---|---|---|---|---|---|
| EFFORT_1 | 0.004 | 0.009 | 0.381 | 0.205 | 0.52 | 0.29 | 0.19 |
| C5_1 | 0.164 | 0.235 | 0.277 | 0.197 | 0.47 | 0.28 | 0.24 |
| C5_2 | 0.086 | 0.176 | 0.277 | 0.144 | 0.87 | 0.10 | 0.02 |
| C5_3 | 0.129 | 0.270 | 0.357 | 0.182 | 0.67 | 0.27 | 0.07 |
| A1 | 0.142 | 0.188 | 0.305 | 0.238 | 0.48 | 0.29 | 0.23 |
| A2 | 0.089 | 0.167 | 0.425 | 0.267 | 0.59 | 0.30 | 0.10 |
| A3 | 0.118 | 0.255 | 0.435 | 0.231 | 0.77 | 0.22 | 0.01 |
| R1 | 0.110 | 0.217 | 0.362 | 0.202 | 0.71 | 0.13 | 0.16 |

Against each C5 seed (descriptive):

| C5 seed | jam reduction | jam diff [CI] | turns30 ratio | failure30 diff |
|---|---|---|---|---|
| C5_1 | 97.1% | -0.1741 [-0.1858, -0.1624] | 0.878 | -0.021 |
| C5_2 | 94.9% | -0.0952 [-0.1045, -0.0859] | 0.894 | +0.029 |
| C5_3 | 96.8% | -0.1533 [-0.1643, -0.1421] | 0.966 | -0.015 |

## full 239-grasp distribution · effort actuator · nominal · seeds 6201+6202

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.193 | 1.148 | 1.784 | 1.625 |
| Turns / 60 s | 2.378 | 2.299 | 3.646 | 3.300 |
| Yaw rate held (rad/s, 30 s) | 0.290 | 0.257 | 0.406 | 0.368 |
| Failure rate (30 s) | 0.249 | 0.175 | 0.180 | 0.178 |
| Failure rate (60 s) | 0.265 | 0.224 | 0.193 | 0.206 |
| P(held 60 s) | 0.735 | 0.776 | 0.807 | 0.794 |
| Survival (s, 60-s rollout) | 46.780 | 51.220 | 50.484 | 50.524 |
| Turns / 60 s if held 60 s | 3.096 | 2.784 | 4.398 | 4.031 |
| Index MCP saturation | 0.0189 | 0.0076 | 0.0362 | 0.0025 |
| Index PIP saturation | 0.0017 | 0.0058 | 0.0025 | 0.0007 |
| Thumb abduction saturation | 0.0057 | 0.0077 | 0.0267 | 0.0037 |
| All joints at ≥ 98% of limit | 0.0105 | 0.0103 | 0.0129 | 0.0012 |
| Index blocked fraction | 0.0196 | 0.0124 | 0.0375 | 0.0027 |
| Index jam proxy (≥ 1 s, 60 s) | 0.1394 | 0.1343 | 0.1460 | 0.0051 |
| Index jam proxy (≥ 1 s, 30 s) | 0.1351 | 0.1279 | 0.1392 | 0.0049 |
| Effort utilisation mean | 0.238 | 0.257 | 0.254 | 0.231 |
| Utilisation p95 | 0.681 | 0.711 | 0.734 | 0.616 |
| Utilisation p99 | 0.986 | 0.985 | 0.997 | 0.784 |
| Tracking error mean (rad) | 0.096 | 0.098 | 0.109 | 0.104 |
| Joint speed mean (rad/s) | 0.255 | 0.245 | 0.316 | 0.326 |
| Command-limit occupancy | 0.260 | 0.321 | 0.260 | 0.225 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.142 | 1.131 | 1.306 | 1.850 | 1.819 | 1.682 | 1.625 |
| Turns / 60 s | 2.281 | 2.223 | 2.630 | 3.813 | 3.768 | 3.358 | 3.300 |
| Yaw rate held (rad/s, 30 s) | 0.277 | 0.278 | 0.315 | 0.425 | 0.405 | 0.387 | 0.368 |
| Failure rate (30 s) | 0.234 | 0.267 | 0.246 | 0.199 | 0.149 | 0.193 | 0.178 |
| Failure rate (60 s) | 0.248 | 0.284 | 0.262 | 0.211 | 0.158 | 0.210 | 0.206 |
| P(held 60 s) | 0.752 | 0.716 | 0.738 | 0.789 | 0.842 | 0.790 | 0.794 |
| Survival (s, 60-s rollout) | 47.592 | 46.021 | 46.727 | 49.597 | 52.054 | 49.801 | 50.524 |
| Turns / 60 s if held 60 s | 2.909 | 2.947 | 3.431 | 4.688 | 4.391 | 4.115 | 4.031 |
| Index MCP saturation | 0.0332 | 0.0123 | 0.0112 | 0.0626 | 0.0178 | 0.0280 | 0.0025 |
| Index PIP saturation | 0.0015 | 0.0014 | 0.0023 | 0.0039 | 0.0018 | 0.0019 | 0.0007 |
| Thumb abduction saturation | 0.0054 | 0.0041 | 0.0075 | 0.0305 | 0.0077 | 0.0418 | 0.0037 |
| All joints at ≥ 98% of limit | 0.0116 | 0.0089 | 0.0110 | 0.0108 | 0.0129 | 0.0150 | 0.0012 |
| Index blocked fraction | 0.0333 | 0.0128 | 0.0126 | 0.0648 | 0.0186 | 0.0289 | 0.0027 |
| Index jam proxy (≥ 1 s, 60 s) | 0.1531 | 0.1113 | 0.1538 | 0.1792 | 0.1003 | 0.1584 | 0.0051 |
| Index jam proxy (≥ 1 s, 30 s) | 0.1421 | 0.1096 | 0.1536 | 0.1648 | 0.1001 | 0.1526 | 0.0049 |
| Effort utilisation mean | 0.230 | 0.252 | 0.232 | 0.248 | 0.248 | 0.266 | 0.231 |
| Utilisation p95 | 0.650 | 0.704 | 0.689 | 0.705 | 0.735 | 0.762 | 0.616 |
| Utilisation p99 | 0.998 | 0.966 | 0.993 | 0.991 | 1.000 | 1.000 | 0.784 |
| Tracking error mean (rad) | 0.093 | 0.104 | 0.092 | 0.104 | 0.107 | 0.116 | 0.104 |
| Joint speed mean (rad/s) | 0.246 | 0.285 | 0.233 | 0.288 | 0.315 | 0.344 | 0.326 |
| Command-limit occupancy | 0.307 | 0.163 | 0.311 | 0.289 | 0.249 | 0.241 | 0.225 |

## full 239-grasp distribution · effort actuator · heavy_slippery · seed 6201

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.163 | 1.144 | 1.728 | 1.503 |
| Turns / 60 s | 2.293 | 2.292 | 3.519 | 2.990 |
| Yaw rate held (rad/s, 30 s) | 0.282 | 0.260 | 0.396 | 0.341 |
| Failure rate (30 s) | 0.232 | 0.167 | 0.179 | 0.170 |
| Failure rate (60 s) | 0.261 | 0.198 | 0.190 | 0.209 |
| P(held 60 s) | 0.739 | 0.802 | 0.810 | 0.791 |
| Survival (s, 60-s rollout) | 47.527 | 51.583 | 50.517 | 50.925 |
| Turns / 60 s if held 60 s | 2.947 | 2.730 | 4.235 | 3.637 |
| Index MCP saturation | 0.0148 | 0.0077 | 0.0248 | 0.0028 |
| Index PIP saturation | 0.0013 | 0.0032 | 0.0017 | 0.0006 |
| Thumb abduction saturation | 0.0072 | 0.0097 | 0.0212 | 0.0047 |
| All joints at ≥ 98% of limit | 0.0099 | 0.0084 | 0.0097 | 0.0014 |
| Index blocked fraction | 0.0151 | 0.0100 | 0.0255 | 0.0029 |
| Index jam proxy (≥ 1 s, 60 s) | 0.1125 | 0.1016 | 0.1051 | 0.0049 |
| Index jam proxy (≥ 1 s, 30 s) | 0.1110 | 0.1016 | 0.1050 | 0.0049 |
| Effort utilisation mean | 0.233 | 0.247 | 0.244 | 0.230 |
| Utilisation p95 | 0.676 | 0.695 | 0.698 | 0.615 |
| Utilisation p99 | 0.977 | 0.958 | 0.961 | 0.786 |
| Tracking error mean (rad) | 0.095 | 0.096 | 0.107 | 0.104 |
| Joint speed mean (rad/s) | 0.255 | 0.246 | 0.320 | 0.326 |
| Command-limit occupancy | 0.266 | 0.327 | 0.259 | 0.225 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.067 | 1.156 | 1.266 | 1.851 | 1.783 | 1.549 | 1.503 |
| Turns / 60 s | 2.082 | 2.267 | 2.530 | 3.869 | 3.725 | 2.963 | 2.990 |
| Yaw rate held (rad/s, 30 s) | 0.259 | 0.277 | 0.309 | 0.423 | 0.406 | 0.360 | 0.341 |
| Failure rate (30 s) | 0.230 | 0.225 | 0.240 | 0.188 | 0.165 | 0.184 | 0.170 |
| Failure rate (60 s) | 0.272 | 0.250 | 0.260 | 0.196 | 0.171 | 0.203 | 0.209 |
| P(held 60 s) | 0.728 | 0.750 | 0.740 | 0.804 | 0.829 | 0.797 | 0.791 |
| Survival (s, 60-s rollout) | 47.477 | 48.127 | 46.977 | 50.045 | 51.296 | 50.210 | 50.925 |
| Turns / 60 s if held 60 s | 2.673 | 2.880 | 3.289 | 4.710 | 4.405 | 3.589 | 3.637 |
| Index MCP saturation | 0.0199 | 0.0135 | 0.0110 | 0.0359 | 0.0185 | 0.0199 | 0.0028 |
| Index PIP saturation | 0.0011 | 0.0010 | 0.0016 | 0.0024 | 0.0014 | 0.0015 | 0.0006 |
| Thumb abduction saturation | 0.0086 | 0.0047 | 0.0082 | 0.0190 | 0.0073 | 0.0374 | 0.0047 |
| All joints at ≥ 98% of limit | 0.0119 | 0.0087 | 0.0092 | 0.0070 | 0.0083 | 0.0138 | 0.0014 |
| Index blocked fraction | 0.0201 | 0.0138 | 0.0116 | 0.0372 | 0.0189 | 0.0205 | 0.0029 |
| Index jam proxy (≥ 1 s, 60 s) | 0.1060 | 0.0884 | 0.1431 | 0.0981 | 0.1069 | 0.1104 | 0.0049 |
| Index jam proxy (≥ 1 s, 30 s) | 0.1040 | 0.0864 | 0.1426 | 0.0977 | 0.1069 | 0.1104 | 0.0049 |
| Effort utilisation mean | 0.224 | 0.246 | 0.229 | 0.236 | 0.238 | 0.259 | 0.230 |
| Utilisation p95 | 0.650 | 0.696 | 0.683 | 0.651 | 0.698 | 0.746 | 0.615 |
| Utilisation p99 | 1.000 | 0.963 | 0.969 | 0.928 | 0.956 | 1.000 | 0.786 |
| Tracking error mean (rad) | 0.092 | 0.102 | 0.092 | 0.101 | 0.105 | 0.115 | 0.104 |
| Joint speed mean (rad/s) | 0.244 | 0.283 | 0.239 | 0.293 | 0.318 | 0.348 | 0.326 |
| Command-limit occupancy | 0.306 | 0.179 | 0.313 | 0.287 | 0.249 | 0.242 | 0.225 |

## full 239-grasp distribution · effort actuator · high_delay · seed 6201

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.098 | 1.052 | 1.672 | 1.497 |
| Turns / 60 s | 2.205 | 2.144 | 3.459 | 3.088 |
| Yaw rate held (rad/s, 30 s) | 0.269 | 0.245 | 0.385 | 0.343 |
| Failure rate (30 s) | 0.274 | 0.201 | 0.202 | 0.202 |
| Failure rate (60 s) | 0.287 | 0.241 | 0.212 | 0.228 |
| P(held 60 s) | 0.713 | 0.759 | 0.788 | 0.772 |
| Survival (s, 60-s rollout) | 45.269 | 49.576 | 49.039 | 48.799 |
| Turns / 60 s if held 60 s | 2.974 | 2.678 | 4.294 | 3.896 |
| Index MCP saturation | 0.0196 | 0.0093 | 0.0366 | 0.0029 |
| Index PIP saturation | 0.0026 | 0.0073 | 0.0032 | 0.0014 |
| Thumb abduction saturation | 0.0062 | 0.0081 | 0.0242 | 0.0040 |
| All joints at ≥ 98% of limit | 0.0109 | 0.0106 | 0.0132 | 0.0018 |
| Index blocked fraction | 0.0202 | 0.0149 | 0.0373 | 0.0034 |
| Index jam proxy (≥ 1 s, 60 s) | 0.1522 | 0.1323 | 0.1444 | 0.0054 |
| Index jam proxy (≥ 1 s, 30 s) | 0.1475 | 0.1274 | 0.1372 | 0.0054 |
| Effort utilisation mean | 0.237 | 0.256 | 0.255 | 0.231 |
| Utilisation p95 | 0.682 | 0.710 | 0.735 | 0.616 |
| Utilisation p99 | 0.989 | 0.988 | 0.997 | 0.787 |
| Tracking error mean (rad) | 0.103 | 0.105 | 0.119 | 0.114 |
| Joint speed mean (rad/s) | 0.253 | 0.245 | 0.317 | 0.324 |
| Command-limit occupancy | 0.261 | 0.322 | 0.259 | 0.224 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.055 | 1.047 | 1.193 | 1.729 | 1.712 | 1.574 | 1.497 |
| Turns / 60 s | 2.132 | 2.068 | 2.416 | 3.612 | 3.573 | 3.191 | 3.088 |
| Yaw rate held (rad/s, 30 s) | 0.256 | 0.259 | 0.294 | 0.398 | 0.387 | 0.369 | 0.343 |
| Failure rate (30 s) | 0.252 | 0.290 | 0.280 | 0.217 | 0.177 | 0.211 | 0.202 |
| Failure rate (60 s) | 0.261 | 0.307 | 0.293 | 0.225 | 0.185 | 0.228 | 0.228 |
| P(held 60 s) | 0.739 | 0.693 | 0.707 | 0.775 | 0.815 | 0.772 | 0.772 |
| Survival (s, 60-s rollout) | 46.487 | 44.566 | 44.753 | 48.334 | 50.246 | 48.537 | 48.799 |
| Turns / 60 s if held 60 s | 2.788 | 2.837 | 3.297 | 4.555 | 4.316 | 4.011 | 3.896 |
| Index MCP saturation | 0.0343 | 0.0123 | 0.0122 | 0.0622 | 0.0203 | 0.0272 | 0.0029 |
| Index PIP saturation | 0.0021 | 0.0023 | 0.0033 | 0.0046 | 0.0024 | 0.0025 | 0.0014 |
| Thumb abduction saturation | 0.0057 | 0.0049 | 0.0080 | 0.0275 | 0.0083 | 0.0368 | 0.0040 |
| All joints at ≥ 98% of limit | 0.0122 | 0.0095 | 0.0110 | 0.0108 | 0.0135 | 0.0154 | 0.0018 |
| Index blocked fraction | 0.0337 | 0.0129 | 0.0139 | 0.0634 | 0.0203 | 0.0281 | 0.0034 |
| Index jam proxy (≥ 1 s, 60 s) | 0.1890 | 0.1216 | 0.1460 | 0.1675 | 0.1079 | 0.1577 | 0.0054 |
| Index jam proxy (≥ 1 s, 30 s) | 0.1763 | 0.1201 | 0.1460 | 0.1533 | 0.1060 | 0.1523 | 0.0054 |
| Effort utilisation mean | 0.230 | 0.252 | 0.230 | 0.247 | 0.251 | 0.268 | 0.231 |
| Utilisation p95 | 0.652 | 0.708 | 0.685 | 0.700 | 0.742 | 0.763 | 0.616 |
| Utilisation p99 | 1.000 | 0.975 | 0.994 | 0.992 | 1.000 | 1.000 | 0.787 |
| Tracking error mean (rad) | 0.100 | 0.112 | 0.098 | 0.112 | 0.118 | 0.127 | 0.114 |
| Joint speed mean (rad/s) | 0.245 | 0.283 | 0.232 | 0.289 | 0.318 | 0.344 | 0.324 |
| Command-limit occupancy | 0.304 | 0.166 | 0.312 | 0.290 | 0.246 | 0.242 | 0.224 |

## full 239-grasp distribution · effort actuator · obs_noise_x3 · seed 6201

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.194 | 1.152 | 1.774 | 1.636 |
| Turns / 60 s | 2.378 | 2.294 | 3.627 | 3.334 |
| Yaw rate held (rad/s, 30 s) | 0.288 | 0.256 | 0.403 | 0.368 |
| Failure rate (30 s) | 0.247 | 0.179 | 0.181 | 0.171 |
| Failure rate (60 s) | 0.263 | 0.229 | 0.195 | 0.196 |
| P(held 60 s) | 0.737 | 0.771 | 0.805 | 0.804 |
| Survival (s, 60-s rollout) | 46.875 | 51.159 | 50.346 | 51.026 |
| Turns / 60 s if held 60 s | 3.090 | 2.788 | 4.390 | 4.024 |
| Index MCP saturation | 0.0194 | 0.0076 | 0.0367 | 0.0027 |
| Index PIP saturation | 0.0018 | 0.0057 | 0.0025 | 0.0007 |
| Thumb abduction saturation | 0.0057 | 0.0078 | 0.0269 | 0.0037 |
| All joints at ≥ 98% of limit | 0.0106 | 0.0105 | 0.0131 | 0.0013 |
| Index blocked fraction | 0.0201 | 0.0125 | 0.0379 | 0.0030 |
| Index jam proxy (≥ 1 s, 60 s) | 0.1440 | 0.1250 | 0.1431 | 0.0073 |
| Index jam proxy (≥ 1 s, 30 s) | 0.1387 | 0.1216 | 0.1370 | 0.0068 |
| Effort utilisation mean | 0.238 | 0.257 | 0.255 | 0.231 |
| Utilisation p95 | 0.682 | 0.713 | 0.736 | 0.617 |
| Utilisation p99 | 0.987 | 0.988 | 0.997 | 0.785 |
| Tracking error mean (rad) | 0.096 | 0.098 | 0.109 | 0.104 |
| Joint speed mean (rad/s) | 0.255 | 0.246 | 0.316 | 0.326 |
| Command-limit occupancy | 0.261 | 0.321 | 0.259 | 0.224 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.150 | 1.126 | 1.304 | 1.832 | 1.811 | 1.679 | 1.636 |
| Turns / 60 s | 2.297 | 2.209 | 2.628 | 3.770 | 3.760 | 3.352 | 3.334 |
| Yaw rate held (rad/s, 30 s) | 0.274 | 0.277 | 0.314 | 0.421 | 0.403 | 0.385 | 0.368 |
| Failure rate (30 s) | 0.224 | 0.273 | 0.246 | 0.206 | 0.148 | 0.190 | 0.171 |
| Failure rate (60 s) | 0.238 | 0.291 | 0.261 | 0.218 | 0.159 | 0.209 | 0.196 |
| P(held 60 s) | 0.762 | 0.709 | 0.739 | 0.782 | 0.841 | 0.791 | 0.804 |
| Survival (s, 60-s rollout) | 48.098 | 45.774 | 46.753 | 49.204 | 51.966 | 49.867 | 51.026 |
| Turns / 60 s if held 60 s | 2.900 | 2.945 | 3.426 | 4.675 | 4.385 | 4.109 | 4.024 |
| Index MCP saturation | 0.0347 | 0.0123 | 0.0112 | 0.0624 | 0.0193 | 0.0283 | 0.0027 |
| Index PIP saturation | 0.0016 | 0.0016 | 0.0023 | 0.0039 | 0.0017 | 0.0019 | 0.0007 |
| Thumb abduction saturation | 0.0057 | 0.0040 | 0.0073 | 0.0307 | 0.0077 | 0.0421 | 0.0037 |
| All joints at ≥ 98% of limit | 0.0118 | 0.0088 | 0.0111 | 0.0109 | 0.0133 | 0.0151 | 0.0013 |
| Index blocked fraction | 0.0349 | 0.0129 | 0.0125 | 0.0646 | 0.0199 | 0.0293 | 0.0030 |
| Index jam proxy (≥ 1 s, 60 s) | 0.1660 | 0.1138 | 0.1523 | 0.1675 | 0.1094 | 0.1523 | 0.0073 |
| Index jam proxy (≥ 1 s, 30 s) | 0.1523 | 0.1118 | 0.1519 | 0.1533 | 0.1089 | 0.1489 | 0.0068 |
| Effort utilisation mean | 0.230 | 0.252 | 0.232 | 0.249 | 0.250 | 0.266 | 0.231 |
| Utilisation p95 | 0.651 | 0.704 | 0.692 | 0.706 | 0.739 | 0.762 | 0.617 |
| Utilisation p99 | 1.000 | 0.966 | 0.996 | 0.992 | 1.000 | 1.000 | 0.785 |
| Tracking error mean (rad) | 0.093 | 0.104 | 0.092 | 0.103 | 0.108 | 0.116 | 0.104 |
| Joint speed mean (rad/s) | 0.246 | 0.285 | 0.234 | 0.288 | 0.316 | 0.345 | 0.326 |
| Command-limit occupancy | 0.307 | 0.163 | 0.311 | 0.289 | 0.248 | 0.241 | 0.224 |

## full 239-grasp distribution · default actuator · nominal · seeds 6201+6202

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.201 | 1.160 | 1.794 | 1.617 |
| Turns / 60 s | 2.391 | 2.308 | 3.663 | 3.288 |
| Yaw rate held (rad/s, 30 s) | 0.295 | 0.263 | 0.412 | 0.369 |
| Failure rate (30 s) | 0.249 | 0.181 | 0.183 | 0.176 |
| Failure rate (60 s) | 0.265 | 0.234 | 0.194 | 0.204 |
| P(held 60 s) | 0.735 | 0.766 | 0.806 | 0.796 |
| Survival (s, 60-s rollout) | 46.722 | 50.732 | 50.368 | 50.464 |
| Turns / 60 s if held 60 s | 3.113 | 2.821 | 4.426 | 4.013 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.151 | 0.164 | 0.162 | 0.147 |
| Utilisation p95 | 0.432 | 0.451 | 0.464 | 0.392 |
| Utilisation p99 | 0.621 | 0.619 | 0.633 | 0.498 |
| Tracking error mean (rad) | 0.096 | 0.098 | 0.109 | 0.104 |
| Joint speed mean (rad/s) | 0.256 | 0.247 | 0.317 | 0.326 |
| Command-limit occupancy | 0.261 | 0.321 | 0.260 | 0.225 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.151 | 1.128 | 1.325 | 1.863 | 1.829 | 1.691 | 1.617 |
| Turns / 60 s | 2.291 | 2.213 | 2.670 | 3.834 | 3.775 | 3.380 | 3.288 |
| Yaw rate held (rad/s, 30 s) | 0.281 | 0.281 | 0.322 | 0.431 | 0.411 | 0.394 | 0.369 |
| Failure rate (30 s) | 0.239 | 0.270 | 0.238 | 0.203 | 0.149 | 0.196 | 0.176 |
| Failure rate (60 s) | 0.255 | 0.286 | 0.254 | 0.215 | 0.157 | 0.209 | 0.204 |
| P(held 60 s) | 0.745 | 0.714 | 0.746 | 0.785 | 0.843 | 0.791 | 0.796 |
| Survival (s, 60-s rollout) | 47.276 | 45.836 | 47.055 | 49.384 | 52.074 | 49.646 | 50.464 |
| Turns / 60 s if held 60 s | 2.941 | 2.944 | 3.454 | 4.731 | 4.395 | 4.153 | 4.013 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.147 | 0.160 | 0.147 | 0.157 | 0.158 | 0.170 | 0.147 |
| Utilisation p95 | 0.413 | 0.448 | 0.435 | 0.443 | 0.467 | 0.483 | 0.392 |
| Utilisation p99 | 0.629 | 0.611 | 0.623 | 0.618 | 0.636 | 0.645 | 0.498 |
| Tracking error mean (rad) | 0.093 | 0.104 | 0.091 | 0.103 | 0.107 | 0.116 | 0.104 |
| Joint speed mean (rad/s) | 0.248 | 0.285 | 0.234 | 0.290 | 0.316 | 0.346 | 0.326 |
| Command-limit occupancy | 0.306 | 0.164 | 0.313 | 0.289 | 0.249 | 0.241 | 0.225 |

## full 239-grasp distribution · default actuator · heavy_slippery · seed 6201

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.156 | 1.138 | 1.732 | 1.487 |
| Turns / 60 s | 2.282 | 2.279 | 3.518 | 2.983 |
| Yaw rate held (rad/s, 30 s) | 0.283 | 0.267 | 0.403 | 0.341 |
| Failure rate (30 s) | 0.238 | 0.177 | 0.189 | 0.174 |
| Failure rate (60 s) | 0.264 | 0.212 | 0.195 | 0.205 |
| P(held 60 s) | 0.736 | 0.788 | 0.805 | 0.795 |
| Survival (s, 60-s rollout) | 47.237 | 51.064 | 50.044 | 50.745 |
| Turns / 60 s if held 60 s | 2.951 | 2.759 | 4.274 | 3.627 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.148 | 0.158 | 0.156 | 0.147 |
| Utilisation p95 | 0.429 | 0.444 | 0.444 | 0.392 |
| Utilisation p99 | 0.617 | 0.606 | 0.610 | 0.500 |
| Tracking error mean (rad) | 0.095 | 0.096 | 0.107 | 0.104 |
| Joint speed mean (rad/s) | 0.257 | 0.247 | 0.321 | 0.326 |
| Command-limit occupancy | 0.266 | 0.328 | 0.259 | 0.225 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.071 | 1.140 | 1.257 | 1.836 | 1.777 | 1.583 | 1.487 |
| Turns / 60 s | 2.094 | 2.232 | 2.518 | 3.823 | 3.705 | 3.027 | 2.983 |
| Yaw rate held (rad/s, 30 s) | 0.263 | 0.276 | 0.310 | 0.427 | 0.407 | 0.373 | 0.341 |
| Failure rate (30 s) | 0.233 | 0.236 | 0.246 | 0.204 | 0.172 | 0.190 | 0.174 |
| Failure rate (60 s) | 0.268 | 0.261 | 0.262 | 0.208 | 0.177 | 0.201 | 0.205 |
| P(held 60 s) | 0.732 | 0.739 | 0.738 | 0.792 | 0.823 | 0.799 | 0.795 |
| Survival (s, 60-s rollout) | 47.504 | 47.494 | 46.714 | 49.325 | 50.864 | 49.943 | 50.745 |
| Turns / 60 s if held 60 s | 2.691 | 2.872 | 3.290 | 4.723 | 4.422 | 3.675 | 3.627 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.143 | 0.157 | 0.145 | 0.150 | 0.152 | 0.166 | 0.147 |
| Utilisation p95 | 0.411 | 0.443 | 0.434 | 0.412 | 0.444 | 0.475 | 0.392 |
| Utilisation p99 | 0.630 | 0.609 | 0.613 | 0.584 | 0.605 | 0.641 | 0.500 |
| Tracking error mean (rad) | 0.092 | 0.102 | 0.092 | 0.101 | 0.105 | 0.115 | 0.104 |
| Joint speed mean (rad/s) | 0.247 | 0.283 | 0.240 | 0.294 | 0.319 | 0.350 | 0.326 |
| Command-limit occupancy | 0.304 | 0.179 | 0.314 | 0.287 | 0.249 | 0.241 | 0.225 |

## full 239-grasp distribution · default actuator · high_delay · seed 6201

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.082 | 1.041 | 1.659 | 1.452 |
| Turns / 60 s | 2.184 | 2.103 | 3.428 | 3.010 |
| Yaw rate held (rad/s, 30 s) | 0.284 | 0.264 | 0.404 | 0.352 |
| Failure rate (30 s) | 0.283 | 0.217 | 0.213 | 0.224 |
| Failure rate (60 s) | 0.295 | 0.263 | 0.224 | 0.250 |
| P(held 60 s) | 0.705 | 0.737 | 0.776 | 0.750 |
| Survival (s, 60-s rollout) | 44.710 | 48.630 | 48.400 | 47.423 |
| Turns / 60 s if held 60 s | 2.981 | 2.684 | 4.314 | 3.911 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0001 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.151 | 0.163 | 0.163 | 0.147 |
| Utilisation p95 | 0.431 | 0.452 | 0.466 | 0.393 |
| Utilisation p99 | 0.624 | 0.623 | 0.636 | 0.502 |
| Tracking error mean (rad) | 0.103 | 0.105 | 0.119 | 0.114 |
| Joint speed mean (rad/s) | 0.255 | 0.247 | 0.319 | 0.326 |
| Command-limit occupancy | 0.260 | 0.322 | 0.259 | 0.224 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.028 | 1.035 | 1.185 | 1.714 | 1.696 | 1.568 | 1.452 |
| Turns / 60 s | 2.082 | 2.055 | 2.415 | 3.576 | 3.535 | 3.172 | 3.010 |
| Yaw rate held (rad/s, 30 s) | 0.270 | 0.277 | 0.306 | 0.416 | 0.408 | 0.387 | 0.352 |
| Failure rate (30 s) | 0.271 | 0.295 | 0.283 | 0.227 | 0.191 | 0.221 | 0.224 |
| Failure rate (60 s) | 0.282 | 0.311 | 0.292 | 0.239 | 0.198 | 0.235 | 0.250 |
| P(held 60 s) | 0.718 | 0.689 | 0.708 | 0.761 | 0.802 | 0.765 | 0.750 |
| Survival (s, 60-s rollout) | 45.313 | 44.279 | 44.539 | 47.692 | 49.511 | 47.996 | 47.423 |
| Turns / 60 s if held 60 s | 2.796 | 2.839 | 3.308 | 4.578 | 4.334 | 4.030 | 3.911 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.147 | 0.161 | 0.146 | 0.157 | 0.160 | 0.172 | 0.147 |
| Utilisation p95 | 0.412 | 0.450 | 0.432 | 0.441 | 0.471 | 0.486 | 0.393 |
| Utilisation p99 | 0.631 | 0.617 | 0.623 | 0.620 | 0.638 | 0.649 | 0.502 |
| Tracking error mean (rad) | 0.100 | 0.112 | 0.097 | 0.112 | 0.118 | 0.127 | 0.114 |
| Joint speed mean (rad/s) | 0.248 | 0.284 | 0.233 | 0.292 | 0.321 | 0.345 | 0.326 |
| Command-limit occupancy | 0.302 | 0.165 | 0.313 | 0.289 | 0.246 | 0.242 | 0.224 |

## full 239-grasp distribution · default actuator · obs_noise_x3 · seed 6201

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.201 | 1.161 | 1.793 | 1.607 |
| Turns / 60 s | 2.391 | 2.309 | 3.661 | 3.279 |
| Yaw rate held (rad/s, 30 s) | 0.293 | 0.261 | 0.409 | 0.361 |
| Failure rate (30 s) | 0.249 | 0.181 | 0.181 | 0.178 |
| Failure rate (60 s) | 0.263 | 0.230 | 0.194 | 0.199 |
| P(held 60 s) | 0.737 | 0.770 | 0.806 | 0.801 |
| Survival (s, 60-s rollout) | 46.802 | 50.955 | 50.332 | 50.604 |
| Turns / 60 s if held 60 s | 3.109 | 2.812 | 4.425 | 3.990 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.152 | 0.164 | 0.162 | 0.147 |
| Utilisation p95 | 0.433 | 0.453 | 0.466 | 0.393 |
| Utilisation p99 | 0.622 | 0.622 | 0.635 | 0.500 |
| Tracking error mean (rad) | 0.096 | 0.098 | 0.109 | 0.104 |
| Joint speed mean (rad/s) | 0.256 | 0.247 | 0.318 | 0.326 |
| Command-limit occupancy | 0.261 | 0.321 | 0.260 | 0.224 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.160 | 1.124 | 1.320 | 1.865 | 1.830 | 1.683 | 1.607 |
| Turns / 60 s | 2.308 | 2.209 | 2.657 | 3.844 | 3.782 | 3.358 | 3.279 |
| Yaw rate held (rad/s, 30 s) | 0.281 | 0.280 | 0.319 | 0.427 | 0.408 | 0.392 | 0.361 |
| Failure rate (30 s) | 0.232 | 0.270 | 0.245 | 0.197 | 0.148 | 0.197 | 0.178 |
| Failure rate (60 s) | 0.247 | 0.285 | 0.256 | 0.211 | 0.160 | 0.213 | 0.199 |
| P(held 60 s) | 0.753 | 0.715 | 0.744 | 0.789 | 0.840 | 0.787 | 0.801 |
| Survival (s, 60-s rollout) | 47.680 | 45.836 | 46.890 | 49.498 | 51.950 | 49.548 | 50.604 |
| Turns / 60 s if held 60 s | 2.936 | 2.937 | 3.454 | 4.731 | 4.414 | 4.128 | 3.990 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.147 | 0.161 | 0.147 | 0.158 | 0.159 | 0.170 | 0.147 |
| Utilisation p95 | 0.414 | 0.448 | 0.438 | 0.445 | 0.469 | 0.484 | 0.393 |
| Utilisation p99 | 0.630 | 0.612 | 0.625 | 0.620 | 0.638 | 0.647 | 0.500 |
| Tracking error mean (rad) | 0.093 | 0.104 | 0.091 | 0.103 | 0.108 | 0.116 | 0.104 |
| Joint speed mean (rad/s) | 0.248 | 0.285 | 0.234 | 0.289 | 0.318 | 0.346 | 0.326 |
| Command-limit occupancy | 0.306 | 0.164 | 0.312 | 0.290 | 0.249 | 0.241 | 0.224 |

## grasp 49 · effort actuator · nominal · seeds 6201+6202

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.587 | 1.600 | 2.349 | 2.154 |
| Turns / 60 s | 3.187 | 3.175 | 4.691 | 4.293 |
| Yaw rate held (rad/s, 30 s) | 0.333 | 0.335 | 0.492 | 0.452 |
| Failure rate (30 s) | 0.007 | 0.001 | 0.002 | 0.009 |
| Failure rate (60 s) | 0.009 | 0.001 | 0.003 | 0.012 |
| P(held 60 s) | 0.991 | 0.999 | 0.997 | 0.988 |
| Survival (s, 60-s rollout) | 59.623 | 59.967 | 59.907 | 59.535 |
| Turns / 60 s if held 60 s | 3.208 | 3.177 | 4.700 | 4.335 |
| Index MCP saturation | 0.0059 | 0.0002 | 0.0345 | 0.0013 |
| Index PIP saturation | 0.0003 | 0.0009 | 0.0004 | 0.0003 |
| Thumb abduction saturation | 0.0032 | 0.0012 | 0.0280 | 0.0037 |
| All joints at ≥ 98% of limit | 0.0068 | 0.0046 | 0.0098 | 0.0007 |
| Index blocked fraction | 0.0057 | 0.0008 | 0.0342 | 0.0014 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0342 | 0.0066 | 0.0245 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0326 | 0.0037 | 0.0164 | 0.0000 |
| Effort utilisation mean | 0.234 | 0.255 | 0.247 | 0.232 |
| Utilisation p95 | 0.647 | 0.656 | 0.714 | 0.616 |
| Utilisation p99 | 0.935 | 0.881 | 0.975 | 0.776 |
| Tracking error mean (rad) | 0.095 | 0.099 | 0.108 | 0.105 |
| Joint speed mean (rad/s) | 0.250 | 0.258 | 0.322 | 0.326 |
| Command-limit occupancy | 0.247 | 0.308 | 0.259 | 0.203 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.433 | 1.555 | 1.773 | 2.452 | 2.386 | 2.209 | 2.154 |
| Turns / 60 s | 2.913 | 3.099 | 3.550 | 4.928 | 4.797 | 4.347 | 4.293 |
| Yaw rate held (rad/s, 30 s) | 0.303 | 0.326 | 0.371 | 0.514 | 0.500 | 0.463 | 0.452 |
| Failure rate (30 s) | 0.016 | 0.003 | 0.001 | 0.002 | 0.000 | 0.003 | 0.009 |
| Failure rate (60 s) | 0.018 | 0.005 | 0.005 | 0.003 | 0.000 | 0.005 | 0.012 |
| P(held 60 s) | 0.982 | 0.995 | 0.995 | 0.997 | 1.000 | 0.995 | 0.988 |
| Survival (s, 60-s rollout) | 59.162 | 59.822 | 59.884 | 59.901 | 59.987 | 59.831 | 59.535 |
| Turns / 60 s if held 60 s | 2.957 | 3.109 | 3.558 | 4.938 | 4.798 | 4.363 | 4.335 |
| Index MCP saturation | 0.0152 | 0.0022 | 0.0003 | 0.0630 | 0.0120 | 0.0284 | 0.0013 |
| Index PIP saturation | 0.0003 | 0.0003 | 0.0003 | 0.0007 | 0.0003 | 0.0003 | 0.0003 |
| Thumb abduction saturation | 0.0038 | 0.0010 | 0.0047 | 0.0242 | 0.0006 | 0.0592 | 0.0037 |
| All joints at ≥ 98% of limit | 0.0075 | 0.0063 | 0.0068 | 0.0080 | 0.0109 | 0.0103 | 0.0007 |
| Index blocked fraction | 0.0147 | 0.0021 | 0.0003 | 0.0626 | 0.0117 | 0.0284 | 0.0014 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0962 | 0.0061 | 0.0002 | 0.0610 | 0.0037 | 0.0088 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0940 | 0.0039 | 0.0000 | 0.0398 | 0.0034 | 0.0059 | 0.0000 |
| Effort utilisation mean | 0.225 | 0.252 | 0.225 | 0.244 | 0.248 | 0.249 | 0.232 |
| Utilisation p95 | 0.611 | 0.685 | 0.644 | 0.689 | 0.728 | 0.726 | 0.616 |
| Utilisation p99 | 0.947 | 0.933 | 0.926 | 0.950 | 0.992 | 0.984 | 0.776 |
| Tracking error mean (rad) | 0.091 | 0.105 | 0.088 | 0.102 | 0.110 | 0.114 | 0.105 |
| Joint speed mean (rad/s) | 0.241 | 0.290 | 0.220 | 0.284 | 0.331 | 0.352 | 0.326 |
| Command-limit occupancy | 0.301 | 0.131 | 0.308 | 0.297 | 0.243 | 0.238 | 0.203 |

## grasp 49 · default actuator · nominal · seeds 6201+6202

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.597 | 1.605 | 2.361 | 2.164 |
| Turns / 60 s | 3.202 | 3.180 | 4.716 | 4.309 |
| Yaw rate held (rad/s, 30 s) | 0.336 | 0.336 | 0.495 | 0.454 |
| Failure rate (30 s) | 0.008 | 0.000 | 0.003 | 0.006 |
| Failure rate (60 s) | 0.010 | 0.002 | 0.003 | 0.009 |
| P(held 60 s) | 0.990 | 0.998 | 0.997 | 0.991 |
| Survival (s, 60-s rollout) | 59.558 | 59.949 | 59.865 | 59.668 |
| Turns / 60 s if held 60 s | 3.227 | 3.183 | 4.727 | 4.338 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.149 | 0.162 | 0.157 | 0.147 |
| Utilisation p95 | 0.410 | 0.418 | 0.453 | 0.392 |
| Utilisation p99 | 0.591 | 0.560 | 0.613 | 0.493 |
| Tracking error mean (rad) | 0.094 | 0.099 | 0.108 | 0.105 |
| Joint speed mean (rad/s) | 0.251 | 0.259 | 0.323 | 0.326 |
| Command-limit occupancy | 0.248 | 0.308 | 0.259 | 0.203 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.452 | 1.551 | 1.788 | 2.469 | 2.398 | 2.217 | 2.164 |
| Turns / 60 s | 2.940 | 3.090 | 3.576 | 4.969 | 4.814 | 4.364 | 4.309 |
| Yaw rate held (rad/s, 30 s) | 0.306 | 0.326 | 0.375 | 0.517 | 0.502 | 0.466 | 0.454 |
| Failure rate (30 s) | 0.014 | 0.007 | 0.004 | 0.001 | 0.001 | 0.005 | 0.006 |
| Failure rate (60 s) | 0.015 | 0.009 | 0.008 | 0.001 | 0.003 | 0.007 | 0.009 |
| P(held 60 s) | 0.985 | 0.991 | 0.992 | 0.999 | 0.997 | 0.993 | 0.991 |
| Survival (s, 60-s rollout) | 59.272 | 59.648 | 59.754 | 59.957 | 59.932 | 59.707 | 59.668 |
| Turns / 60 s if held 60 s | 2.979 | 3.110 | 3.592 | 4.973 | 4.821 | 4.388 | 4.338 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.143 | 0.161 | 0.142 | 0.155 | 0.158 | 0.159 | 0.147 |
| Utilisation p95 | 0.388 | 0.437 | 0.406 | 0.434 | 0.463 | 0.461 | 0.392 |
| Utilisation p99 | 0.599 | 0.593 | 0.580 | 0.595 | 0.623 | 0.620 | 0.493 |
| Tracking error mean (rad) | 0.091 | 0.105 | 0.087 | 0.101 | 0.110 | 0.114 | 0.105 |
| Joint speed mean (rad/s) | 0.242 | 0.291 | 0.221 | 0.286 | 0.332 | 0.352 | 0.326 |
| Command-limit occupancy | 0.302 | 0.132 | 0.309 | 0.296 | 0.243 | 0.238 | 0.203 |

## robust 10-grasp subset · effort actuator · nominal · seeds 6201+6202

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.569 | 1.484 | 2.241 | 2.060 |
| Turns / 60 s | 3.144 | 3.015 | 4.558 | 4.161 |
| Yaw rate held (rad/s, 30 s) | 0.331 | 0.312 | 0.472 | 0.435 |
| Failure rate (30 s) | 0.015 | 0.008 | 0.012 | 0.017 |
| Failure rate (60 s) | 0.021 | 0.013 | 0.014 | 0.022 |
| P(held 60 s) | 0.979 | 0.987 | 0.986 | 0.978 |
| Survival (s, 60-s rollout) | 59.218 | 59.539 | 59.359 | 59.111 |
| Turns / 60 s if held 60 s | 3.189 | 3.040 | 4.611 | 4.231 |
| Index MCP saturation | 0.0144 | 0.0036 | 0.0334 | 0.0019 |
| Index PIP saturation | 0.0011 | 0.0023 | 0.0010 | 0.0004 |
| Thumb abduction saturation | 0.0047 | 0.0029 | 0.0262 | 0.0037 |
| All joints at ≥ 98% of limit | 0.0100 | 0.0065 | 0.0119 | 0.0010 |
| Index blocked fraction | 0.0148 | 0.0053 | 0.0334 | 0.0020 |
| Index jam proxy (≥ 1 s, 60 s) | 0.1010 | 0.0684 | 0.0627 | 0.0012 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0968 | 0.0652 | 0.0531 | 0.0005 |
| Effort utilisation mean | 0.236 | 0.254 | 0.255 | 0.232 |
| Utilisation p95 | 0.677 | 0.671 | 0.730 | 0.619 |
| Utilisation p99 | 0.977 | 0.920 | 0.988 | 0.782 |
| Tracking error mean (rad) | 0.096 | 0.098 | 0.110 | 0.104 |
| Joint speed mean (rad/s) | 0.255 | 0.252 | 0.321 | 0.325 |
| Command-limit occupancy | 0.264 | 0.315 | 0.258 | 0.209 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.469 | 1.499 | 1.739 | 2.334 | 2.279 | 2.110 | 2.060 |
| Turns / 60 s | 2.943 | 2.996 | 3.494 | 4.787 | 4.677 | 4.210 | 4.161 |
| Yaw rate held (rad/s, 30 s) | 0.310 | 0.316 | 0.366 | 0.491 | 0.478 | 0.446 | 0.435 |
| Failure rate (30 s) | 0.016 | 0.015 | 0.014 | 0.011 | 0.006 | 0.020 | 0.017 |
| Failure rate (60 s) | 0.018 | 0.019 | 0.025 | 0.012 | 0.007 | 0.022 | 0.022 |
| P(held 60 s) | 0.982 | 0.981 | 0.975 | 0.988 | 0.993 | 0.978 | 0.978 |
| Survival (s, 60-s rollout) | 59.220 | 59.240 | 59.193 | 59.434 | 59.689 | 58.954 | 59.111 |
| Turns / 60 s if held 60 s | 2.983 | 3.038 | 3.546 | 4.838 | 4.704 | 4.291 | 4.231 |
| Index MCP saturation | 0.0266 | 0.0075 | 0.0091 | 0.0620 | 0.0146 | 0.0236 | 0.0019 |
| Index PIP saturation | 0.0011 | 0.0007 | 0.0017 | 0.0014 | 0.0007 | 0.0008 | 0.0004 |
| Thumb abduction saturation | 0.0044 | 0.0021 | 0.0076 | 0.0290 | 0.0050 | 0.0447 | 0.0037 |
| All joints at ≥ 98% of limit | 0.0103 | 0.0074 | 0.0124 | 0.0089 | 0.0121 | 0.0149 | 0.0010 |
| Index blocked fraction | 0.0266 | 0.0075 | 0.0102 | 0.0621 | 0.0144 | 0.0237 | 0.0020 |
| Index jam proxy (≥ 1 s, 60 s) | 0.1011 | 0.0930 | 0.1089 | 0.0769 | 0.0205 | 0.0908 | 0.0012 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0920 | 0.0906 | 0.1079 | 0.0542 | 0.0203 | 0.0850 | 0.0005 |
| Effort utilisation mean | 0.227 | 0.250 | 0.231 | 0.245 | 0.249 | 0.270 | 0.232 |
| Utilisation p95 | 0.634 | 0.694 | 0.703 | 0.693 | 0.732 | 0.765 | 0.619 |
| Utilisation p99 | 0.984 | 0.947 | 1.000 | 0.964 | 1.000 | 1.000 | 0.782 |
| Tracking error mean (rad) | 0.092 | 0.104 | 0.092 | 0.102 | 0.110 | 0.118 | 0.104 |
| Joint speed mean (rad/s) | 0.241 | 0.285 | 0.239 | 0.285 | 0.328 | 0.351 | 0.325 |
| Command-limit occupancy | 0.315 | 0.156 | 0.322 | 0.294 | 0.245 | 0.236 | 0.209 |

## robust 10-grasp subset · default actuator · nominal · seeds 6201+6202

| Metric | A_mean | R1 | C5_mean | EFFORT_1 |
|---|---:|---:|---:|---:|
| Turns / 30 s | 1.587 | 1.499 | 2.268 | 2.061 |
| Turns / 60 s | 3.168 | 3.030 | 4.600 | 4.166 |
| Yaw rate held (rad/s, 30 s) | 0.335 | 0.315 | 0.477 | 0.435 |
| Failure rate (30 s) | 0.015 | 0.009 | 0.011 | 0.015 |
| Failure rate (60 s) | 0.022 | 0.014 | 0.013 | 0.019 |
| P(held 60 s) | 0.978 | 0.986 | 0.987 | 0.981 |
| Survival (s, 60-s rollout) | 59.184 | 59.479 | 59.391 | 59.237 |
| Turns / 60 s if held 60 s | 3.215 | 3.059 | 4.650 | 4.226 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.151 | 0.162 | 0.162 | 0.148 |
| Utilisation p95 | 0.430 | 0.427 | 0.462 | 0.394 |
| Utilisation p99 | 0.619 | 0.583 | 0.626 | 0.498 |
| Tracking error mean (rad) | 0.096 | 0.098 | 0.110 | 0.104 |
| Joint speed mean (rad/s) | 0.256 | 0.253 | 0.323 | 0.325 |
| Command-limit occupancy | 0.265 | 0.315 | 0.258 | 0.210 |

Per training seed:

| Metric | A1 | A2 | A3 | C5_1 | C5_2 | C5_3 | EFFORT_1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Turns / 30 s | 1.498 | 1.501 | 1.763 | 2.365 | 2.298 | 2.140 | 2.061 |
| Turns / 60 s | 2.982 | 2.991 | 3.532 | 4.847 | 4.695 | 4.258 | 4.166 |
| Yaw rate held (rad/s, 30 s) | 0.316 | 0.316 | 0.371 | 0.497 | 0.482 | 0.453 | 0.435 |
| Failure rate (30 s) | 0.016 | 0.018 | 0.012 | 0.008 | 0.006 | 0.020 | 0.015 |
| Failure rate (60 s) | 0.022 | 0.023 | 0.022 | 0.010 | 0.008 | 0.023 | 0.019 |
| P(held 60 s) | 0.978 | 0.977 | 0.978 | 0.990 | 0.992 | 0.977 | 0.981 |
| Survival (s, 60-s rollout) | 59.175 | 59.106 | 59.270 | 59.570 | 59.675 | 58.928 | 59.237 |
| Turns / 60 s if held 60 s | 3.025 | 3.042 | 3.579 | 4.886 | 4.724 | 4.341 | 4.226 |
| Index MCP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index PIP saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Thumb abduction saturation | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| All joints at ≥ 98% of limit | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index blocked fraction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 60 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Index jam proxy (≥ 1 s, 30 s) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Effort utilisation mean | 0.145 | 0.160 | 0.147 | 0.155 | 0.159 | 0.172 | 0.148 |
| Utilisation p95 | 0.404 | 0.442 | 0.445 | 0.436 | 0.466 | 0.486 | 0.394 |
| Utilisation p99 | 0.622 | 0.601 | 0.634 | 0.602 | 0.630 | 0.646 | 0.498 |
| Tracking error mean (rad) | 0.092 | 0.104 | 0.092 | 0.102 | 0.109 | 0.118 | 0.104 |
| Joint speed mean (rad/s) | 0.243 | 0.285 | 0.240 | 0.287 | 0.330 | 0.352 | 0.325 |
| Command-limit occupancy | 0.314 | 0.157 | 0.323 | 0.294 | 0.245 | 0.236 | 0.210 |

