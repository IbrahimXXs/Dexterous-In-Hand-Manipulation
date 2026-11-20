# Stage check at update 500 (PROTOCOL.md Section 3), 2026-11-20 ~02:45

Validation seed 7001, nominal, 1024 environments, 30 s (A1 and R1 from their own curve files, same evaluator).

| Update 500 | A1 | R1 | P0_1 | C5_1 |
|---|---|---|---|---|
| 30-s turns | 0.676 | 0.647 | 0.932 | 0.977 |
| Failure | 0.388 | 0.304 | 0.201 | 0.264 |
| Survival (s) | 22.7 | 25.5 | 26.7 | 24.9 |
| Yaw rate (rad/s) | 0.184 | 0.156 | 0.215 | 0.239 |
| 60-s turns | — | 1.105 | 1.840 | 1.834 |
| 60-s P(held) | — | 0.546 | 0.704 | 0.657 |

- **Stop rules:** yaw rate ≥ 50% of R1's (0.078) for both. No instability: action noise 1.21 and 1.18, surrogate losses normal.
- **No exploit:** the validation metric is genuine cube yaw, and it rises with the training rotation reward (P0 0.074, C5 0.100 per step).
- **Retention penalty share of the rotation reward:** P0 7% (tilt part only), C5 17%.

Decision: continue both runs to 3000 updates.

# Futility check at update 1500 (PROTOCOL.md Section 3), 2026-11-20 04:18

Rule: stop if turns < R1(1500) − 0.15 = 0.937 AND failure > R1(1500) + 0.10 = 0.264.

| Update 1500 | A1 | A2 | A3 | R1 | P0_1 | C5_1 |
|---|---|---|---|---|---|---|
| 30-s turns | 0.971 | 1.015 | 1.263 | 1.087 | 1.270 | 1.718 |
| Failure | 0.296 | 0.271 | 0.262 | 0.164 | 0.116 | 0.197 |
| Yaw rate (rad/s) | 0.245 | 0.252 | 0.312 | 0.242 | 0.280 | 0.395 |
| 60-s turns | — | — | — | 2.237 | 2.616 | 3.555 |
| 60-s P(held) | — | — | — | 0.804 | 0.857 | 0.797 |

Decision: neither candidate meets the futility rule; both continue to 3000.
