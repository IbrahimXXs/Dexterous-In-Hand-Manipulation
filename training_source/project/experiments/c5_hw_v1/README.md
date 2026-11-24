# C5-HW-V1: effort-matched C5 successor (simulation)

| Document | Content |
|---|---|
| `SPEC.md` | The 20 Nov hardware evidence, hypotheses, conditions and the pre-registered §7 criteria (committed in f8e47b8) |
| `PROTOCOL_EFFORT_V1.md` | The single-seed protocol for C5-HW-EFFORT-V1, frozen and hashed before training (`logs/frozen.sha256`) |
| `REPORT_EFFORT_V1.md` | Results: comparison with A and C5, index-jam analysis, deployment verification, limitations |
| `TOMORROW_LAB_READINESS.md` | Go/no-go for supervisor review and a proposed supervised session plan (nothing scheduled or started) |

**C5-HW-EFFORT-V1** is one training run (PPO seed 1), i.e. a **single-seed exploratory experiment**. Its pre-registered verdict is **SINGLE-SEED PASS (exploratory; not confirmed)**, on the full distribution under the 350 mA-equivalent actuator:

| | EFFORT_1 | C5 3-seed mean | Ratio or change |
|---|---|---|---|
| Index-jam proxy | 0.5% | 14.6% | −96.5% |
| Turns per 30 s | 1.625 | 1.784 | 0.911 (bar 0.90); 1.36 × A |
| Failure, 30 s | 0.178 | 0.180 | −0.003 |

Simulation only. No hardware was run and no kit file or safety setting was changed. Not trained, as instructed: seeds 2 and 3, C5-HW-delay, and the combined condition.

## Reproduce (from `training_source/project`)

```bash
P=/home/ibrahim.alsarraj/miniconda3/envs/leap-estimator/bin/python
$P experiments/c5_hw_v1/scripts/calibrate_effort_weight.py      # w (already frozen in env_cfgs.py)
SEED=1 sbatch experiments/c5_hw_v1/scripts/sbatch_train.sh      # seed 1 only (others refused); validation curve afterwards
$P experiments/c5_hw_v1/scripts/curves.py assess                # selection -> results/convergence.json
$P experiments/c5_hw_v1/scripts/freeze_checkpoints.py
$P experiments/c5_hw_v1/scripts/eval_items.py --make-items --out experiments/c5_hw_v1/items/baselines_eval.json
$P experiments/c5_hw_v1/scripts/eval_items.py --make-items --policies EFFORT_1 --out experiments/c5_hw_v1/items/effort1_eval.json
ITEMS=experiments/c5_hw_v1/items/<file>.json sbatch experiments/c5_hw_v1/scripts/sbatch_eval.sh   # n workers
$P experiments/c5_hw_v1/scripts/compare_effort.py              # verdict, tables, figures
$P experiments/c5_hw_v1/scripts/build_deploy.py                # only if the verdict is a pass
$P experiments/c5_hw_v1/scripts/export_candidate.py
$P experiments/c5_hw_v1/scripts/rehearse_deploy.py
bash experiments/c5_hw_v1/scripts/check_preservation.sh
$P -m pytest tests/test_c5_hw_v1.py
```

Per-environment rollout arrays (`data/`) are gitignored. Everything reported is in `results/`.
