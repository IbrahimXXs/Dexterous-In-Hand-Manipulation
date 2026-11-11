# Original continuous-yaw training source

This directory preserves the source for the exact `Mjlab-Leap-Right-HandCube-MBZUAI-Rotate` task, together with the exported environment and agent configuration used for the supplied checkpoint.

- `project/` contains the MJLab task, robot model, rewards, events, training and playback scripts, and its own `pyproject.toml` and `uv.lock`.
- `training_params/` contains the exported environment and agent YAML files.
- `../checkpoint/model_400.pt` is the original training checkpoint.
- `../checkpoint/policy_weights.npz` is the same actor exported for lightweight NumPy inference.

The training project is intentionally separate from the portable viewer. Its original stack uses Python 3.10, MJLab/MuJoCo Warp, PyTorch and GPU-oriented dependencies. A Linux workstation with a compatible NVIDIA/CUDA setup is recommended. macOS students can inspect and modify every source file and can run the portable native-MuJoCo viewer, but the original accelerated training environment is not claimed to be macOS-compatible.

This is continuous positive-Z/yaw rotation. Do not substitute one-shot orientation-goal or arbitrary cube-reorientation tasks when comparing results.

## Reproduce the exact stable preview

From the standalone package root, use a working Python 3.10 MJLab/MuJoCo-Warp environment and run:

```bash
PROJECT_PYTHON=/absolute/path/to/the/working/python3.10
"$PROJECT_PYTHON" training_source/project/scripts/record_policy_video.py \
  Mjlab-Leap-Right-HandCube-MBZUAI-Rotate \
  --checkpoint-file checkpoint/model_400.pt \
  --num-envs 1 --length 600 --width 960 --height 720 \
  --seed 11 --mbzuai-cube \
  --output-dir outputs/exact_mjlab_preview
```

This is the command that generated the stable 30-second bundle preview. The cross-platform portable viewer replays a state trajectory captured from this exact task/checkpoint/seed contract.

