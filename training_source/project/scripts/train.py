"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  video: bool = False
  # None maps to env.max_episode_length (full nominal horizon) in run_train().
  video_length: int | None = None
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  # Keep checkpoint models/moments but apply the CLI PPO learning rate.
  reset_optimizer_learning_rate_on_resume: bool = False
  # Initialize the actor mean from a compatible policy without resuming its
  # critic, optimizer, iteration counter, or exploration standard deviation.
  bootstrap_actor_checkpoint: str | None = None
  # Optionally seed appended actor inputs from semantically matching source
  # columns. For Task 1, columns 32:35 encode target-minus-object position.
  bootstrap_appended_input_source_start: int | None = None
  bootstrap_appended_input_scale: float = 1.0
  # Adapter stage: preserve the mature policy and update only first-layer
  # columns belonging to newly appended observations.
  bootstrap_train_appended_inputs_only: bool = False
  # Resume/same-width adapter stage: freeze the actor and update first-layer
  # columns from this input index onward (Task 1 uses 57).
  train_actor_appended_inputs_from: int | None = None
  # Delta-action tasks need an exactly zero fresh deterministic action;
  # otherwise a small random mean integrates until joints hit their limits.
  zero_actor_output: bool = False
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _prepare_agent_cfg(agent_cfg: RslRlOnPolicyRunnerCfg) -> dict:
  """Convert dataclass config to runner dict and drop incompatible MLP fields."""
  cfg_dict = asdict(agent_cfg)
  for model_key in ("actor", "critic"):
    model_cfg = cfg_dict.get(model_key)
    if isinstance(model_cfg, dict):
      if model_cfg.get("class_name", "MLPModel") != "CNNModel":
        model_cfg.pop("cnn_cfg", None)
  return cfg_dict

def _bootstrap_actor(
  runner: OnPolicyRunner,
  checkpoint_file: str,
  appended_input_source_start: int | None = None,
  appended_input_scale: float = 1.0,
) -> int:
  """Warm-start a wider actor from a compatible narrower actor checkpoint.

  Shared MLP weights and normalizer statistics are copied. Appended input
  columns start at zero, so the initial mean remains the proven source policy.
  The target PPO configuration keeps its own low exploration standard deviation.
  """
  checkpoint_path = Path(checkpoint_file).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(
      f"Actor bootstrap checkpoint not found: {checkpoint_path}"
    )
  loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  source = loaded.get("actor_state_dict")
  if not isinstance(source, dict):
    raise ValueError(f"Checkpoint has no actor_state_dict: {checkpoint_path}")

  actor = runner.alg.get_policy()
  target = actor.state_dict()
  first_weight_key = "mlp.0.weight"
  if first_weight_key not in source or first_weight_key not in target:
    raise ValueError("Actor checkpoint is missing mlp.0.weight")
  source_input_dim = source[first_weight_key].shape[1]
  target_input_dim = target[first_weight_key].shape[1]
  if source_input_dim > target_input_dim:
    raise ValueError(
      "Bootstrap actor input is wider than target: "
      f"source={source_input_dim}, target={target_input_dim}"
    )

  copied: list[str] = []
  for key, target_tensor in target.items():
    source_tensor = source.get(key)
    if source_tensor is None or key in {"std", "log_std"}:
      continue
    if source_tensor.shape == target_tensor.shape:
      target[key] = source_tensor.to(
        device=target_tensor.device, dtype=target_tensor.dtype
      )
      copied.append(key)

  # Copy common first-layer columns and neutralize appended camera/goal inputs.
  first_weight = target[first_weight_key].clone()
  first_weight.zero_()
  source_first_weight = source[first_weight_key].to(
    device=first_weight.device, dtype=first_weight.dtype
  )
  first_weight[:, :source_input_dim] = source_first_weight
  appended_input_dim = target_input_dim - source_input_dim
  appended_copy_description = "zero-initialized"
  if appended_input_source_start is not None:
    source_start = int(appended_input_source_start)
    source_end = source_start + appended_input_dim
    if appended_input_dim <= 0:
      raise ValueError(
        "Appended-input initialization requires a wider target actor."
      )
    if source_start < 0 or source_end > source_input_dim:
      raise ValueError(
        "Appended-input source range is outside the source actor: "
        f"[{source_start}:{source_end}] versus width {source_input_dim}."
      )
    if not torch.isfinite(torch.tensor(float(appended_input_scale))):
      raise ValueError("bootstrap_appended_input_scale must be finite.")
    first_weight[:, source_input_dim:] = (
      float(appended_input_scale)
      * source_first_weight[:, source_start:source_end]
    )
    appended_copy_description = (
      f"copied source columns {source_start}:{source_end} "
      f"at scale {float(appended_input_scale):g}"
    )
  target[first_weight_key] = first_weight
  copied.append(first_weight_key)

  # Preserve reliable common normalizer statistics. New bounded observations
  # begin at identity normalization and adapt during face-task training.
  for suffix in ("_mean", "_var", "_std"):
    key = f"obs_normalizer.{suffix}"
    if key in source and key in target:
      target_tensor = target[key].clone()
      target_tensor[..., :source_input_dim] = source[key].to(
        device=target_tensor.device, dtype=target_tensor.dtype
      )
      target[key] = target_tensor
  count_key = "obs_normalizer.count"
  if count_key in target:
    target[count_key] = torch.tensor(
      1_000_000, device=target[count_key].device, dtype=target[count_key].dtype
    )

  actor.load_state_dict(target)
  print(
    "[INFO] Bootstrapped actor mean from "
    f"{checkpoint_path} ({source_input_dim} -> {target_input_dim} inputs; "
    f"{len(set(copied))} shared tensors; appended inputs "
    f"{appended_copy_description})."
  )
  return source_input_dim


def _train_appended_actor_inputs_only(
  runner: OnPolicyRunner,
  source_input_dim: int,
) -> None:
  """Freeze the source actor and optimize only newly appended input columns."""
  actor = runner.alg.get_policy()
  parameters = dict(actor.named_parameters())
  first_weight_key = "mlp.0.weight"
  if first_weight_key not in parameters:
    raise ValueError("Actor has no trainable mlp.0.weight parameter.")
  first_weight = parameters[first_weight_key]
  if source_input_dim >= first_weight.shape[1]:
    raise ValueError(
      "Appended-input-only training requires a wider target actor: "
      f"source={source_input_dim}, target={first_weight.shape[1]}."
    )

  for parameter in actor.parameters():
    parameter.requires_grad_(False)
  first_weight.requires_grad_(True)
  gradient_mask = torch.zeros_like(first_weight)
  gradient_mask[:, source_input_dim:] = 1.0
  first_weight.register_hook(lambda gradient: gradient * gradient_mask)
  print(
    "[INFO] Frozen source actor parameters; training only first-layer "
    f"columns {source_input_dim}:{first_weight.shape[1]}."
  )


def _zero_actor_output(runner: OnPolicyRunner) -> None:
  """Zero the final actor layer while preserving stochastic exploration."""
  actor = runner.alg.get_policy()
  linear_layers = [
    module for module in actor.modules() if isinstance(module, torch.nn.Linear)
  ]
  if not linear_layers:
    raise RuntimeError("Actor contains no Linear layer to zero-initialize.")
  output_layer = linear_layers[-1]
  torch.nn.init.zeros_(output_layer.weight)
  if output_layer.bias is not None:
    torch.nn.init.zeros_(output_layer.bias)
  print("[INFO] Zero-initialized the fresh actor output layer for delta actions.")


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = (
    cfg.env.commands is not None
    and "motion" in cfg.env.commands
    and isinstance(cfg.env.commands["motion"], MotionCommandCfg)
  )

  if is_tracking_task:
    if not cfg.registry_name:
      raise ValueError("Must provide --registry-name for tracking tasks.")

    # Check if the registry name includes alias, if not, append ":latest".
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)

    assert cfg.env.commands is not None
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.checkpoint_file is not None:
    if cfg.wandb_run_path is not None:
      raise ValueError(
        "Use either checkpoint_file or wandb_run_path for resume, not both."
      )
    resume_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not resume_path.is_file():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    if rank == 0:
      print(f"[INFO]: Loading checkpoint from explicit path: {resume_path}")
  elif cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    # Default to a full horizon clip when video_length is not explicitly set.
    # This avoids ultra-short clips when env[0] terminates early.
    video_length = (
      cfg.video_length
      if cfg.video_length is not None
      else env.unwrapped.max_episode_length
    )
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      # `step` is the environment control-step index on this rank.
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=video_length,
      disable_logger=True,
    )
    print(
      "[INFO] Video config: "
      f"interval_steps={cfg.video_interval}, length_frames={video_length}"
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = _prepare_agent_cfg(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = OnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))
    if cfg.reset_optimizer_learning_rate_on_resume:
      requested_learning_rate = float(cfg.agent.algorithm.learning_rate)
      for parameter_group in runner.alg.optimizer.param_groups:
        parameter_group["lr"] = requested_learning_rate
      runner.alg.learning_rate = requested_learning_rate
      print(
        "[INFO] Reset resumed optimizer learning rate to "
        f"{requested_learning_rate:g}."
      )
  bootstrap_source_input_dim: int | None = None
  if cfg.bootstrap_actor_checkpoint is not None:
    if resume_path is not None:
      raise ValueError(
        "Use either bootstrap_actor_checkpoint or checkpoint resume, not both."
      )
    bootstrap_source_input_dim = _bootstrap_actor(
      runner,
      cfg.bootstrap_actor_checkpoint,
      appended_input_source_start=cfg.bootstrap_appended_input_source_start,
      appended_input_scale=cfg.bootstrap_appended_input_scale,
    )
  if (
    cfg.bootstrap_train_appended_inputs_only
    and cfg.train_actor_appended_inputs_from is not None
  ):
    raise ValueError(
      "Use either bootstrap_train_appended_inputs_only or "
      "train_actor_appended_inputs_from, not both."
    )
  if cfg.bootstrap_train_appended_inputs_only:
    if bootstrap_source_input_dim is None:
      raise ValueError(
        "bootstrap_train_appended_inputs_only requires an actor bootstrap."
      )
    _train_appended_actor_inputs_only(runner, bootstrap_source_input_dim)
  if cfg.train_actor_appended_inputs_from is not None:
    if resume_path is None and cfg.bootstrap_actor_checkpoint is None:
      raise ValueError(
        "train_actor_appended_inputs_from requires checkpoint resume/bootstrap."
      )
    _train_appended_actor_inputs_only(
      runner, int(cfg.train_actor_appended_inputs_from)
    )

  if cfg.zero_actor_output:
    if resume_path is not None or cfg.bootstrap_actor_checkpoint is not None:
      raise ValueError(
        "zero_actor_output is only valid for a fresh policy, not resume/bootstrap."
      )
    _zero_actor_output(runner)

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
