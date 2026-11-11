import argparse
from dataclasses import asdict
import math
from pathlib import Path

import mujoco
import torch

from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_inv, quat_mul
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

import mjlab.tasks  # noqa: F401
import in_hand_rotation_mjlab.tasks  # noqa: F401
from in_hand_rotation_mjlab.tasks.hand_cube import mdp
from in_hand_rotation_mjlab.tasks.hand_cube.config.leap_right_cylinder_translate.env_cfgs import (
    CYLINDER_SPAWN_XY,
    TABLE_HEIGHT,
)
from in_hand_rotation_mjlab.tasks.hand_cube.mdp.cylinder_pickup import (
    MANIPULATE_PHASE,
)


def add_mbzuai_cube_visual(env_cfg) -> None:
    """Replace only the play-time cube pixels with the MBZUAI texture."""
    asset_root = Path(__file__).resolve().parents[1] / "assets/mbzuai_cube"
    mesh_path = asset_root / "dex_cube.obj"
    texture_path = asset_root / "mbzuai_cube.png"
    missing = [path for path in (mesh_path, texture_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing MBZUAI cube assets: "
            + ", ".join(str(path) for path in missing)
        )

    cube_cfg = env_cfg.scene.entities["cube"]
    original_spec_fn = cube_cfg.spec_fn

    def mbzuai_cube_spec() -> mujoco.MjSpec:
        spec = original_spec_fn()
        texture_name = "record_mbzuai_cube_texture"
        material_name = "record_mbzuai_cube_material"
        mesh_name = "record_mbzuai_cube_mesh"
        spec.add_texture(
            name=texture_name,
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=str(texture_path),
        )
        spec.add_material(
            name=material_name,
            textures=["", texture_name],
            rgba=(1.0, 1.0, 1.0, 1.0),
        )
        cube_half_extents = tuple(
            float(value) for value in spec.geom("cube_geom").size
        )
        spec.add_mesh(
            name=mesh_name,
            file=str(mesh_path),
            scale=cube_half_extents,
        )
        body = spec.body("cube")
        body.add_geom(
            name="record_mbzuai_cube_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            material=material_name,
            contype=0,
            conaffinity=0,
            density=0.0,
            group=2,
        )
        # Retain the original box as the exact collision/inertia model.
        spec.geom("cube_geom").rgba[:] = (0.0, 0.0, 0.0, 0.0)
        return spec

    cube_cfg.spec_fn = mbzuai_cube_spec


def prepare_agent_cfg(agent_cfg) -> dict:
    cfg_dict = asdict(agent_cfg)
    for model_key in ("actor", "critic"):
        model_cfg = cfg_dict.get(model_key)
        if isinstance(model_cfg, dict):
            if model_cfg.get("class_name", "MLPModel") != "CNNModel":
                model_cfg.pop("cnn_cfg", None)
    return cfg_dict


def set_relative_goal_from_current_cube(
    raw_env: ManagerBasedRlEnv,
    axis_index: int,
    angle_rad: float,
) -> None:
    """Hold a goal rotated from the current cube pose about a palm-frame axis."""
    command = raw_env.command_manager.get_term("pose_goal")
    hand = raw_env.scene["robot"]
    cube = raw_env.scene["cube"]
    palm_ids, _ = hand.find_bodies("palm", preserve_order=True)
    if len(palm_ids) != 1:
        raise RuntimeError(f"Expected one palm body, found {len(palm_ids)}.")

    palm_quat_w = hand.data.body_link_quat_w[:, palm_ids[0]]
    cube_quat_p = quat_mul(
        quat_inv(palm_quat_w), cube.data.root_link_quat_w
    )
    half = 0.5 * angle_rad
    delta = torch.zeros_like(cube_quat_p)
    delta[:, 0] = math.cos(half)
    delta[:, axis_index + 1] = math.sin(half)
    target_quat_w = quat_mul(
        palm_quat_w, quat_mul(delta, cube_quat_p)
    )
    command.set_goal_quaternion(target_quat_w)


def current_pose_error_deg(raw_env: ManagerBasedRlEnv) -> float:
    command = raw_env.command_manager.get_term("pose_goal")
    cube_quat = torch.nn.functional.normalize(
        raw_env.scene["cube"].data.root_link_quat_w, dim=-1
    )
    target_quat = torch.nn.functional.normalize(command.target_quat_w, dim=-1)
    dot = torch.sum(cube_quat * target_quat, dim=-1).abs()
    angle = 2.0 * torch.acos(torch.clamp(dot, max=1.0))
    return math.degrees(float(angle.mean().item()))


def has_pose_goal(raw_env: ManagerBasedRlEnv) -> bool:
    """Return whether this task exposes the continuous reorientation command."""
    try:
        raw_env.command_manager.get_term("pose_goal")
    except (KeyError, ValueError):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--viewer-env-idx",
        type=int,
        default=0,
        help="Parallel environment index followed by the asset-body camera.",
    )
    parser.add_argument(
        "--disable-recovery-event",
        action="store_true",
        help="Match the first-pickup evaluator without timed object returns.",
    )
    parser.add_argument(
        "--pickup-recovery",
        choices=("none", "external_replace"),
        default="none",
        help=(
            "For Task 1, record the evaluator's physical order: remove the "
            "cylinder, wait for measured opening, replace it on the table, "
            "and attempt a second pickup."
        ),
    )
    parser.add_argument("--output-dir", default="videos/manual_play")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Environment/reset seed for reproducible checkpoint comparisons.",
    )
    parser.add_argument(
        "--goal-script",
        choices=("none", "local_xyz"),
        default="none",
        help=(
            "Optional interactive-style goal sequence. local_xyz requests "
            "small palm-frame X/Y/Z rotations from the current cube pose."
        ),
    )
    parser.add_argument("--goal-step-deg", type=float, default=15.0)
    parser.add_argument("--goal-hold-frames", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument(
        "--mbzuai-cube",
        action="store_true",
        help="Render the six-color MBZUAI skin without changing cube physics.",
    )
    args = parser.parse_args()

    if not 0.0 < args.goal_step_deg <= 90.0:
        raise ValueError("--goal-step-deg must lie in (0, 90].")
    if args.goal_hold_frames <= 0:
        raise ValueError("--goal-hold-frames must be positive.")

    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = Path(args.checkpoint_file)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    if args.mbzuai_cube:
        add_mbzuai_cube_visual(env_cfg)
    if args.disable_recovery_event or args.pickup_recovery != "none":
        env_cfg.events.pop("return_cylinder_to_table", None)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    if not 0 <= args.viewer_env_idx < args.num_envs:
        raise ValueError("--viewer-env-idx must be within --num-envs")
    env_cfg.viewer.env_idx = args.viewer_env_idx

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Task: {args.task}")
    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Seed: {args.seed}")
    print(f"[INFO] Video length: {args.length} frames")
    print(f"[INFO] Output directory: {output_dir}")

    raw_env = ManagerBasedRlEnv(
        cfg=env_cfg, device=device, render_mode="rgb_array"
    )

    env = VideoRecorder(
        raw_env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=args.length,
        disable_logger=True,
    )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(args.task) or OnPolicyRunner
    runner = runner_cls(env, prepare_agent_cfg(agent_cfg), device=device)
    runner.load(str(checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)

    goal_axes = ((0, 1.0), (1, 1.0), (2, 1.0), (0, -1.0), (1, -1.0), (2, -1.0))
    goal_index = 0
    pose_goal_available = has_pose_goal(raw_env)
    if args.goal_script != "none" and not pose_goal_available:
        raise ValueError(
            f"--goal-script {args.goal_script!r} requires a task with a "
            "'pose_goal' command."
        )
    if args.goal_script == "local_xyz":
        axis, sign = goal_axes[goal_index]
        set_relative_goal_from_current_cube(
            raw_env, axis, sign * math.radians(args.goal_step_deg)
        )
        print(
            f"[GOAL] frame 0: palm {'XYZ'[axis]} {sign * args.goal_step_deg:+.1f} deg; "
            f"initial error {current_pose_error_deg(raw_env):.2f} deg"
        )
        goal_index += 1
    elif pose_goal_available:
        print(
            f"[GOAL] fixed reset target; initial error "
            f"{current_pose_error_deg(raw_env):.2f} deg"
        )

    obs = env.get_observations()

    pickup_command = None
    recovery_requested = None
    replacement_pending = None
    cylinder_cfg = None
    if args.pickup_recovery == "external_replace":
        try:
            pickup_command = raw_env.command_manager.get_term("pickup_phase")
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "--pickup-recovery requires a Task 1 pickup_phase command"
            ) from exc
        recovery_requested = torch.zeros(
            raw_env.num_envs, device=raw_env.device, dtype=torch.bool
        )
        replacement_pending = torch.zeros_like(recovery_requested)
        cylinder_cfg = SceneEntityCfg("cube", geom_names=["cube_geom"])

    with torch.no_grad():
        for step in range(args.length + 10):
            if (
                args.goal_script == "local_xyz"
                and step > 0
                and step < args.length
                and step % args.goal_hold_frames == 0
            ):
                axis, sign = goal_axes[goal_index % len(goal_axes)]
                set_relative_goal_from_current_cube(
                    raw_env, axis, sign * math.radians(args.goal_step_deg)
                )
                obs = env.get_observations()
                print(
                    f"[GOAL] frame {step}: palm {'XYZ'[axis]} "
                    f"{sign * args.goal_step_deg:+.1f} deg; initial error "
                    f"{current_pose_error_deg(raw_env):.2f} deg"
                )
                goal_index += 1

            actions = policy(obs)
            result = env.step(actions)

            if isinstance(result, tuple):
                obs = result[0]
            else:
                obs = env.get_observations()

            if pickup_command is not None:
                assert recovery_requested is not None
                assert replacement_pending is not None
                assert cylinder_cfg is not None
                ready = (
                    (pickup_command.completed_pickups >= 1)
                    & ~recovery_requested
                    & (pickup_command.phase == MANIPULATE_PHASE)
                )
                ready_ids = ready.nonzero(as_tuple=False).squeeze(-1)
                if len(ready_ids) > 0:
                    mdp.reset_cylinder_on_table(
                        raw_env,
                        ready_ids,
                        nominal_spawn_xy=(
                            CYLINDER_SPAWN_XY[0] + 0.12,
                            CYLINDER_SPAWN_XY[1],
                        ),
                        xy_jitter=(0.0, 0.0),
                        yaw_range=(-3.14159, 3.14159),
                        table_height=TABLE_HEIGHT,
                        spawn_clearance=0.0,
                        scale_list=(1.0,),
                        scale_jitter=0.0,
                        cylinder_cfg=cylinder_cfg,
                    )
                    pickup_command.request_table_pickup(ready_ids)
                    recovery_requested[ready_ids] = True
                    replacement_pending[ready_ids] = True
                    if bool(ready[args.viewer_env_idx]):
                        print(
                            f"[PICKUP] frame {step}: cylinder removed; "
                            "waiting for open pose"
                        )
                    obs = env.get_observations()

                open_error = torch.max(
                    torch.abs(
                        raw_env.scene["robot"].data.joint_pos
                        - pickup_command.open_joint_target
                    ),
                    dim=-1,
                ).values
                opened = recovery_requested & (
                    open_error
                    <= pickup_command.cfg.open_joint_tolerance_rad
                )
                replace_ids = (opened & replacement_pending).nonzero(
                    as_tuple=False
                ).squeeze(-1)
                if len(replace_ids) > 0:
                    mdp.reset_cylinder_on_table(
                        raw_env,
                        replace_ids,
                        nominal_spawn_xy=CYLINDER_SPAWN_XY,
                        xy_jitter=(0.0, 0.0),
                        yaw_range=(-3.14159, 3.14159),
                        table_height=TABLE_HEIGHT,
                        spawn_clearance=0.0,
                        scale_list=(1.0,),
                        scale_jitter=0.0,
                        cylinder_cfg=cylinder_cfg,
                    )
                    replacement_pending[replace_ids] = False
                    viewer_replaced = bool(
                        (replace_ids == args.viewer_env_idx).any()
                    )
                    if viewer_replaced:
                        print(
                            f"[PICKUP] frame {step}: open pose verified; "
                            "cylinder returned to table"
                        )
                    obs = env.get_observations()

            if (step + 1) % args.print_every == 0:
                status = (
                    f"; goal error {current_pose_error_deg(raw_env):.2f} deg"
                    if pose_goal_available
                    else ""
                )
                print(
                    f"[INFO] Recorded/simulated step {step + 1}/{args.length}"
                    f"{status}"
                )

    env.close()
    print("[INFO] Done. Searching for video files:")
    for path in sorted(output_dir.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
