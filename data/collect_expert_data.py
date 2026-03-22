#!/usr/bin/env python3
"""
Expert Data Collection — G1 Pick-and-Place
=============================================
Runs the RL expert policies (loco + arm) through the full pick-and-place
pipeline and records (image, state, action) at each step.

Output: Raw episode data saved as .pt files (one per episode).
Each file contains lists of dicts with keys: image, state, action, timestamp.

Later: convert_to_lerobot.py will transform these into LeRobot v2 format.

Usage (from C:\\IsaacLab):
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\isaac-g1-vla\\.claude\\worktrees\\nifty-mcclintock\\data\\collect_expert_data.py --enable_cameras --num_envs 4

    # Smoke test (1 env, 1 episode):
    .\\isaaclab.bat -p <path>/collect_expert_data.py --enable_cameras --num_envs 1 --num_episodes 1

    # Full collection (64 envs, 200 episodes):
    .\\isaaclab.bat -p <path>/collect_expert_data.py --enable_cameras --num_envs 64 --num_episodes 200
"""

# ============================================================================
# AppLauncher MUST be created before any Isaac Lab imports
# ============================================================================
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect expert data from G1 RL policies")
parser.add_argument("--num_envs", type=int, default=4, help="Parallel environments")
parser.add_argument("--num_episodes", type=int, default=1, help="Episodes to collect")
parser.add_argument(
    "--loco_checkpoint", type=str,
    default="C:/IsaacLab/logs/ulc/g1_stage2_loco_2026-03-18_17-40-54/model_29500.pt",
    help="Loco policy checkpoint",
)
parser.add_argument(
    "--arm_checkpoint", type=str,
    default="C:/IsaacLab/logs/ulc/g1_stage2_arm_2026-03-06_18-51-31/model_best.pt",
    help="Arm policy checkpoint",
)
parser.add_argument("--output_dir", type=str, default="data/raw_episodes", help="Output directory")
parser.add_argument("--walk_distance", type=float, default=3.0, help="Walk distance (m)")
parser.add_argument("--smoke_test", action="store_true", help="Quick smoke test (print shapes only)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# Post-AppLauncher imports
# ============================================================================
import os
import sys
import time
import torch

sys.stdout.reconfigure(line_buffering=True)

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply_inverse, quat_apply

# Add paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_SCRIPT_DIR)
_DIRECT_DIR = os.path.join(
    "C:", os.sep, "IsaacLab", "source", "isaaclab_tasks",
    "isaaclab_tasks", "direct",
)
for p in [_PKG_DIR, _DIRECT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from envs.g1_data_collection_env import G1DataCollectionEnv, DataCollectionSceneCfg, PHYSICS_DT
from high_low_hierarchical_g1.skills.walk_to import WalkToSkill
from high_low_hierarchical_g1.config.skill_config import WalkToConfig
from high_low_hierarchical_g1.low_level.arm_policy_wrapper import ARM_DEFAULT, SHOULDER_OFFSET


class EpisodeRecorder:
    """Records (image, state, action) tuples during an episode."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.steps = []  # list of dicts per step
        self._recording = False

    def start(self):
        """Start recording a new episode."""
        self.steps = []
        self._recording = True

    def stop(self):
        """Stop recording."""
        self._recording = False

    def record(self, env: G1DataCollectionEnv):
        """Record one step from the environment."""
        if not self._recording:
            return

        # Get data from env
        rgb = env.get_camera_rgb()           # [N, H, W, 3] uint8
        state = env.get_joint_state()        # [N, 22]
        action = env.get_action()            # [N, 23]

        self.steps.append({
            "image": rgb.cpu(),              # Move to CPU for storage
            "state": state.cpu().clone(),
            "action": action.cpu().clone(),
            "timestamp": env.step_count * env.control_dt,
        })

    def save(self, output_dir: str, episode_idx: int):
        """Save recorded episode to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Save per-env episodes
        for env_i in range(self.num_envs):
            ep_data = {
                "images": torch.stack([s["image"][env_i] for s in self.steps]),  # [T, H, W, 3]
                "states": torch.stack([s["state"][env_i] for s in self.steps]),  # [T, 22]
                "actions": torch.stack([s["action"][env_i] for s in self.steps]),  # [T, 23]
                "timestamps": torch.tensor([s["timestamp"] for s in self.steps]),  # [T]
                "num_steps": len(self.steps),
                "control_dt": PHYSICS_DT * 4,  # 0.02s
            }

            global_idx = episode_idx * self.num_envs + env_i
            filepath = os.path.join(output_dir, f"episode_{global_idx:06d}.pt")
            torch.save(ep_data, filepath)

        return len(self.steps)


def run_pick_and_place_episode(
    env: G1DataCollectionEnv,
    recorder: EpisodeRecorder,
    walk_distance: float = 3.0,
    smoke_test: bool = False,
):
    """
    Run one full pick-and-place episode and record all steps.

    Pipeline (matches test_hierarchical.py):
      1. Walk to table
      2. Stabilize
      3. Reach with arm policy
      4. Hold position
      5. Close fingers (grasp)
      6. Attach object magnetically
      7. Carry
      8. Open fingers / return

    Returns:
        success: bool — whether episode completed successfully
    """
    num_envs = env.num_envs
    device = env.device

    obs = env.reset()
    recorder.start()

    stand_cmd = torch.zeros(num_envs, 3, device=device)

    # ================================================================
    # Phase 1: Walk to table
    # ================================================================
    initial_pos = env.initial_positions
    target_pos = initial_pos.clone()
    target_pos[:, 0] += walk_distance

    walk_cfg = WalkToConfig()
    walk_cfg.max_steps = 1500
    skill = WalkToSkill(config=walk_cfg, device=device)
    skill.reset(target_positions=target_pos)

    print(f"  Phase 1: Walk to table ({walk_distance}m)")
    walk_done = False
    while simulation_app.is_running() and not walk_done:
        vel_cmd, walk_done, result = skill.step(obs)
        obs = env.step_walk(vel_cmd)
        recorder.record(env)

        if env.step_count % 100 == 0:
            dist = torch.norm(obs["root_pos"][:, :2] - target_pos, dim=-1).mean()
            print(f"    Step {env.step_count}: dist={dist:.2f}m, h={obs['base_height'].mean():.2f}m")

        if smoke_test and env.step_count > 50:
            print(f"  [Smoke test] Walk truncated at step {env.step_count}")
            break

    # ================================================================
    # Phase 2: Stabilize
    # ================================================================
    print(f"  Phase 2: Stabilize")
    for _ in range(100):
        if not simulation_app.is_running():
            break
        obs = env.step_walk(stand_cmd)
        recorder.record(env)

    # ================================================================
    # Phase 3: Reach with arm policy
    # ================================================================
    print(f"  Phase 3: Reach")

    # Compute reachable target
    shoulder_offset = torch.tensor(SHOULDER_OFFSET, device=device)
    max_reach = 0.35

    cup_pos = env.pickup_obj.data.root_pos_w.clone()
    root_pos = env.robot.data.root_pos_w
    root_quat = env.robot.data.root_quat_w

    cup_body = quat_apply_inverse(root_quat, cup_pos - root_pos)
    cup_from_shoulder = cup_body - shoulder_offset.unsqueeze(0)
    dist_from_shoulder = cup_from_shoulder.norm(dim=-1, keepdim=True)

    scale = torch.clamp(max_reach / (dist_from_shoulder + 1e-6), max=1.0)
    reachable_body = shoulder_offset.unsqueeze(0) + cup_from_shoulder * scale
    reachable_world = quat_apply(root_quat, reachable_body) + root_pos

    env.set_arm_target_world(reachable_world)
    env.reset_arm_policy_state()

    lean_cmd = torch.zeros(num_envs, 3, device=device)
    lean_cmd[:, 0] = 0.15

    reach_steps = 80 if smoke_test else 120
    for step in range(reach_steps):
        if not simulation_app.is_running():
            break
        cmd = lean_cmd if step < 80 else stand_cmd
        obs = env.step_manipulation(cmd)
        recorder.record(env)

    # Phase 3b: Hold position
    print(f"  Phase 3b: Hold position")
    hold_arm_targets = env.robot.data.joint_pos[:, env._arm_idx].clone()
    hold_steps = 40 if smoke_test else 80
    for _ in range(hold_steps):
        if not simulation_app.is_running():
            break
        obs = env.step_hold(stand_cmd, hold_arm_targets)
        recorder.record(env)

    # ================================================================
    # Phase 4: Close fingers + magnetic grasp
    # ================================================================
    print(f"  Phase 4: Grasp")
    env.finger_controller.close(hand="both")

    for step in range(100):
        if not simulation_app.is_running():
            break
        obs = env.step_hold(stand_cmd, hold_arm_targets)
        recorder.record(env)

        # Try magnetic attach at step 50 (fingers partially closed)
        if step == 50:
            env.attach_object_to_hand(max_dist=0.25)

    env._gripper_state[:] = 1.0  # Record gripper as closed

    # ================================================================
    # Phase 5: Carry
    # ================================================================
    print(f"  Phase 5: Carry")
    carry_steps = 100 if smoke_test else 200
    for _ in range(carry_steps):
        if not simulation_app.is_running():
            break
        obs = env.step_hold(stand_cmd, hold_arm_targets)
        recorder.record(env)

    # ================================================================
    # Phase 6: Return + open fingers
    # ================================================================
    print(f"  Phase 6: Return + release")
    from high_low_hierarchical_g1.low_level.arm_controller import ArmPose
    env.arm_controller.set_pose(ArmPose.DEFAULT)
    env.finger_controller.open(hand="both")
    env.detach_object()
    env._gripper_state[:] = 0.0

    return_steps = 100 if smoke_test else 200
    for _ in range(return_steps):
        if not simulation_app.is_running():
            break
        arm_targets = env.arm_controller.get_targets()
        obs = env.step_hold(stand_cmd, arm_targets)
        recorder.record(env)

    recorder.stop()

    h = obs["base_height"].mean().item()
    standing = (obs["base_height"] > 0.3).sum().item()
    success = standing > 0 and env._object_attached is False
    print(f"  Episode done: {len(recorder.steps)} steps, "
          f"h={h:.2f}m, standing={standing}/{num_envs}")

    return success


def main():
    num_envs = args_cli.num_envs
    device = "cuda:0"

    print("=" * 60)
    print("  G1 Expert Data Collection")
    print("=" * 60)
    print(f"  Environments  : {num_envs}")
    print(f"  Episodes      : {args_cli.num_episodes}")
    print(f"  Loco ckpt     : {args_cli.loco_checkpoint}")
    print(f"  Arm ckpt      : {args_cli.arm_checkpoint}")
    print(f"  Output        : {args_cli.output_dir}")
    print(f"  Smoke test    : {args_cli.smoke_test}")
    print("=" * 60)

    # Create sim
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[8.0, -4.0, 3.5], target=[3.0, 0.0, 0.5])

    # Create env
    scene_cfg = DataCollectionSceneCfg()
    env = G1DataCollectionEnv(
        sim=sim,
        scene_cfg=scene_cfg,
        loco_checkpoint=args_cli.loco_checkpoint,
        arm_checkpoint=args_cli.arm_checkpoint,
        num_envs=num_envs,
        device=device,
    )

    # Recorder
    recorder = EpisodeRecorder(num_envs=num_envs, device=device)

    # Smoke test: just print shapes
    if args_cli.smoke_test:
        print("\n[Smoke Test] Running 1 episode with shape checks...")
        obs = env.reset()

        # Camera warmup — needs ~20 steps to start rendering
        stand_cmd = torch.zeros(num_envs, 3, device=device)
        print("[Smoke Test] Camera warmup (20 steps)...")
        for i in range(20):
            obs = env.step_walk(stand_cmd)

        # Check shapes
        rgb = env.get_camera_rgb()
        state = env.get_joint_state()
        action = env.get_action()

        camera_working = rgb.float().mean() > 0.1  # check mean pixel value
        print(f"\n[Smoke Test] Data shapes:")
        print(f"  RGB:    {rgb.shape} {rgb.dtype}")      # [N, 480, 640, 3] uint8
        print(f"  State:  {state.shape} {state.dtype}")   # [N, 22] float32
        print(f"  Action: {action.shape} {action.dtype}") # [N, 23] float32
        print(f"  Height: {obs['base_height'].mean():.2f}m")
        print(f"\n[Smoke Test] Camera working: {camera_working}")
        print(f"[Smoke Test] RGB mean pixel: {rgb.float().mean():.1f}")
        print(f"[Smoke Test] RGB max pixel: {rgb.max().item()}")
        print(f"[Smoke Test] State range: [{state.min():.3f}, {state.max():.3f}]")
        print(f"[Smoke Test] Action range: [{action.min():.3f}, {action.max():.3f}]")

        # Run truncated episode
        print(f"\n[Smoke Test] Running truncated pick-and-place...")
        success = run_pick_and_place_episode(
            env, recorder, args_cli.walk_distance, smoke_test=True,
        )

        steps_collected = len(recorder.steps)
        print(f"\n[Smoke Test] Collected {steps_collected} steps")
        if steps_collected > 0:
            print(f"  First step image shape: {recorder.steps[0]['image'].shape}")
            print(f"  First step state shape: {recorder.steps[0]['state'].shape}")
            print(f"  First step action shape: {recorder.steps[0]['action'].shape}")

            # Save test episode
            os.makedirs(args_cli.output_dir, exist_ok=True)
            recorder.save(args_cli.output_dir, episode_idx=0)
            print(f"\n[Smoke Test] Saved to {args_cli.output_dir}/")

        print(f"\n[Smoke Test] PASSED!")
        env.close()
        try:
            simulation_app.close()
        except Exception:
            pass  # Isaac Sim shutdown crash is harmless
        return

    # Full data collection
    total_episodes_saved = 0
    start_time = time.time()

    for ep_idx in range(args_cli.num_episodes):
        if not simulation_app.is_running():
            break

        print(f"\n{'='*50}")
        print(f"  Episode {ep_idx + 1}/{args_cli.num_episodes} "
              f"(saving {num_envs} parallel episodes)")
        print(f"{'='*50}")

        success = run_pick_and_place_episode(
            env, recorder, args_cli.walk_distance, smoke_test=False,
        )

        num_steps = recorder.save(args_cli.output_dir, episode_idx=ep_idx)
        total_episodes_saved += num_envs

        elapsed = time.time() - start_time
        eps_per_sec = total_episodes_saved / elapsed if elapsed > 0 else 0
        print(f"  Saved {num_envs} episodes ({num_steps} steps each)")
        print(f"  Total: {total_episodes_saved} episodes, "
              f"{elapsed:.0f}s elapsed, {eps_per_sec:.1f} ep/s")

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total episodes : {total_episodes_saved}")
    print(f"  Output dir     : {args_cli.output_dir}")
    print(f"  Time elapsed   : {elapsed:.0f}s")
    print(f"{'='*60}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
