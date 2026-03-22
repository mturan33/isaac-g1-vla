"""
G1 Data Collection Environment
================================
Extends HierarchicalG1Env with TiledCamera for collecting
(image, state, action) tuples from RL expert policies.

Scene: G1 robot + PackingTable + steering wheel + TiledCamera on head
Camera: TiledCamera attached to head_link, looking forward-down
Data: Each step records (rgb, joint_state_22, action_23, gripper)

Usage (from C:\\IsaacLab):
    .\\isaaclab.bat -p <script.py> --enable_cameras --num_envs 4
"""

from __future__ import annotations

import math
import torch
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.math import quat_apply_inverse

import sys
import os

# Add hierarchical demo to path for policy wrappers
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DIRECT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))),
)
# Worktree path resolves to the 'direct' folder parent
# We need the actual direct folder in the main repo
_MAIN_DIRECT_DIR = os.path.join(
    "C:", os.sep, "IsaacLab", "source", "isaaclab_tasks",
    "isaaclab_tasks", "direct",
)
if _MAIN_DIRECT_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIRECT_DIR)


# ============================================================================
# Constants — must match hierarchical env EXACTLY
# ============================================================================

PHYSICS_DT = 1.0 / 200.0
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION  # 0.02s = 50 Hz

HEIGHT_DEFAULT = 0.80
GAIT_FREQUENCY = 1.5

LEG_ACTION_SCALE = 0.4
WAIST_ACTION_SCALE = 0.2

DEX3_USD_PATH = "C:/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0.usd"

# Joint names
LOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]  # 15

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]  # 14

RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]  # 7

HAND_JOINT_NAMES = [
    "left_hand_index_0_joint", "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_1_joint", "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_1_joint", "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]  # 14

# Default poses
DEFAULT_LOCO_POSES = {
    "left_hip_pitch_joint": -0.20, "right_hip_pitch_joint": -0.20,
    "left_hip_roll_joint": 0.0, "right_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.42, "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23, "right_ankle_pitch_joint": -0.23,
    "left_ankle_roll_joint": 0.0, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
}
DEFAULT_ARM_POSES = {
    "left_shoulder_pitch_joint": 0.35, "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 0.87,
    "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.35, "right_shoulder_roll_joint": -0.18,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 0.87,
    "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
DEFAULT_HAND_POSES = {name: 0.0 for name in HAND_JOINT_NAMES}
DEFAULT_ALL_POSES = {**DEFAULT_LOCO_POSES, **DEFAULT_ARM_POSES, **DEFAULT_HAND_POSES}

DEFAULT_LOCO_LIST = [DEFAULT_LOCO_POSES[j] for j in LOCO_JOINT_NAMES]
DEFAULT_ARM_LIST = [DEFAULT_ARM_POSES[j] for j in ARM_JOINT_NAMES]
DEFAULT_HAND_LIST = [DEFAULT_HAND_POSES[j] for j in HAND_JOINT_NAMES]

# IL joint order: loco (15) + right arm (7) = 22
IL_JOINT_NAMES = LOCO_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES


def quat_to_euler_xyz_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert wxyz quaternion to roll, pitch, yaw."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)


# ============================================================================
# Scene Configuration — hierarchical env + TiledCamera
# ============================================================================

@configclass
class DataCollectionSceneCfg(InteractiveSceneCfg):
    """Scene: G1 robot + table + object + camera.
    Actuator params match V6.2 training config EXACTLY."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=DEX3_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.80),
            joint_pos=DEFAULT_ALL_POSES,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.90,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint", ".*_hip_roll_joint",
                    ".*_hip_pitch_joint", ".*_knee_joint", ".*waist.*",
                ],
                effort_limit_sim={
                    ".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0,
                    ".*_hip_pitch_joint": 88.0, ".*_knee_joint": 139.0,
                    ".*waist_yaw_joint": 88.0, ".*waist_roll_joint": 88.0,
                    ".*waist_pitch_joint": 88.0,
                },
                velocity_limit_sim={
                    ".*_hip_yaw_joint": 32.0, ".*_hip_roll_joint": 20.0,
                    ".*_hip_pitch_joint": 32.0, ".*_knee_joint": 20.0,
                    ".*waist_yaw_joint": 32.0, ".*waist_roll_joint": 30.0,
                    ".*waist_pitch_joint": 30.0,
                },
                stiffness={
                    ".*_hip_yaw_joint": 150.0, ".*_hip_roll_joint": 150.0,
                    ".*_hip_pitch_joint": 200.0, ".*_knee_joint": 200.0,
                    ".*waist.*": 200.0,
                },
                damping={
                    ".*_hip_yaw_joint": 5.0, ".*_hip_roll_joint": 5.0,
                    ".*_hip_pitch_joint": 5.0, ".*_knee_joint": 5.0,
                    ".*waist.*": 10.0,
                },
                armature=0.01,
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit_sim={
                    ".*_ankle_pitch_joint": 35.0, ".*_ankle_roll_joint": 35.0,
                },
                velocity_limit_sim={
                    ".*_ankle_pitch_joint": 30.0, ".*_ankle_roll_joint": 30.0,
                },
                stiffness=20.0,
                damping=2.0,
                armature=0.01,
            ),
            "shoulders": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
                effort_limit_sim={
                    ".*_shoulder_pitch_joint": 25.0, ".*_shoulder_roll_joint": 25.0,
                },
                velocity_limit_sim={
                    ".*_shoulder_pitch_joint": 37.0, ".*_shoulder_roll_joint": 37.0,
                },
                stiffness=100.0,
                damping=2.0,
                armature=0.01,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint"],
                effort_limit_sim={
                    ".*_shoulder_yaw_joint": 25.0, ".*_elbow_joint": 25.0,
                },
                velocity_limit_sim={
                    ".*_shoulder_yaw_joint": 37.0, ".*_elbow_joint": 37.0,
                },
                stiffness=50.0,
                damping=2.0,
                armature=0.01,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_.*"],
                effort_limit_sim={
                    ".*_wrist_yaw_joint": 5.0, ".*_wrist_roll_joint": 25.0,
                    ".*_wrist_pitch_joint": 5.0,
                },
                velocity_limit_sim={
                    ".*_wrist_yaw_joint": 22.0, ".*_wrist_roll_joint": 37.0,
                    ".*_wrist_pitch_joint": 22.0,
                },
                stiffness=40.0,
                damping=2.0,
                armature=0.01,
            ),
            "hands": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_hand_index_.*_joint",
                    ".*_hand_middle_.*_joint",
                    ".*_hand_thumb_.*_joint",
                ],
                effort_limit=300,
                velocity_limit=100.0,
                stiffness={".*": 100.0},
                damping={".*": 10.0},
                armature={".*": 0.1},
            ),
        },
    )

    # PackingTable: 3m ahead
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="C:/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/unitree_sim_isaaclab/assets/objects/PackingTable/PackingTable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(3.0, 0.0, -0.3),
            rot=(0.7071, 0.0, 0.0, -0.7071),
        ),
    )

    # Steering wheel on table
    pickup_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=10.0,
                max_linear_velocity=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.85, -0.05, 0.72),
        ),
    )

    # TiledCamera on robot head — looking forward-down for manipulation
    # head_link is on the torso; camera looks toward the table
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_link/data_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.05),
            rot=(0.5, -0.5, 0.5, -0.5),  # looking forward-down
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.5,
            horizontal_aperture=4.8,
            clipping_range=(0.1, 10.0),
        ),
        width=640,
        height=480,
    )

    # Dome light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ============================================================================
# Data Collection Environment
# ============================================================================

class G1DataCollectionEnv:
    """
    G1 environment for expert data collection.

    Wraps the hierarchical control pipeline (loco + arm + finger policies)
    and adds a TiledCamera for capturing RGB images.

    Each step provides:
        - rgb: [N, H, W, 3] camera image (uint8)
        - joint_state: [N, 22] current joint positions (15 loco + 7 right arm)
        - action: [N, 23] applied action targets (22 joints + 1 gripper)
        - obs_dict: full observation dictionary

    Uses HierarchicalG1Env's policy wrappers for loco/arm control.
    """

    def __init__(
        self,
        sim: sim_utils.SimulationContext,
        scene_cfg: DataCollectionSceneCfg,
        loco_checkpoint: str,
        arm_checkpoint: str,
        num_envs: int = 4,
        device: str = "cuda:0",
    ):
        self.sim = sim
        self.device = device
        self.num_envs = num_envs
        self.decimation = DECIMATION
        self.physics_dt = PHYSICS_DT
        self.control_dt = CONTROL_DT
        self.step_count = 0

        # -- Create scene --
        scene_cfg.num_envs = num_envs
        scene_cfg.env_spacing = 8.0
        self.scene = InteractiveScene(scene_cfg)

        # -- Entity handles --
        self.robot: Articulation = self.scene["robot"]
        self.table: RigidObject = self.scene["table"]
        self.pickup_obj: RigidObject = self.scene["pickup_object"]
        self.camera: TiledCamera = self.scene["camera"]

        # -- Load policies from hierarchical demo --
        from high_low_hierarchical_g1.low_level.policy_wrapper import LocomotionPolicy
        from high_low_hierarchical_g1.low_level.arm_policy_wrapper import ArmPolicyWrapper
        from high_low_hierarchical_g1.low_level.finger_controller import FingerController
        from high_low_hierarchical_g1.low_level.arm_controller import ArmController

        self.loco_policy = LocomotionPolicy(
            checkpoint_path=loco_checkpoint, device=device,
        )
        self.arm_policy = ArmPolicyWrapper(
            checkpoint_path=arm_checkpoint, device=device,
        )
        self.finger_controller = FingerController(
            num_envs=num_envs, device=device,
        )
        self.arm_controller = ArmController(
            num_envs=num_envs, device=device,
        )

        # -- Joint index mapping (set after first reset) --
        self._loco_idx: Optional[torch.Tensor] = None
        self._arm_idx: Optional[torch.Tensor] = None
        self._hand_idx: Optional[torch.Tensor] = None
        self._right_arm_idx: Optional[torch.Tensor] = None
        self._il_joint_idx: Optional[torch.Tensor] = None  # 22 joints for IL
        self._arm_policy_joint_idx: Optional[torch.Tensor] = None
        self._palm_body_idx: Optional[int] = None
        self._indices_resolved = False

        # -- V6.2 state --
        self._default_loco = torch.tensor(DEFAULT_LOCO_LIST, dtype=torch.float32, device=device)
        self._default_arm = torch.tensor(DEFAULT_ARM_LIST, dtype=torch.float32, device=device)
        self._default_hand = torch.tensor(DEFAULT_HAND_LIST, dtype=torch.float32, device=device)

        leg_scales = [LEG_ACTION_SCALE] * 12
        waist_scales = [WAIST_ACTION_SCALE] * 3
        self._action_scales = torch.tensor(leg_scales + waist_scales, dtype=torch.float32, device=device)

        self._prev_act = torch.zeros(num_envs, 15, device=device)
        self._phase = torch.zeros(num_envs, device=device)
        self._height_cmd = torch.ones(num_envs, device=device) * HEIGHT_DEFAULT
        self._torso_cmd = torch.zeros(num_envs, 3, device=device)
        self._gravity_vec = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=device)

        # -- Arm policy state --
        self._arm_steps_since_spawn = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._arm_target_body = torch.zeros(num_envs, 3, device=device)
        self._arm_target_world = torch.zeros(num_envs, 3, device=device)
        self._arm_target_orient = torch.zeros(num_envs, 3, device=device)
        self._arm_target_orient[:, 2] = -1.0

        # -- Magnetic grasp --
        self._object_attached = False
        self._attach_offset_body = torch.zeros(num_envs, 3, device=device)
        self._attach_quat_offset = torch.zeros(num_envs, 4, device=device)
        self._attach_quat_offset[:, 0] = 1.0

        # -- Gripper state for recording (0=open, 1=closed) --
        self._gripper_state = torch.zeros(num_envs, 1, device=device)

        # -- Last applied action for recording --
        self._last_loco_targets = torch.zeros(num_envs, 15, device=device)
        self._last_right_arm_targets = torch.zeros(num_envs, 7, device=device)

        self._initial_pos: Optional[torch.Tensor] = None
        self._is_reset = False

        print(f"[G1DataCollectionEnv] {num_envs} envs, device={device}")
        print(f"[G1DataCollectionEnv] Camera: {scene_cfg.camera.width}x{scene_cfg.camera.height}")

    # ----------------------------------------------------------------- #
    # Joint index resolution
    # ----------------------------------------------------------------- #
    def _resolve_joint_indices(self):
        """Map joint names to robot articulation indices."""
        joint_names = self.robot.joint_names

        # Loco (15)
        loco_idx = []
        for name in LOCO_JOINT_NAMES:
            if name not in joint_names:
                raise RuntimeError(f"Loco joint '{name}' not found!")
            loco_idx.append(joint_names.index(name))
        self._loco_idx = torch.tensor(loco_idx, device=self.device, dtype=torch.long)

        # Full arm (14)
        arm_idx = []
        for name in ARM_JOINT_NAMES:
            if name not in joint_names:
                raise RuntimeError(f"Arm joint '{name}' not found!")
            arm_idx.append(joint_names.index(name))
        self._arm_idx = torch.tensor(arm_idx, device=self.device, dtype=torch.long)

        # Right arm only (7) — for arm policy
        right_arm_idx = []
        for name in RIGHT_ARM_JOINT_NAMES:
            right_arm_idx.append(joint_names.index(name))
        self._right_arm_idx = torch.tensor(right_arm_idx, device=self.device, dtype=torch.long)
        self._arm_policy_joint_idx = self._right_arm_idx  # alias

        # Hand (14)
        hand_idx = []
        for name in HAND_JOINT_NAMES:
            if name not in joint_names:
                raise RuntimeError(f"Hand joint '{name}' not found!")
            hand_idx.append(joint_names.index(name))
        self._hand_idx = torch.tensor(hand_idx, device=self.device, dtype=torch.long)

        # IL joints (22) = loco + right arm
        il_idx = loco_idx + right_arm_idx
        self._il_joint_idx = torch.tensor(il_idx, device=self.device, dtype=torch.long)

        # Palm body for EE computation
        body_names = self.robot.body_names
        self._palm_body_idx = None
        for search_term in ["right_hand_palm", "right_palm", "right_wrist_pitch"]:
            for i, name in enumerate(body_names):
                if search_term in name.lower():
                    self._palm_body_idx = i
                    break
            if self._palm_body_idx is not None:
                break
        if self._palm_body_idx is None:
            self._palm_body_idx = len(body_names) - 1

        self._indices_resolved = True
        print(f"[G1DataCollectionEnv] Joint indices resolved: "
              f"{len(loco_idx)} loco + {len(right_arm_idx)} right_arm + "
              f"{len(arm_idx)} full_arm + {len(hand_idx)} hand")
        print(f"[G1DataCollectionEnv] IL joint indices (22): OK")
        print(f"[G1DataCollectionEnv] Palm body: {body_names[self._palm_body_idx]} "
              f"(idx={self._palm_body_idx})")

    # ----------------------------------------------------------------- #
    # Reset
    # ----------------------------------------------------------------- #
    def reset(self) -> dict:
        """Reset environment, return initial observations."""
        if not self._is_reset:
            self.sim.reset()
            self._is_reset = True

        indices = torch.arange(self.num_envs, device=self.device)
        self.robot.reset(indices)
        self.table.reset(indices)
        self.pickup_obj.reset(indices)
        self._object_attached = False

        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.physics_dt)

        if not self._indices_resolved:
            self._resolve_joint_indices()

        # Reset V6.2 state
        self._prev_act.zero_()
        self._phase[:] = torch.rand(self.num_envs, device=self.device)
        self._height_cmd[:] = HEIGHT_DEFAULT
        self._torso_cmd.zero_()
        self.step_count = 0

        # Reset controllers
        self.finger_controller.reset()
        self.arm_controller.reset()

        # Reset arm policy
        self._arm_steps_since_spawn.zero_()
        self._arm_target_world.zero_()
        self._arm_target_body.zero_()
        self._arm_target_orient.zero_()
        self._arm_target_orient[:, 2] = -1.0

        # Reset gripper state
        self._gripper_state.zero_()

        # Store initial positions
        self._initial_pos = self.robot.data.root_pos_w[:, :2].clone()

        return self.get_obs()

    # ----------------------------------------------------------------- #
    # Step functions
    # ----------------------------------------------------------------- #
    def _build_loco_obs(self, velocity_command: torch.Tensor) -> torch.Tensor:
        """Build 66-dim V6.2 loco observation."""
        n = self.num_envs
        r = self.robot
        q = r.data.root_quat_w

        lin_vel_b = quat_apply_inverse(q, r.data.root_lin_vel_w)
        ang_vel_b = quat_apply_inverse(q, r.data.root_ang_vel_w)
        proj_gravity = quat_apply_inverse(q, self._gravity_vec.expand(n, -1))

        jp_all = r.data.joint_pos
        jv_all = r.data.joint_vel

        jp_leg = jp_all[:, self._loco_idx[:12]]
        jv_leg = jv_all[:, self._loco_idx[:12]] * 0.1
        jp_waist = jp_all[:, self._loco_idx[12:15]]
        jv_waist = jv_all[:, self._loco_idx[12:15]] * 0.1

        gait = torch.stack([
            torch.sin(2 * math.pi * self._phase),
            torch.cos(2 * math.pi * self._phase),
        ], dim=-1)

        torso_euler = quat_to_euler_xyz_wxyz(q)

        obs = torch.cat([
            lin_vel_b, ang_vel_b, proj_gravity,
            jp_leg, jv_leg, jp_waist, jv_waist,
            self._height_cmd[:, None], velocity_command, gait,
            self._prev_act, torso_euler, self._torso_cmd,
        ], dim=-1)

        return obs.clamp(-10, 10).nan_to_num()

    def _run_loco_policy(self, velocity_command: torch.Tensor) -> torch.Tensor:
        """Run loco policy, return [N, 15] absolute joint targets."""
        obs = self._build_loco_obs(velocity_command)

        with torch.inference_mode():
            raw_actions = self.loco_policy.get_raw_action(obs)

        targets = self._default_loco.unsqueeze(0) + raw_actions * self._action_scales.unsqueeze(0)

        # Clamps matching V6.2 training
        targets[:, 12].clamp_(-0.15, 0.15)  # waist_yaw
        targets[:, 13].clamp_(-0.15, 0.15)  # waist_roll
        targets[:, 14].clamp_(-0.2, 0.2)    # waist_pitch
        targets[:, 4].clamp_(-0.3, 0.3)     # left_hip_yaw
        targets[:, 5].clamp_(-0.3, 0.3)     # right_hip_yaw

        self._prev_act = raw_actions.clone()
        self._phase = (self._phase + GAIT_FREQUENCY * CONTROL_DT) % 1.0

        return targets

    def _compute_palm_ee(self):
        """Compute EE position and palm quaternion."""
        from high_low_hierarchical_g1.low_level.arm_policy_wrapper import (
            get_palm_forward, PALM_FORWARD_OFFSET,
        )
        palm_pos = self.robot.data.body_pos_w[:, self._palm_body_idx]
        palm_quat = self.robot.data.body_quat_w[:, self._palm_body_idx]
        palm_fwd = get_palm_forward(palm_quat)
        ee_pos = palm_pos + PALM_FORWARD_OFFSET * palm_fwd
        return ee_pos, palm_quat

    def _build_arm_obs(self) -> torch.Tensor:
        """Build 39-dim arm observation."""
        from high_low_hierarchical_g1.low_level.arm_policy_wrapper import (
            ArmPolicyWrapper, ARM_ACT_DIM,
        )

        r = self.robot
        root_pos = r.data.root_pos_w
        root_quat = r.data.root_quat_w

        arm_pos = r.data.joint_pos[:, self._arm_policy_joint_idx]
        arm_vel = r.data.joint_vel[:, self._arm_policy_joint_idx]

        ee_world, palm_quat = self._compute_palm_ee()
        ee_body = quat_apply_inverse(root_quat, ee_world - root_pos)

        prev_arm_act = self.arm_policy.prev_action
        if prev_arm_act is None:
            prev_arm_act = torch.zeros(self.num_envs, ARM_ACT_DIM, device=self.device)

        obs = ArmPolicyWrapper.build_obs(
            arm_pos=arm_pos,
            arm_vel=arm_vel,
            ee_body=ee_body,
            palm_quat=palm_quat,
            target_body=self._arm_target_body,
            prev_arm_act=prev_arm_act,
            steps_since_spawn=self._arm_steps_since_spawn,
            target_orient=self._arm_target_orient,
        )

        self._arm_steps_since_spawn += 1
        return obs

    def _get_arm_policy_targets(self) -> torch.Tensor:
        """Run arm policy, return [N, 14] full arm targets."""
        obs = self._build_arm_obs()
        right_7_targets = self.arm_policy.get_arm_targets(obs)

        arm_targets = self._default_arm.unsqueeze(0).expand(self.num_envs, -1).clone()
        arm_targets[:, 7:14] = right_7_targets

        return arm_targets, right_7_targets

    def _apply_targets(self, loco_targets, arm_targets, finger_targets):
        """Set all 43 joint position targets."""
        tgt = self.robot.data.default_joint_pos.clone()
        tgt[:, self._loco_idx] = loco_targets
        tgt[:, self._arm_idx] = arm_targets
        tgt[:, self._hand_idx] = finger_targets
        self.robot.set_joint_position_target(tgt)

    def _update_attached_object(self):
        """Teleport attached object to follow palm."""
        if not self._object_attached:
            return
        from isaaclab.utils.math import quat_apply, quat_mul
        ee_world, palm_quat = self._compute_palm_ee()
        obj_target = ee_world + quat_apply(palm_quat, self._attach_offset_body)
        obj_quat = quat_mul(palm_quat, self._attach_quat_offset)
        root_state = self.pickup_obj.data.default_root_state.clone()
        root_state[:, :3] = obj_target
        root_state[:, 3:7] = obj_quat
        root_state[:, 7:] = 0.0
        self.pickup_obj.write_root_state_to_sim(root_state)

    def step_walk(self, velocity_command: torch.Tensor) -> dict:
        """Step in walking mode (arms at default). Records action."""
        loco_targets = self._run_loco_policy(velocity_command)
        arm_targets = self._default_arm.unsqueeze(0).expand(self.num_envs, -1)
        finger_targets = self.finger_controller.get_targets()

        self._apply_targets(loco_targets, arm_targets, finger_targets)

        # Record action targets
        self._last_loco_targets = loco_targets.clone()
        # Right arm at default during walk
        from high_low_hierarchical_g1.low_level.arm_policy_wrapper import ARM_DEFAULT
        self._last_right_arm_targets = torch.tensor(
            ARM_DEFAULT, dtype=torch.float32, device=self.device,
        ).unsqueeze(0).expand(self.num_envs, -1)

        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self._update_attached_object()
            self.sim.step()

        self.scene.update(self.control_dt)
        self.step_count += 1
        return self.get_obs()

    def step_manipulation(self, velocity_command: torch.Tensor) -> dict:
        """Step with arm policy active. Records action."""
        loco_targets = self._run_loco_policy(velocity_command)
        arm_targets, right_7_targets = self._get_arm_policy_targets()
        finger_targets = self.finger_controller.get_targets()

        self._apply_targets(loco_targets, arm_targets, finger_targets)

        # Record action targets
        self._last_loco_targets = loco_targets.clone()
        self._last_right_arm_targets = right_7_targets.clone()

        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self._update_attached_object()
            self.sim.step()

        self.scene.update(self.control_dt)
        self.step_count += 1
        return self.get_obs()

    def step_hold(self, velocity_command: torch.Tensor, arm_targets_14: torch.Tensor) -> dict:
        """Step with fixed arm targets (hold position). Records action."""
        loco_targets = self._run_loco_policy(velocity_command)
        finger_targets = self.finger_controller.get_targets()

        self._apply_targets(loco_targets, arm_targets_14, finger_targets)

        self._last_loco_targets = loco_targets.clone()
        self._last_right_arm_targets = arm_targets_14[:, 7:14].clone()

        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self._update_attached_object()
            self.sim.step()

        self.scene.update(self.control_dt)
        self.step_count += 1
        return self.get_obs()

    # ----------------------------------------------------------------- #
    # Arm target setting
    # ----------------------------------------------------------------- #
    def set_arm_target_world(self, target_world: torch.Tensor):
        """Set arm reach target in world frame (freeze body-frame target)."""
        if target_world.ndim == 1:
            target_world = target_world.unsqueeze(0).expand(self.num_envs, -1)
        self._arm_target_world = target_world.clone()
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        self._arm_target_body = quat_apply_inverse(root_quat, target_world - root_pos)

    def reset_arm_policy_state(self):
        """Reset arm policy internal state."""
        current_arm_7 = self.robot.data.joint_pos[:, self._arm_policy_joint_idx]
        self.arm_policy.reset_state(current_targets=current_arm_7)
        self._arm_steps_since_spawn.zero_()

    # ----------------------------------------------------------------- #
    # Magnetic grasp
    # ----------------------------------------------------------------- #
    def attach_object_to_hand(self, max_dist: float = 0.22) -> bool:
        """Attach object to palm if within distance."""
        ee_world, palm_quat = self._compute_palm_ee()
        obj_pos = self.pickup_obj.data.root_pos_w
        dist = (ee_world - obj_pos).norm(dim=-1).mean().item()

        if dist < max_dist:
            from isaaclab.utils.math import quat_mul, quat_conjugate
            diff_world = obj_pos - ee_world
            self._attach_offset_body = quat_apply_inverse(palm_quat, diff_world)
            obj_quat = self.pickup_obj.data.root_quat_w
            self._attach_quat_offset = quat_mul(quat_conjugate(palm_quat), obj_quat)
            self._object_attached = True
            self._gripper_state[:] = 1.0
            print(f"  [MagneticGrasp] Attached! dist={dist:.3f}m")
            return True
        else:
            print(f"  [MagneticGrasp] Too far: {dist:.3f}m")
            return False

    def detach_object(self):
        """Release attached object."""
        if self._object_attached:
            self._object_attached = False
            self._gripper_state[:] = 0.0

    # ----------------------------------------------------------------- #
    # Data access
    # ----------------------------------------------------------------- #
    def get_camera_rgb(self) -> torch.Tensor:
        """Get current RGB frame. Shape: [N, H, W, 3] uint8."""
        return self.camera.data.output["rgb"][..., :3]

    def get_joint_state(self) -> torch.Tensor:
        """Get current IL joint positions [N, 22] (15 loco + 7 right arm)."""
        return self.robot.data.joint_pos[:, self._il_joint_idx]

    def get_action(self) -> torch.Tensor:
        """Get last applied action [N, 23] (22 joint targets + 1 gripper)."""
        joint_targets = torch.cat([
            self._last_loco_targets,       # 15
            self._last_right_arm_targets,  # 7
        ], dim=-1)  # 22
        return torch.cat([joint_targets, self._gripper_state], dim=-1)  # 23

    def get_obs(self) -> dict:
        """Get full observation dictionary."""
        jp = self.robot.data.joint_pos
        jv = self.robot.data.joint_vel
        q = self.robot.data.root_quat_w

        return {
            "root_pos": self.robot.data.root_pos_w,
            "root_quat": q,
            "base_height": self.robot.data.root_pos_w[:, 2],
            "base_lin_vel": quat_apply_inverse(q, self.robot.data.root_lin_vel_w),
            "base_ang_vel": quat_apply_inverse(q, self.robot.data.root_ang_vel_w),
            "projected_gravity": quat_apply_inverse(
                q, self._gravity_vec.expand(self.num_envs, -1)),
            "joint_pos_loco": jp[:, self._loco_idx],
            "joint_vel_loco": jv[:, self._loco_idx],
            "joint_pos_arm": jp[:, self._arm_idx],
            "joint_vel_arm": jv[:, self._arm_idx],
            "joint_pos_finger": jp[:, self._hand_idx],
            "joint_vel_finger": jv[:, self._hand_idx],
        }

    @property
    def initial_positions(self) -> torch.Tensor:
        """Initial XY positions [N, 2]."""
        return self._initial_pos

    def close(self):
        """Clean up."""
        pass
