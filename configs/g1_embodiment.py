"""
G1 Embodiment Configuration
============================
Joint mapping, action/observation space definitions, and constants
for the Unitree G1 robot used in VLA/IL data collection.

Action space for IL: 22 joint position targets (15 loco + 7 right arm) + 1 gripper binary = 23
Observation state for IL: 22 joint positions (current)
"""

# ============================================================================
# JOINT NAMES — matching hierarchical demo exactly
# ============================================================================

# Locomotion joints (15): 12 legs + 3 waist
LOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]  # 15

# Right arm joints (7) — only right arm is active in manipulation
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]  # 7

# IL action/state joint order: loco (15) + right arm (7) = 22
IL_JOINT_NAMES = LOCO_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES  # 22

# Full arm joint names (both arms, 14 total) — for env compatibility
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

# ============================================================================
# DIMENSIONS
# ============================================================================

NUM_LOCO_JOINTS = 15
NUM_RIGHT_ARM_JOINTS = 7
NUM_IL_JOINTS = NUM_LOCO_JOINTS + NUM_RIGHT_ARM_JOINTS  # 22
NUM_IL_ACTIONS = NUM_IL_JOINTS + 1  # 23 (22 joints + 1 gripper binary)

# ============================================================================
# PHYSICS — must match hierarchical demo exactly
# ============================================================================

PHYSICS_DT = 1.0 / 200.0   # 200 Hz
DECIMATION = 4
CONTROL_DT = PHYSICS_DT * DECIMATION  # 0.02s = 50 Hz
CONTROL_FREQ = 50  # Hz

# ============================================================================
# POLICY CONSTANTS — from hierarchical env
# ============================================================================

LEG_ACTION_SCALE = 0.4   # radians
WAIST_ACTION_SCALE = 0.2  # radians
ARM_ACTION_SCALE = 2.0
ARM_ACTION_CLAMP = 1.5

HEIGHT_DEFAULT = 0.80
GAIT_FREQUENCY = 1.5  # Hz

# ============================================================================
# DEFAULT POSES — from hierarchical env
# ============================================================================

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

# As ordered list for IL state
DEFAULT_IL_JOINT_LIST = [DEFAULT_LOCO_POSES.get(j, DEFAULT_ARM_POSES.get(j, 0.0)) for j in IL_JOINT_NAMES]

# ============================================================================
# CAMERA
# ============================================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_RESIZE = 224  # For IL policy input (224x224)

# ============================================================================
# DATA COLLECTION
# ============================================================================

TASK_DESCRIPTION = "pick up the steering wheel and place it in the basket"
