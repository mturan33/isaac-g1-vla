"""
Data utilities for G1 VLA project.
Helper functions for loading, validating, and visualizing collected data.
"""

import os
import torch
from typing import List, Dict, Optional


def load_episode(filepath: str) -> Dict[str, torch.Tensor]:
    """
    Load a saved episode .pt file.

    Returns dict with keys:
        images: [T, H, W, 3] uint8
        states: [T, 22] float32
        actions: [T, 23] float32
        timestamps: [T] float32
        num_steps: int
        control_dt: float
    """
    return torch.load(filepath, weights_only=False)


def validate_episode(ep_data: Dict[str, torch.Tensor]) -> bool:
    """Check that an episode has valid shapes and ranges."""
    T = ep_data["num_steps"]

    checks = {
        "images shape": ep_data["images"].shape == (T, 480, 640, 3),
        "states shape": ep_data["states"].shape == (T, 22),
        "actions shape": ep_data["actions"].shape == (T, 23),
        "timestamps shape": ep_data["timestamps"].shape == (T,),
        "images dtype": ep_data["images"].dtype == torch.uint8,
        "states finite": ep_data["states"].isfinite().all(),
        "actions finite": ep_data["actions"].isfinite().all(),
        "gripper binary": ep_data["actions"][:, -1].unique().numel() <= 2,
    }

    all_ok = all(checks.values())
    if not all_ok:
        for name, ok in checks.items():
            if not ok:
                print(f"  [FAIL] {name}")
    return all_ok


def count_episodes(output_dir: str) -> int:
    """Count .pt files in output directory."""
    if not os.path.exists(output_dir):
        return 0
    return len([f for f in os.listdir(output_dir) if f.endswith(".pt")])


def print_episode_stats(ep_data: Dict[str, torch.Tensor]):
    """Print summary statistics of an episode."""
    T = ep_data["num_steps"]
    dt = ep_data["control_dt"]
    duration = T * dt

    states = ep_data["states"]
    actions = ep_data["actions"]
    gripper = actions[:, -1]
    joint_actions = actions[:, :22]

    print(f"  Steps: {T}, Duration: {duration:.1f}s")
    print(f"  State range: [{states.min():.3f}, {states.max():.3f}]")
    print(f"  Action range: [{joint_actions.min():.3f}, {joint_actions.max():.3f}]")
    print(f"  Gripper transitions: {(gripper[1:] != gripper[:-1]).sum().item()}")
    print(f"  Image mean: {ep_data['images'].float().mean():.1f}")
