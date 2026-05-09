#!/usr/bin/env python3

"""
View the current G0 default pose defined in g0.py.

This script only creates:
    - ground
    - light
    - G0 robot from G0_CFG

It does not create G0-Velocity-v0.
It does not load policy, reward, command, observation, or termination logic.

Usage:

1. Frozen pose checking:
    /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/view_g0_pose.py

2. Free physics with PD targets set to G0_DEFAULT_JOINT_POS:
    /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/view_g0_pose.py --free_fall
"""

from __future__ import annotations

import argparse
import inspect

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="View G0 default pose from g0.py.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--free_fall",
    action="store_true",
    default=False,
    help="Set default pose once, then let physics run while PD targets stay at default pose.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

import g0_robot_lab.assets.robots.g0.g0 as g0_module
from g0_robot_lab.assets.robots.g0.g0 import G0_CFG


@configclass
class G0PoseSceneCfg(InteractiveSceneCfg):
    """Minimal scene for viewing G0 default pose."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(20.0, 20.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=800.0,
        ),
    )

    robot: ArticulationCfg = G0_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


def apply_default_state_and_target(scene: InteractiveScene) -> None:
    """Set both joint state and PD target to the default pose from G0_CFG."""

    robot = scene["robot"]

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)

    # 1. Set actual simulated state.
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # 2. Set actuator / PD drive target.
    # Without this line, implicit actuators may pull joints back to zero.
    robot.set_joint_position_target(joint_pos)

    scene.write_data_to_sim()


def apply_default_target_only(scene: InteractiveScene) -> None:
    """Keep PD target at the default pose without forcibly resetting joint state."""

    robot = scene["robot"]
    joint_pos = robot.data.default_joint_pos.clone()

    # Keep drive target equal to G0_DEFAULT_JOINT_POS.
    # This allows gravity and contacts to act, but prevents drives from targeting zero.
    robot.set_joint_position_target(joint_pos)

    scene.write_data_to_sim()


def print_debug_info(scene: InteractiveScene) -> None:
    """Print useful debug information."""

    robot = scene["robot"]

    print("=" * 80)
    print("[INFO] G0 pose viewer started.")
    print("[INFO] Imported g0.py from:")
    print(f"       {inspect.getfile(g0_module)}")
    print("[INFO] G0_CFG init_state.pos:")
    print(f"       {G0_CFG.init_state.pos}")
    print("[INFO] G0_CFG init_state.joint_pos:")
    for key, value in G0_CFG.init_state.joint_pos.items():
        print(f"       {key}: {value}")

    print("[INFO] Resolved joint order inside Isaac Lab:")
    for i, name in enumerate(robot.data.joint_names):
        default_pos = robot.data.default_joint_pos[0, i].item()
        print(f"       [{i:02d}] {name:30s} default_pos = {default_pos:+.4f}")

    print("[INFO] Mode:")
    if args_cli.free_fall:
        print("       free_fall=True")
        print("       State is set once. PD target is kept at default pose.")
    else:
        print("       frozen pose mode")
        print("       State and PD target are both reset every frame.")
    print("=" * 80)


def main() -> None:
    """Launch minimal simulation scene."""

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=1,
        device=args_cli.device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=(3.0, 3.0, 1.8),
        target=(0.0, 0.0, 0.6),
    )

    scene_cfg = G0PoseSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    apply_default_state_and_target(scene)
    print_debug_info(scene)

    sim_dt = sim.get_physics_dt()

    while simulation_app.is_running():
        if args_cli.free_fall:
            # Do not overwrite the robot state.
            # Only keep the actuator target at the default pose.
            apply_default_target_only(scene)
        else:
            # Fully freeze the robot at the default pose.
            apply_default_state_and_target(scene)

        sim.step()
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
