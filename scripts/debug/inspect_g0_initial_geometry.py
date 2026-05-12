#!/usr/bin/env python3
"""Inspect G0 initial geometry, mass properties, COM, and foot contact.

This script is debug-only. It does not train, does not modify the asset, and does
not change actuator hardware constants.
"""

from __future__ import annotations

import argparse
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect G0 reset geometry and mass/contact properties.")
parser.add_argument("--task", type=str, default="G0-Velocity-v0")
parser.add_argument("--root_z", type=float, default=0.23)
parser.add_argument("--hip", type=float, default=0.20)
parser.add_argument("--knee", type=float, default=0.34)
parser.add_argument("--ankle", type=float, default=0.14)
parser.add_argument("--settle-steps", type=int, default=1, help="Small number of zero-action steps to refresh contacts.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


ASSET_ROOT = Path("source/g0_robot_lab/g0_robot_lab/assets/robots/g0")
URDF_PATH = ASSET_ROOT / "urdf" / "g0.urdf"
FOOT_NAMES = ("l_foot_link", "r_foot_link")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _fmt_vec(values: Any, precision: int = 5) -> str:
    if values is None:
        return "n/a"
    return "(" + ", ".join(f"{float(value):.{precision}f}" for value in values) + ")"


def _quat_wxyz_to_euler_deg(quat: torch.Tensor) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat.detach().cpu().tolist()]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _candidate_joint_pos() -> dict[str, float]:
    return {
        "l_hip_pitch_joint": -args_cli.hip,
        "l_hip_roll_joint": 0.0,
        "l_hip_yaw_joint": 0.0,
        "l_knee_pitch_joint": -args_cli.knee,
        "l_ankle_pitch_joint": args_cli.ankle,
        "l_ankle_roll_joint": 0.0,
        "r_hip_pitch_joint": args_cli.hip,
        "r_hip_roll_joint": 0.0,
        "r_hip_yaw_joint": 0.0,
        "r_knee_pitch_joint": args_cli.knee,
        "r_ankle_pitch_joint": -args_cli.ankle,
        "r_ankle_roll_joint": 0.0,
    }


def _make_env_cfg():
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

    env_cfg = G0RobotLabEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.root_z)
    env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.joint_pos.update(_candidate_joint_pos())
    env_cfg.scene.robot.init_state.joint_vel = {".*": 0.0}
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device

    env_cfg.events.physics_material = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base = None
    env_cfg.events.reset_robot_joints = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None

    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.terminations.base_height = None
    env_cfg.terminations.bad_orientation = None
    return env_cfg


def _load_urdf_links() -> dict[str, dict[str, Any]]:
    root = ET.parse(URDF_PATH).getroot()
    links: dict[str, dict[str, Any]] = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        item: dict[str, Any] = {"mass": None, "com": None, "inertia_diag": None, "collision_mesh": None}
        inertial = link.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            if origin is not None:
                item["com"] = tuple(float(v) for v in origin.attrib.get("xyz", "0 0 0").split())
            if mass is not None:
                item["mass"] = float(mass.attrib["value"])
            if inertia is not None:
                item["inertia_diag"] = (
                    float(inertia.attrib["ixx"]),
                    float(inertia.attrib["iyy"]),
                    float(inertia.attrib["izz"]),
                )
        collision = link.find("collision")
        if collision is not None:
            mesh = collision.find("./geometry/mesh")
            origin = collision.find("origin")
            if mesh is not None:
                item["collision_mesh"] = {
                    "filename": mesh.attrib.get("filename"),
                    "origin_xyz": tuple(float(v) for v in origin.attrib.get("xyz", "0 0 0").split())
                    if origin is not None
                    else (0.0, 0.0, 0.0),
                    "origin_rpy": tuple(float(v) for v in origin.attrib.get("rpy", "0 0 0").split())
                    if origin is not None
                    else (0.0, 0.0, 0.0),
                }
        links[name] = item
    return links


def _read_stl_vertices(path: Path) -> torch.Tensor | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    vertices: list[tuple[float, float, float]] = []
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + tri_count * 50
        if expected_size == len(data):
            offset = 84
            for _ in range(tri_count):
                offset += 12
                for _ in range(3):
                    vertices.append(struct.unpack_from("<fff", data, offset))
                    offset += 12
                offset += 2
            return torch.tensor(vertices, dtype=torch.float32) if vertices else None
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return None
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return torch.tensor(vertices, dtype=torch.float32) if vertices else None


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[1:].expand_as(vec)
    t = torch.cross(xyz, vec, dim=-1) * 2.0
    return vec + quat[0] * t + torch.cross(xyz, t, dim=-1)


def _mesh_path(mesh_filename: str) -> Path:
    if mesh_filename.startswith("../"):
        return (URDF_PATH.parent / mesh_filename).resolve()
    return (URDF_PATH.parent / mesh_filename).resolve()


def _foot_mesh_report(robot: Any, body_name_to_id: dict[str, int], urdf_links: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for foot_name in FOOT_NAMES:
        collision = urdf_links.get(foot_name, {}).get("collision_mesh")
        item: dict[str, Any] = {"collision": collision, "local_bbox": None, "world_lowest_z": None}
        if collision is not None:
            path = _mesh_path(collision["filename"])
            vertices = _read_stl_vertices(path)
            if vertices is not None:
                mins = torch.min(vertices, dim=0).values
                maxs = torch.max(vertices, dim=0).values
                item["local_bbox"] = (tuple(float(v) for v in mins.tolist()), tuple(float(v) for v in maxs.tolist()))
                body_id = body_name_to_id[foot_name]
                pos = robot.data.body_pos_w[0, body_id].detach().cpu()
                quat = robot.data.body_quat_w[0, body_id].detach().cpu()
                world_vertices = pos.unsqueeze(0) + _quat_apply(quat, vertices)
                item["world_lowest_z"] = float(torch.min(world_vertices[:, 2]).item())
        report[foot_name] = item
    return report


def _get_foot_force_z(base_env: Any) -> tuple[float, float] | None:
    try:
        contact_sensor = base_env.scene["contact_forces"]
        body_ids, body_names = contact_sensor.find_bodies(["l_foot_link", "r_foot_link"], preserve_order=True)
        forces = contact_sensor.data.net_forces_w
        force_by_name = {
            str(name): forces[0, int(body_id)].detach().cpu()
            for body_id, name in zip(body_ids, body_names)
        }
        return abs(float(force_by_name["l_foot_link"][2])), abs(float(force_by_name["r_foot_link"][2]))
    except Exception:
        return None


def _get_contact_flags(base_env: Any) -> tuple[bool, bool] | None:
    try:
        contact_sensor = base_env.scene["contact_forces"]
        body_ids, body_names = contact_sensor.find_bodies(["l_foot_link", "r_foot_link"], preserve_order=True)
        contact_time = contact_sensor.data.current_contact_time
        flags = {
            str(name): bool((contact_time[0, int(body_id)] > 0.0).item())
            for body_id, name in zip(body_ids, body_names)
        }
        return flags["l_foot_link"], flags["r_foot_link"]
    except Exception:
        return None


def _get_masses(robot: Any) -> torch.Tensor | None:
    try:
        masses = robot.root_physx_view.get_masses().detach().cpu()
        if masses.ndim == 1:
            masses = masses.unsqueeze(0)
        return masses[0]
    except Exception:
        return None


def _get_inertia_diagonal(robot: Any) -> list[tuple[float, float, float] | None]:
    try:
        inertias = robot.root_physx_view.get_inertias().detach().cpu()
        if inertias.ndim == 2:
            inertias = inertias.unsqueeze(0)
        values = inertias[0]
        diagonals = []
        for item in values:
            flat = item.flatten()
            if flat.numel() >= 9:
                diagonals.append((float(flat[0]), float(flat[4]), float(flat[8])))
            elif flat.numel() >= 3:
                diagonals.append((float(flat[0]), float(flat[1]), float(flat[2])))
            else:
                diagonals.append(None)
        return diagonals
    except Exception:
        return []


def _step_env(env: Any, action: torch.Tensor):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = torch.logical_or(terminated, truncated)
        return obs, reward, done, terminated, truncated, info
    obs, reward, done, info = out
    return obs, reward, done, done, torch.zeros_like(done, dtype=torch.bool), info


def main() -> None:
    import gymnasium as gym
    import g0_robot_lab.tasks  # noqa: F401

    env = gym.make(args_cli.task, cfg=_make_env_cfg())
    try:
        env.reset()
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        action_dim = base_env.action_manager.total_action_dim
        zero_action = torch.zeros((1, action_dim), device=base_env.device, dtype=torch.float32)
        for _ in range(max(args_cli.settle_steps, 0)):
            _step_env(env, zero_action)

        body_names = list(getattr(robot, "body_names", getattr(robot.data, "body_names", [])))
        body_name_to_id = {name: index for index, name in enumerate(body_names)}
        masses = _get_masses(robot)
        total_mass = float(torch.sum(masses).item()) if masses is not None else None
        urdf_links = _load_urdf_links()
        inertia_diag_runtime = _get_inertia_diagonal(robot)

        root_pos = robot.data.root_pos_w[0].detach().cpu()
        root_quat = robot.data.root_quat_w[0].detach().cpu()
        roll_deg, pitch_deg, yaw_deg = _quat_wxyz_to_euler_deg(root_quat)

        body_com_pos_w = robot.data.body_com_pos_w[0].detach().cpu()
        if masses is not None and total_mass and total_mass > 1.0e-8:
            whole_com = torch.sum(body_com_pos_w * masses.unsqueeze(-1), dim=0) / total_mass
        else:
            whole_com = None

        left_foot_pos = robot.data.body_pos_w[0, body_name_to_id["l_foot_link"]].detach().cpu()
        right_foot_pos = robot.data.body_pos_w[0, body_name_to_id["r_foot_link"]].detach().cpu()
        support_center = (left_foot_pos + right_foot_pos) * 0.5
        foot_force_z = _get_foot_force_z(base_env)
        contact_flags = _get_contact_flags(base_env)
        mesh_report = _foot_mesh_report(robot, body_name_to_id, urdf_links)

        log("G0 INITIAL GEOMETRY DEBUG")
        log(f"task: {args_cli.task}")
        log(f"candidate: root_z={args_cli.root_z:.2f} hip={args_cli.hip:.2f} knee={args_cli.knee:.2f} ankle={args_cli.ankle:.2f}")
        log(f"settle_steps: {args_cli.settle_steps}")
        log("")
        log(f"root position: {_fmt_vec(root_pos)}")
        log(f"root orientation wxyz: {_fmt_vec(root_quat)}")
        log(f"root_z: {float(root_pos[2]):.5f}")
        log(f"root_roll_deg: {roll_deg:.5f}")
        log(f"root_pitch_deg: {pitch_deg:.5f}")
        log(f"root_yaw_deg: {yaw_deg:.5f}")
        log(f"total robot mass: {total_mass:.6f} kg" if total_mass is not None else "total robot mass: n/a")
        if whole_com is not None:
            log(f"whole-body COM world position: {_fmt_vec(whole_com)}")
            log(f"COM projection on ground: ({float(whole_com[0]):.5f}, {float(whole_com[1]):.5f})")
            log(f"support center from feet: ({float(support_center[0]):.5f}, {float(support_center[1]):.5f})")
            log(f"com_dx: {float(whole_com[0] - support_center[0]):.5f}")
            log(f"com_dy: {float(whole_com[1] - support_center[1]):.5f}")
        log(f"l_foot_link world position: {_fmt_vec(left_foot_pos)}")
        log(f"r_foot_link world position: {_fmt_vec(right_foot_pos)}")
        for foot_name in FOOT_NAMES:
            item = mesh_report[foot_name]
            bbox = item["local_bbox"]
            if bbox is None:
                log(f"{foot_name} collision mesh bbox: n/a")
            else:
                mins, maxs = bbox
                log(f"{foot_name} collision mesh local bbox min={_fmt_vec(mins)} max={_fmt_vec(maxs)}")
            log(f"{foot_name} lowest collision/mesh point z: {item['world_lowest_z']}")
            log(f"{foot_name} collision source: {item['collision']}")
        if foot_force_z is not None:
            log(f"left/right foot contact force z: {foot_force_z[0]:.5f}, {foot_force_z[1]:.5f}")
        else:
            log("left/right foot contact force z: n/a")
        if contact_flags is not None:
            log(f"left/right contact point count: {int(contact_flags[0])}, {int(contact_flags[1])} (contact sensor body-level proxy)")
        else:
            log("left/right contact point count: n/a")

        log("")
        log("LINK MASS/COM/INERTIA")
        log("link_name                 mass_kg      local_com                  world_com                  inertia_diag")
        warnings: list[str] = []
        for body_id, body_name in enumerate(body_names):
            mass = float(masses[body_id].item()) if masses is not None else urdf_links.get(body_name, {}).get("mass")
            local_com_tensor = getattr(robot.data, "body_com_pos_b", None)
            local_com = (
                tuple(float(v) for v in local_com_tensor[0, body_id].detach().cpu().tolist())
                if local_com_tensor is not None
                else urdf_links.get(body_name, {}).get("com")
            )
            world_com = tuple(float(v) for v in body_com_pos_w[body_id].tolist())
            inertia_diag = inertia_diag_runtime[body_id] if body_id < len(inertia_diag_runtime) else None
            if inertia_diag is None:
                inertia_diag = urdf_links.get(body_name, {}).get("inertia_diag")
            log(f"{body_name:24s} {mass:9.6f}  {_fmt_vec(local_com):26s} {_fmt_vec(world_com):26s} {_fmt_vec(inertia_diag)}")
            if mass <= 0.0:
                warnings.append(f"{body_name}: mass is zero or negative")
            if mass > 1.0:
                warnings.append(f"{body_name}: mass is unusually large for G0 link ({mass:.4f} kg)")
            if inertia_diag is not None:
                if min(inertia_diag) <= 1.0e-9:
                    warnings.append(f"{body_name}: inertia diagonal is extremely small {_fmt_vec(inertia_diag)}")
                if max(inertia_diag) > 0.05:
                    warnings.append(f"{body_name}: inertia diagonal is unusually large {_fmt_vec(inertia_diag)}")

        log("")
        log("WARNINGS")
        if total_mass is not None and not (1.5 <= total_mass <= 2.3):
            warnings.append(f"total_mass {total_mass:.4f} kg is not close to expected 1.9 kg")
        if whole_com is not None:
            if abs(float(whole_com[0] - support_center[0])) > 0.03:
                warnings.append("COM projection has large x offset relative to foot support center")
            if abs(float(whole_com[1] - support_center[1])) > 0.03:
                warnings.append("COM projection has large y offset relative to foot support center")
        left_low = mesh_report["l_foot_link"]["world_lowest_z"]
        right_low = mesh_report["r_foot_link"]["world_lowest_z"]
        if left_low is not None and right_low is not None:
            if abs(left_low - right_low) > 0.005:
                warnings.append("left/right foot lowest collision z differs by more than 5 mm")
            for name, value in (("l_foot_link", left_low), ("r_foot_link", right_low)):
                if value < -0.002:
                    warnings.append(f"{name}: initial foot collision appears to penetrate ground ({value:.5f} m)")
                if value > 0.005:
                    warnings.append(f"{name}: initial foot collision appears suspended above ground ({value:.5f} m)")
        if foot_force_z is not None:
            total_force = foot_force_z[0] + foot_force_z[1]
            if total_force <= 1.0e-6:
                warnings.append("both foot contact forces are near zero")
            elif abs(foot_force_z[0] - foot_force_z[1]) / total_force > 0.5:
                warnings.append("left/right foot contact force imbalance is above 0.5")
        if not warnings:
            log("none")
        else:
            for warning in warnings:
                log(f"- {warning}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
