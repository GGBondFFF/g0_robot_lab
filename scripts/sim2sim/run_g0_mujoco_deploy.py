#!/usr/bin/env python3
"""Run a Unitree-style deploy rollout for G0 in MuJoCo."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface
    from scripts.sim2sim.play_mujoco_g0 import collect_contact_diagnostics
except ModuleNotFoundError:
    from g0_mujoco_interface import G0MuJoCoInterface
    from play_mujoco_g0 import collect_contact_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mujoco/g0.xml", help="Path to MuJoCo XML model.")
    parser.add_argument("--deploy-cfg", default="logs/sim2sim/g0_deploy/params/deploy.yaml", help="Path to deploy.yaml.")
    parser.add_argument("--policy", default=None, help="Path to policy.pt or policy.onnx.")
    parser.add_argument("--steps", type=int, default=500, help="Number of control steps.")
    parser.add_argument("--command", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="lin_x lin_y yaw command.")
    parser.add_argument("--zero-action", action="store_true", help="Use zero policy action.")
    parser.add_argument("--record-rollout", default="logs/sim2sim/g0_deploy/mujoco_deploy_rollout.npz", help="Output .npz path.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy.pt inference.")
    parser.add_argument("--render", action="store_true", help="Render with MuJoCo viewer if available.")
    parser.add_argument("--control-mode", choices=["position", "pd_torque"], default="position", help="MuJoCo control backend.")
    return parser.parse_args()


def load_deploy_cfg(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"deploy.yaml does not exist: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def joint_position_action(deploy_cfg: dict[str, Any]) -> dict[str, Any]:
    actions = deploy_cfg.get("actions") or {}
    if "JointPositionAction" in actions:
        return actions["JointPositionAction"]
    if len(actions) == 1:
        return next(iter(actions.values()))
    raise KeyError(f"Could not find JointPositionAction in deploy actions: {list(actions)}")


def as_vector(values: Any, expected: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 1 and expected > 1:
        array = np.repeat(array, expected)
    if array.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {array.shape}")
    return array


def load_policy(policy_path: str | None, device: str):
    if policy_path is None:
        return None, "zero"
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file does not exist: {path}")
    if path.suffix == ".onnx":
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("ONNX policy requires onnxruntime. Install it or use policy.pt.") from exc
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return session, "onnx"

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Torch is required to load policy.pt. Use --zero-action to run without torch.") from exc
    policy = torch.jit.load(str(path), map_location=device)
    policy.eval()
    return policy, "torchscript"


def infer_policy(policy: Any, policy_kind: str, obs: np.ndarray, action_dim: int, device: str) -> np.ndarray:
    if policy is None or policy_kind == "zero":
        return np.zeros(action_dim, dtype=np.float64)
    if policy_kind == "onnx":
        input_name = policy.get_inputs()[0].name
        output = policy.run(None, {input_name: obs.astype(np.float32)[None, :]})[0]
        return np.asarray(output).squeeze(0).astype(np.float64)

    import torch

    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy(obs_tensor)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float64)


def write_pd_torque_xml(source_model: Path, deploy_cfg: dict[str, Any]) -> Path:
    """Create a motor-actuated XML next to the formal model without modifying it."""

    output = source_model.with_name("g0_pd_torque.xml")
    tree = ET.parse(source_model)
    root = tree.getroot()
    actuator = root.find("actuator")
    if actuator is None:
        raise RuntimeError(f"MuJoCo XML has no <actuator> section: {source_model}")
    actuator.clear()
    joint_names = list(deploy_cfg["joint_names"])
    effort = as_vector(deploy_cfg["effort_limit_sim"], len(joint_names), "effort_limit_sim")
    actuator.append(ET.Comment("Generated by run_g0_mujoco_deploy.py for LowCmd-style PD torque control."))
    for name, limit in zip(joint_names, effort, strict=True):
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": name,
                "joint": name,
                "gear": "1",
                "forcelimited": "true",
                "forcerange": f"{-float(limit):.12g} {float(limit):.12g}",
            },
        )
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def process_action(policy_action: np.ndarray, scale: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clipped_action = np.clip(np.asarray(policy_action, dtype=np.float64), -1.0, 1.0)
    target_joint_pos = offset + scale * clipped_action
    return clipped_action, target_joint_pos


def compute_pd_command(
    target_joint_pos: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    effort_limit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_des = target_joint_pos.copy()
    dq_des = np.zeros_like(q_des)
    tau_ff = np.zeros_like(q_des)
    tau_cmd = tau_ff + kp * (q_des - joint_pos) + kd * (dq_des - joint_vel)
    tau_cmd_clipped = np.clip(tau_cmd, -effort_limit, effort_limit)
    return q_des, dq_des, kp.copy(), kd.copy(), tau_ff, tau_cmd, tau_cmd_clipped


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        print("ERROR: --steps must be positive.", file=sys.stderr)
        return 2
    if args.policy is None and not args.zero_action:
        print("ERROR: provide --policy or use --zero-action.", file=sys.stderr)
        return 2

    deploy_cfg = load_deploy_cfg(args.deploy_cfg)
    joint_names = list(deploy_cfg["joint_names"])
    action_cfg = joint_position_action(deploy_cfg)
    action_dim = len(joint_names)
    scale = as_vector(action_cfg["scale"], action_dim, "action scale")
    offset = as_vector(action_cfg["offset"], action_dim, "action offset")
    kp = as_vector(deploy_cfg["stiffness"], action_dim, "stiffness")
    kd = as_vector(deploy_cfg["damping"], action_dim, "damping")
    effort_limit = as_vector(deploy_cfg["effort_limit_sim"], action_dim, "effort_limit_sim")

    model_path = Path(args.model)
    if args.control_mode == "pd_torque":
        model_path = write_pd_torque_xml(model_path, deploy_cfg)
        print(f"Using generated torque-mode model: {model_path}")

    policy = None
    policy_kind = "zero"
    if not args.zero_action:
        policy, policy_kind = load_policy(args.policy, args.device)

    interface = G0MuJoCoInterface(model_path, command=np.asarray(args.command, dtype=np.float64))
    if interface.joint_names != joint_names:
        raise RuntimeError(
            "MuJoCo joint order does not match deploy.yaml joint_names: "
            f"{interface.joint_names} != {joint_names}"
        )

    viewer = None
    if args.render:
        try:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(interface.model, interface.data)
        except Exception as exc:
            print(f"WARNING: could not start MuJoCo viewer: {exc}", file=sys.stderr)

    rows: dict[str, list[Any]] = {
        "time": [],
        "obs": [],
        "policy_action": [],
        "processed_action": [],
        "target_joint_pos": [],
        "pd_q_des": [],
        "pd_dq_des": [],
        "pd_kp": [],
        "pd_kd": [],
        "pd_tau_ff": [],
        "pd_tau_cmd": [],
        "pd_tau_cmd_clipped": [],
        "joint_pos": [],
        "joint_vel": [],
        "root_pos": [],
        "root_quat": [],
        "root_height": [],
        "base_ang_vel": [],
        "projected_gravity": [],
        "command": [],
        "contact_count": [],
        "foot_ground_contact_count": [],
        "foot_contact_force_norm": [],
        "left_foot_contact_force_norm": [],
        "right_foot_contact_force_norm": [],
    }

    obs = interface.build_observation()
    for _ in range(args.steps):
        policy_action = infer_policy(policy, policy_kind, obs, action_dim, args.device)
        processed_action, target_joint_pos = process_action(policy_action, scale, offset)

        joint_pos_before = interface.get_joint_pos()
        joint_vel_before = interface.get_joint_vel()
        q_des, dq_des, pd_kp, pd_kd, tau_ff, tau_cmd, tau_cmd_clipped = compute_pd_command(
            target_joint_pos, joint_pos_before, joint_vel_before, kp, kd, effort_limit
        )
        if args.control_mode == "position":
            interface.set_position_target(q_des)
        else:
            interface.set_torque_command(tau_cmd_clipped)
        interface.last_action = processed_action.copy()
        interface.step_control_interval()

        root_pos, root_quat = interface.get_root_pose()
        (
            contact_count,
            _max_contact_force_norm,
            foot_ground_contact_count,
            left_foot_force_norm,
            right_foot_force_norm,
            total_foot_force_norm,
        ) = collect_contact_diagnostics(interface)

        rows["time"].append(float(interface.sim_time))
        rows["obs"].append(obs.copy())
        rows["policy_action"].append(policy_action.copy())
        rows["processed_action"].append(processed_action.copy())
        rows["target_joint_pos"].append(target_joint_pos.copy())
        rows["pd_q_des"].append(q_des.copy())
        rows["pd_dq_des"].append(dq_des.copy())
        rows["pd_kp"].append(pd_kp.copy())
        rows["pd_kd"].append(pd_kd.copy())
        rows["pd_tau_ff"].append(tau_ff.copy())
        rows["pd_tau_cmd"].append(tau_cmd.copy())
        rows["pd_tau_cmd_clipped"].append(tau_cmd_clipped.copy())
        rows["joint_pos"].append(interface.get_joint_pos())
        rows["joint_vel"].append(interface.get_joint_vel())
        rows["root_pos"].append(np.full(3, np.nan) if root_pos is None else root_pos)
        rows["root_quat"].append(np.full(4, np.nan) if root_quat is None else root_quat)
        rows["root_height"].append(float("nan") if root_pos is None else float(root_pos[2]))
        rows["base_ang_vel"].append(interface.get_base_ang_vel())
        rows["projected_gravity"].append(interface.get_projected_gravity())
        rows["command"].append(interface.command.copy())
        rows["contact_count"].append(contact_count)
        rows["foot_ground_contact_count"].append(foot_ground_contact_count)
        rows["foot_contact_force_norm"].append(total_foot_force_norm)
        rows["left_foot_contact_force_norm"].append(left_foot_force_norm)
        rows["right_foot_contact_force_norm"].append(right_foot_force_norm)

        obs = interface.build_observation()
        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()

    output = Path(args.record_rollout)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        **{key: np.asarray(value) for key, value in rows.items()},
        action_scale=scale,
        default_joint_pos=offset,
        joint_names=np.asarray(joint_names),
        deploy_yaml_path=np.asarray(str(Path(args.deploy_cfg))),
        policy_path=np.asarray("" if args.policy is None else str(Path(args.policy))),
        control_mode=np.asarray(args.control_mode),
        sim_dt=np.asarray(interface.model.opt.timestep),
        control_dt=np.asarray(float(deploy_cfg["step_dt"])),
        decimation=np.asarray(int(deploy_cfg["decimation"])),
    )
    print(f"Saved deploy MuJoCo rollout: {output}")
    print(f"Finished {args.steps} deploy control steps in {args.control_mode} mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
