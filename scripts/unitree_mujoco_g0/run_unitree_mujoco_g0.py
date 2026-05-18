#!/usr/bin/env python3
"""Run G0 in a Unitree LowCmd-style Python MuJoCo control loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sim2sim import g0_sim2sim_config as cfg  # noqa: E402
from scripts.sim2sim.policy_io import load_policy, metadata_from_npz, policy_metadata, require_absolute_path  # noqa: E402
from scripts.sim2sim.play_mujoco_g0 import collect_contact_diagnostics  # noqa: E402
from scripts.unitree_mujoco_g0.g0_lowcmd_bridge import G0UnitreeMujocoBridge  # noqa: E402
from scripts.unitree_mujoco_g0.generate_g0_unitree_mjcf import DEFAULT_OUTPUT_DIR, generate  # noqa: E402


DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "g0.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Unitree-style G0 MJCF path.")
    parser.add_argument("--source-model", default="mujoco/g0.xml", help="Source model used when --prepare-model is set.")
    parser.add_argument("--prepare-model", action="store_true", help="Generate mujoco/unitree_mujoco_g0/g0.xml before running.")
    parser.add_argument("--mode", choices=["zero", "replay_target", "policy"], required=True)
    parser.add_argument("--golden", default=None, help="Isaac golden .npz for replay_target.")
    parser.add_argument("--policy", default=None, help="Absolute path to TorchScript policy or raw RSL-RL checkpoint for policy mode.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--command", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--task", default="G0-Velocity-v0", help="Task id recorded in rollout metadata.")
    parser.add_argument("--record-rollout", required=True, help="Output .npz.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def infer_action(policy: Any, obs: np.ndarray, device: str) -> np.ndarray:
    if policy is None:
        return np.zeros(cfg.get_action_dim(), dtype=np.float64)
    import torch

    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy(obs_tensor)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float64)


def load_golden(path: str | None, steps: int) -> dict[str, np.ndarray]:
    if path is None:
        raise ValueError("--golden is required for replay_target mode")
    golden_path = Path(path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Isaac golden file does not exist: {golden_path}")
    data = np.load(golden_path, allow_pickle=True)
    required = ["target_joint_pos"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Golden file is missing required keys: {missing}")
    result = {key: np.asarray(data[key], dtype=np.float64) for key in data.files if key in {"target_joint_pos", "action", "command"}}
    result.update(metadata_from_npz(data))
    if result["target_joint_pos"].shape[0] < steps:
        raise ValueError(f"Golden target_joint_pos has {result['target_joint_pos'].shape[0]} steps, need {steps}")
    return result


def _cmd_arrays(lowcmd) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray([motor.q for motor in lowcmd.motor_cmd], dtype=np.float64)
    dq = np.asarray([motor.dq for motor in lowcmd.motor_cmd], dtype=np.float64)
    kp = np.asarray([motor.kp for motor in lowcmd.motor_cmd], dtype=np.float64)
    kd = np.asarray([motor.kd for motor in lowcmd.motor_cmd], dtype=np.float64)
    tau = np.asarray([motor.tau for motor in lowcmd.motor_cmd], dtype=np.float64)
    return q, dq, kp, kd, tau


def run(args: argparse.Namespace) -> Path:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.mode == "policy" and args.policy is None:
        raise ValueError("policy mode requires --policy with an absolute path")
    model_path = Path(args.model)
    if args.prepare_model or not model_path.exists():
        model_path, _scene_path = generate(Path(args.source_model), DEFAULT_OUTPUT_DIR)

    policy = None
    policy_path_for_metadata: str | None = None
    if args.mode == "policy":
        policy_path = require_absolute_path(args.policy, "--policy")
        policy, _policy_kind = load_policy(policy_path, args.device)
        policy_path_for_metadata = str(policy_path)
    golden = load_golden(args.golden, args.steps) if args.mode == "replay_target" else None
    if golden is not None and "policy_path" in golden and str(np.asarray(golden["policy_path"]).item()):
        policy_path_for_metadata = str(np.asarray(golden["policy_path"]).item())
    bridge = G0UnitreeMujocoBridge(model_path, command=np.asarray(args.command, dtype=np.float64))

    rows: dict[str, list[Any]] = {
        "time": [],
        "obs": [],
        "action": [],
        "policy_action": [],
        "processed_action": [],
        "target_joint_pos": [],
        "lowcmd_q": [],
        "lowcmd_dq": [],
        "lowcmd_kp": [],
        "lowcmd_kd": [],
        "lowcmd_tau": [],
        "tau_cmd": [],
        "tau_cmd_clipped": [],
        "tau_saturation": [],
        "joint_pos": [],
        "joint_vel": [],
        "motor_tau_est": [],
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

    obs = bridge.build_observation()
    for step_index in range(args.steps):
        if golden is not None and "command" in golden:
            bridge.command = golden["command"][step_index].copy()

        if args.mode == "zero":
            processed_action = np.zeros(cfg.get_action_dim(), dtype=np.float64)
            policy_action = processed_action.copy()
            target_joint_pos = cfg.compute_target_joint_pos(processed_action)
        elif args.mode == "replay_target":
            target_joint_pos = golden["target_joint_pos"][step_index]
            if "action" in golden:
                policy_action = golden["action"][step_index]
                processed_action = np.clip(policy_action, -1.0, 1.0)
            else:
                processed_action = (target_joint_pos - bridge.default_joint_pos) / cfg.ACTION_SCALE
                policy_action = processed_action.copy()
        else:
            policy_action = infer_action(policy, obs, args.device)
            processed_action, target_joint_pos = bridge.policy_action_to_target(policy_action)

        lowcmd = bridge.build_lowcmd(target_joint_pos)
        obs = bridge.step_lowcmd(lowcmd, last_action=processed_action)
        lowcmd_q, lowcmd_dq, lowcmd_kp, lowcmd_kd, lowcmd_tau = _cmd_arrays(lowcmd)
        root_pos, root_quat = bridge.get_root_pose()
        (
            contact_count,
            _max_contact_force_norm,
            foot_ground_contact_count,
            left_foot_force_norm,
            right_foot_force_norm,
            total_foot_force_norm,
        ) = collect_contact_diagnostics(bridge)

        rows["time"].append(float(bridge.sim_time))
        rows["obs"].append(obs.copy())
        rows["action"].append(processed_action.copy())
        rows["policy_action"].append(policy_action.copy())
        rows["processed_action"].append(processed_action.copy())
        rows["target_joint_pos"].append(target_joint_pos.copy())
        rows["lowcmd_q"].append(lowcmd_q)
        rows["lowcmd_dq"].append(lowcmd_dq)
        rows["lowcmd_kp"].append(lowcmd_kp)
        rows["lowcmd_kd"].append(lowcmd_kd)
        rows["lowcmd_tau"].append(lowcmd_tau)
        rows["tau_cmd"].append(bridge.last_tau_cmd.copy())
        rows["tau_cmd_clipped"].append(bridge.last_tau_cmd_clipped.copy())
        rows["tau_saturation"].append(np.abs(bridge.last_tau_cmd) > bridge.effort_limit + 1e-12)
        rows["joint_pos"].append(bridge.get_joint_pos())
        rows["joint_vel"].append(bridge.get_joint_vel())
        rows["motor_tau_est"].append(bridge.get_motor_tau_est())
        rows["root_pos"].append(root_pos)
        rows["root_quat"].append(root_quat)
        rows["root_height"].append(float(root_pos[2]))
        rows["base_ang_vel"].append(bridge.get_base_ang_vel())
        rows["projected_gravity"].append(bridge.get_projected_gravity())
        rows["command"].append(bridge.command.copy())
        rows["contact_count"].append(contact_count)
        rows["foot_ground_contact_count"].append(foot_ground_contact_count)
        rows["foot_contact_force_norm"].append(total_foot_force_norm)
        rows["left_foot_contact_force_norm"].append(left_foot_force_norm)
        rows["right_foot_contact_force_norm"].append(right_foot_force_norm)

    output = Path(args.record_rollout)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = policy_metadata(
        policy_path_for_metadata,
        task=args.task,
        command=bridge.command,
        steps=args.steps,
    )
    metadata.pop("command", None)
    np.savez(
        output,
        **{key: np.asarray(value) for key, value in rows.items()},
        default_joint_pos=bridge.default_joint_pos,
        stiffness=bridge.kp,
        damping=bridge.kd,
        effort_limit_sim=bridge.effort_limit,
        model_path=np.asarray(str(model_path)),
        mode=np.asarray(args.mode),
        sim_dt=np.asarray(float(bridge.model.opt.timestep)),
        control_dt=np.asarray(cfg.CONTROL_DT),
        decimation=np.asarray(cfg.ISAAC_DECIMATION),
        **metadata,
    )
    print(f"Saved Unitree-style G0 rollout: {output}")
    print(f"Finished {args.steps} steps in {args.mode} mode.")
    return output


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
