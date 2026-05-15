#!/usr/bin/env python3
"""Roll out a G0 policy in MuJoCo for sim2sim validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.sim2sim import g0_sim2sim_config as cfg
    from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface
except ModuleNotFoundError:
    import g0_sim2sim_config as cfg
    from g0_mujoco_interface import G0MuJoCoInterface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mujoco/g0.xml", help="Path to MuJoCo XML model.")
    parser.add_argument("--policy", default=None, help="Path to TorchScript policy.pt. ONNX is reserved for later.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of control steps to run.")
    parser.add_argument("--command", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="lin_x lin_y yaw command.")
    parser.add_argument("--render", action="store_true", help="Render with MuJoCo viewer if available.")
    parser.add_argument("--record-rollout", default=None, help="Optional .npz output path for rollout data.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy inference.")
    parser.add_argument("--zero-action", action="store_true", help="Run without a policy using zero actions.")
    return parser.parse_args()


def load_torchscript_policy(policy_path: str | None, device: str):
    """Load a TorchScript policy, or return None for zero-action mode."""

    if policy_path is None:
        return None
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file does not exist: {path}")
    if path.suffix == ".onnx":
        raise NotImplementedError("ONNX policy rollout is not implemented yet. Use policy.pt or --zero-action.")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Torch is required to load policy.pt. Use --zero-action to run without torch.") from exc
    policy = torch.jit.load(str(path), map_location=device)
    policy.eval()
    return policy


def infer_action(policy, obs: np.ndarray, device: str) -> np.ndarray:
    """Run TorchScript policy inference for one observation."""

    if policy is None:
        return np.zeros(cfg.get_action_dim(), dtype=np.float64)
    import torch

    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy(obs_tensor)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float64)


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: MuJoCo model does not exist: {model_path}", file=sys.stderr)
        return 2
    if args.steps <= 0:
        print("ERROR: --steps must be positive.", file=sys.stderr)
        return 2
    if args.policy is None and not args.zero_action:
        print("ERROR: provide --policy or use --zero-action.", file=sys.stderr)
        return 2

    policy = load_torchscript_policy(args.policy, args.device)
    interface = G0MuJoCoInterface(model_path, command=np.asarray(args.command, dtype=np.float64))

    viewer = None
    if args.render:
        try:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(interface.model, interface.data)
        except Exception as exc:
            print(f"WARNING: could not start MuJoCo viewer: {exc}", file=sys.stderr)

    rows: dict[str, list[np.ndarray | float]] = {
        "time": [],
        "command": [],
        "action": [],
        "target_joint_pos": [],
        "joint_pos": [],
        "joint_vel": [],
        "root_pos": [],
        "root_quat": [],
        "base_ang_vel": [],
        "projected_gravity": [],
        "obs": [],
    }

    obs = interface.build_observation()
    for _ in range(args.steps):
        action = infer_action(policy, obs, args.device)
        obs, target_joint_pos = interface.step(action)
        root_pos, root_quat = interface.get_root_pose()
        rows["time"].append(float(interface.sim_time))
        rows["command"].append(interface.command.copy())
        rows["action"].append(np.clip(action, -1.0, 1.0))
        rows["target_joint_pos"].append(target_joint_pos.copy())
        rows["joint_pos"].append(interface.get_joint_pos())
        rows["joint_vel"].append(interface.get_joint_vel())
        rows["root_pos"].append(np.full(3, np.nan) if root_pos is None else root_pos)
        rows["root_quat"].append(np.full(4, np.nan) if root_quat is None else root_quat)
        rows["base_ang_vel"].append(interface.get_base_ang_vel())
        rows["projected_gravity"].append(interface.get_projected_gravity())
        rows["obs"].append(obs.copy())
        if viewer is not None:
            viewer.sync()

    if viewer is not None:
        viewer.close()

    if args.record_rollout:
        output = Path(args.record_rollout)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output,
            **{key: np.asarray(value) for key, value in rows.items()},
            joint_names=np.asarray(cfg.get_joint_names()),
            default_joint_pos=cfg.get_default_joint_pos_array(),
            action_scale=np.asarray(cfg.ACTION_SCALE),
            sim_dt=np.asarray(interface.model.opt.timestep),
            decimation=np.asarray(cfg.ISAAC_DECIMATION),
            control_dt=np.asarray(cfg.CONTROL_DT),
        )
        print(f"Saved MuJoCo rollout: {output}")

    print(f"Finished {args.steps} MuJoCo control steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
