#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path
from typing import Any

import torch

from isaaclab.app import AppLauncher

LOG_PATH = Path("logs/debug_zero_action_stability.txt")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.write_text("", encoding="utf-8")


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


parser = argparse.ArgumentParser(description="Run zero-action stability test for G0.")
parser.add_argument("--task", type=str, default="G0-Velocity-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--torque_limit_ratio", type=float, default=0.90)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

log("[ZERO-DBG] before AppLauncher")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
log("[ZERO-DBG] after AppLauncher")


def _reset_env(env):
    """Handle different Isaac Lab / Gym reset return styles."""
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env, action):
    """Handle different Isaac Lab / Gym step return styles."""
    out = env.step(action)

    if not isinstance(out, tuple):
        raise RuntimeError(f"env.step(action) returned non-tuple: {type(out)}")

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = torch.logical_or(terminated, truncated)
        return obs, reward, done, terminated, truncated, info

    if len(out) == 4:
        obs, reward, done, info = out
        terminated = done
        truncated = torch.zeros_like(done, dtype=torch.bool)
        return obs, reward, done, terminated, truncated, info

    raise RuntimeError(f"Unsupported env.step(action) return length: {len(out)}")


def _as_env0(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "ndim") and value.ndim > 0:
        value = value[0]
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value


def _get_env0_joint_scalar(value: Any, joint_id: int) -> float | None:
    value = _as_env0(value)
    if value is None:
        return None
    value = value[joint_id]
    if hasattr(value, "ndim") and value.ndim > 0:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _fmt_float(value: float | None, width: int = 8, precision: int = 4) -> str:
    if value is None:
        return "unavailable"
    return f"{value:{width}.{precision}f}"


def _fmt_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _quat_wxyz_to_euler_deg(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    """Return roll, pitch, yaw in degrees for Isaac Lab's wxyz root quaternion."""
    quat = quat_wxyz.detach().cpu().tolist()
    w, x, y, z = [float(v) for v in quat]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _get_robot_joint_names(robot: Any) -> list[str]:
    joint_names = getattr(robot.data, "joint_names", None)
    if joint_names is None:
        joint_names = getattr(robot, "joint_names", None)
    if joint_names is None:
        raise RuntimeError("Cannot find robot joint names.")
    return [str(name) for name in joint_names]


def _resolve_names(names: list[str], requested_names: list[str]) -> dict[str, int | None]:
    ids: dict[str, int | None] = {}
    for requested_name in requested_names:
        try:
            ids[requested_name] = names.index(requested_name)
        except ValueError:
            ids[requested_name] = None
    return ids


def _get_term_diagnostics(base_env: Any, env_index: int = 0) -> list[tuple[str, bool]]:
    termination_manager = getattr(base_env, "termination_manager", None)
    if termination_manager is None:
        return [("termination_manager_unavailable", False)]

    if hasattr(termination_manager, "get_active_iterable_terms"):
        try:
            terms = termination_manager.get_active_iterable_terms(env_index)
            return [(str(name), bool(float(values[0]))) for name, values in terms]
        except Exception as exc:
            return [(f"get_active_iterable_terms_failed:{exc.__class__.__name__}", False)]

    active_terms = getattr(termination_manager, "active_terms", None)
    if active_terms is None:
        active_terms = getattr(termination_manager, "_term_names", None)
    if active_terms is None:
        return [("termination_terms_unavailable", False)]

    diagnostics: list[tuple[str, bool]] = []
    for term_name in active_terms:
        try:
            value = termination_manager.get_term(term_name)
            diagnostics.append((str(term_name), _fmt_bool(value[env_index])))
        except Exception as exc:
            diagnostics.append((f"{term_name}:unavailable:{exc.__class__.__name__}", False))
    return diagnostics


def _get_contact_force_diagnostics(base_env: Any) -> list[tuple[str, str]]:
    try:
        contact_sensor = base_env.scene["contact_forces"]
    except Exception:
        return [("l_foot_link", "unavailable"), ("r_foot_link", "unavailable")]

    try:
        body_ids, body_names = contact_sensor.find_bodies(["l_foot_link", "r_foot_link"], preserve_order=True)
    except Exception:
        body_names = list(getattr(contact_sensor, "body_names", []))
        body_ids = [body_names.index(name) for name in ("l_foot_link", "r_foot_link") if name in body_names]
        body_names = [body_names[i] for i in body_ids]

    forces = getattr(contact_sensor.data, "net_forces_w", None)
    if forces is None:
        return [(name, "unavailable") for name in ("l_foot_link", "r_foot_link")]

    force_by_name = {name: forces[0, int(body_id)].detach().cpu().tolist() for body_id, name in zip(body_ids, body_names)}
    diagnostics = []
    for name in ("l_foot_link", "r_foot_link"):
        force = force_by_name.get(name)
        if force is None:
            diagnostics.append((name, "unavailable"))
        else:
            diagnostics.append((name, f"[{force[0]: .4f}, {force[1]: .4f}, {force[2]: .4f}]"))
    return diagnostics


def _get_pitch_joint_diagnostics(robot: Any, joint_ids: dict[str, int | None], torque_limit_ratio: float):
    data = robot.data
    joint_pos = getattr(data, "joint_pos", None)
    joint_target = getattr(data, "joint_pos_target", None)
    applied_torque = getattr(data, "applied_torque", None)
    computed_torque = getattr(data, "computed_torque", None)
    effort_limit = getattr(data, "joint_effort_limits", None)
    if effort_limit is None:
        effort_limit = getattr(data, "default_joint_effort_limits", None)

    diagnostics = []
    any_near_limit = False
    any_torque_available = False
    any_nonzero_torque = False

    for joint_name, joint_id in joint_ids.items():
        if joint_id is None:
            diagnostics.append(
                {
                    "joint_name": joint_name,
                    "joint_pos": None,
                    "joint_target": None,
                    "position_error": None,
                    "torque_name": "torque",
                    "torque": None,
                    "effort_limit": None,
                    "near_limit": None,
                }
            )
            continue

        pos = _get_env0_joint_scalar(joint_pos, joint_id)
        target = _get_env0_joint_scalar(joint_target, joint_id)
        position_error = None if pos is None or target is None else target - pos

        applied = _get_env0_joint_scalar(applied_torque, joint_id)
        computed = _get_env0_joint_scalar(computed_torque, joint_id)
        torque_name = "applied_torque"
        torque = applied
        if torque is None and computed is not None:
            torque_name = "computed_torque"
            torque = computed
        elif torque is not None and computed is not None and abs(torque) == 0.0 and abs(computed) > 0.0:
            torque_name = "computed_torque"
            torque = computed

        limit = _get_env0_joint_scalar(effort_limit, joint_id)
        near_limit = None
        if torque is not None and limit is not None and abs(limit) > 0.0:
            any_torque_available = True
            any_nonzero_torque = any_nonzero_torque or abs(torque) > 1.0e-8
            near_limit = abs(torque) >= torque_limit_ratio * abs(limit)
            any_near_limit = any_near_limit or near_limit

        diagnostics.append(
            {
                "joint_name": joint_name,
                "joint_pos": pos,
                "joint_target": target,
                "position_error": position_error,
                "torque_name": torque_name,
                "torque": torque,
                "effort_limit": limit,
                "near_limit": near_limit,
            }
        )

    return diagnostics, any_torque_available, any_nonzero_torque, any_near_limit


def _log_step_diagnostics(
    step: int,
    root_z: float,
    root_euler_deg: tuple[float, float, float],
    reward0: float,
    done0: bool,
    terminated0: bool,
    truncated0: bool,
    term_diagnostics: list[tuple[str, bool]],
    joint_diagnostics: list[dict[str, Any]],
    contact_diagnostics: list[tuple[str, str]],
) -> None:
    roll_deg, pitch_deg, yaw_deg = root_euler_deg

    log("")
    log("-" * 100)
    log(
        f"step={step:04d} "
        f"root_z={root_z: .4f} "
        f"root_roll_deg={roll_deg: .3f} "
        f"root_pitch_deg={pitch_deg: .3f} "
        f"root_yaw_deg={yaw_deg: .3f} "
        f"reward0={reward0: .4f} "
        f"terminated={terminated0} "
        f"truncated={truncated0} "
        f"done={done0}"
    )

    term_text = ", ".join(f"{name}={value}" for name, value in term_diagnostics)
    log(f"termination_terms: {term_text}")

    log("pitch_joint_diagnostics:")
    for item in joint_diagnostics:
        near_limit = item["near_limit"]
        near_limit_text = "unavailable" if near_limit is None else str(near_limit)
        torque = item["torque"]
        effort_limit = item["effort_limit"]
        log(
            f"  {item['joint_name']:22s} "
            f"joint_pos={_fmt_float(item['joint_pos'])} "
            f"joint_target={_fmt_float(item['joint_target'])} "
            f"position_error={_fmt_float(item['position_error'])} "
            f"{item['torque_name']}={_fmt_float(torque)} "
            f"effort_limit_sim={_fmt_float(effort_limit)} "
            f"near_effort_limit={near_limit_text}"
        )

    contact_text = ", ".join(f"{name}={force}" for name, force in contact_diagnostics)
    log(f"foot_contact_forces_w: {contact_text}")


def main() -> None:
    log("[ZERO-DBG] enter main")

    import gymnasium as gym
    import g0_robot_lab.tasks  # noqa: F401

    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

    log("[ZERO-DBG] before env_cfg")
    env_cfg = G0RobotLabEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device

    log("[ZERO-DBG] before gym.make")
    env = gym.make(args_cli.task, cfg=env_cfg)
    log("[ZERO-DBG] after gym.make")

    try:
        log("[ZERO-DBG] before env.reset")
        obs = _reset_env(env)
        log("[ZERO-DBG] after env.reset")

        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        log("")
        log("=" * 100)
        log("ZERO ACTION STABILITY TEST")
        log("=" * 100)
        action_dim = base_env.action_manager.total_action_dim

        log(f"num_envs    = {base_env.num_envs}")
        log(f"num_actions = {action_dim}")
        log(f"steps       = {args_cli.steps}")
        log(f"device      = {base_env.device}")
        log(f"torque_limit_ratio = {args_cli.torque_limit_ratio}")

        pitch_joint_names = [
            "l_hip_pitch_joint",
            "l_knee_pitch_joint",
            "l_ankle_pitch_joint",
            "r_hip_pitch_joint",
            "r_knee_pitch_joint",
            "r_ankle_pitch_joint",
        ]
        robot_joint_names = _get_robot_joint_names(robot)
        pitch_joint_ids = _resolve_names(robot_joint_names, pitch_joint_names)
        log(f"pitch_joint_ids = {pitch_joint_ids}")

        zero_action = torch.zeros(
            (base_env.num_envs, action_dim),
            device=base_env.device,
            dtype=torch.float32,
        )

        log("[ZERO-DBG] before first env.step")

        any_done = False
        any_torque_available = False
        any_nonzero_torque = False
        any_pitch_joint_near_limit = False

        for step in range(args_cli.steps):
            obs, reward, done, terminated, truncated, info = _step_env(env, zero_action)

            root_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
            root_quat = robot.data.root_quat_w[0]
            root_euler_deg = _quat_wxyz_to_euler_deg(root_quat)
            done0 = bool(done[0].item())
            terminated0 = bool(terminated[0].item())
            truncated0 = bool(truncated[0].item())

            term_diagnostics = _get_term_diagnostics(base_env, env_index=0)
            joint_diagnostics, torque_available, nonzero_torque, near_limit = _get_pitch_joint_diagnostics(
                robot, pitch_joint_ids, args_cli.torque_limit_ratio
            )
            contact_diagnostics = _get_contact_force_diagnostics(base_env)

            any_torque_available = any_torque_available or torque_available
            any_nonzero_torque = any_nonzero_torque or nonzero_torque
            any_pitch_joint_near_limit = any_pitch_joint_near_limit or near_limit

            _log_step_diagnostics(
                step=step,
                root_z=float(root_pos[2]),
                root_euler_deg=root_euler_deg,
                reward0=float(reward[0].item()),
                done0=done0,
                terminated0=terminated0,
                truncated0=truncated0,
                term_diagnostics=term_diagnostics,
                joint_diagnostics=joint_diagnostics,
                contact_diagnostics=contact_diagnostics,
            )

            if bool(torch.any(done)):
                any_done = True
                log("[STOP] Environment done during zero-action test.")
                break

        log("")
        log("=" * 100)
        log("ZERO ACTION DIAGNOSIS SUMMARY")
        log("=" * 100)
        if any_done and any_pitch_joint_near_limit:
            log(
                "[DIAGNOSIS] Robot fell and at least one pitch joint torque was near effort_limit_sim. "
                "Prioritize adjusting default_joint_pos."
            )
        elif any_done and any_torque_available and any_nonzero_torque and not any_pitch_joint_near_limit:
            log(
                "[DIAGNOSIS] Robot fell but readable pitch joint torque was not near effort_limit_sim. "
                "Prioritize increasing hip_pitch / knee_pitch / ankle_pitch stiffness and damping."
            )
        elif any_done and (not any_torque_available or not any_nonzero_torque):
            log(
                "[DIAGNOSIS] Robot fell, but Isaac Lab did not expose non-zero readable pitch joint torque for this actuator path. "
                "Use joint position_error and contact force trends, or add lower-level actuator/drive-force instrumentation."
            )
        else:
            log("[DIAGNOSIS] No done signal observed within the configured zero-action steps.")

        log("[ZERO-DBG] finished loop")

    finally:
        log("[ZERO-DBG] before env.close")
        env.close()
        log("[ZERO-DBG] after env.close")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("")
        log("=" * 100)
        log("[ZERO-DBG] ERROR")
        log("=" * 100)
        log(traceback.format_exc())
    finally:
        log("[ZERO-DBG] before simulation_app.close")
        simulation_app.close()
        log("[ZERO-DBG] after simulation_app.close")
