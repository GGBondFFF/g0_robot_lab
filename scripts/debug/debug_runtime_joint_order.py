#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

LOG_PATH = Path("logs/debug_runtime_joint_order.txt")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


LOG_PATH.write_text("", encoding="utf-8")
log("[JOINT-DBG] file entered")
log(f"[JOINT-DBG] argv = {sys.argv}")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Debug G0 runtime joint order and action joint order.")
parser.add_argument("--task", type=str, default="G0-Velocity-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--expect_action_sdk_order", action="store_true")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

log("[JOINT-DBG] before AppLauncher")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
log("[JOINT-DBG] after AppLauncher")


def _as_list(value: Any):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.cpu().tolist()
    if isinstance(value, range):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return value


def _print_order(title: str, names: list[str]) -> None:
    log("")
    log("=" * 100)
    log(title)
    log("=" * 100)
    log(f"length = {len(names)}")
    for i, name in enumerate(names):
        log(f"[{i:02d}] {name}")


def _compare_order(title: str, expected: list[str], actual: list[str]) -> bool:
    log("")
    log("=" * 100)
    log(title)
    log("=" * 100)

    same_set = set(expected) == set(actual)
    same_order = expected == actual

    log(f"expected length = {len(expected)}")
    log(f"actual length   = {len(actual)}")
    log(f"same set        = {same_set}")
    log(f"same order      = {same_order}")

    for i in range(max(len(expected), len(actual))):
        e = expected[i] if i < len(expected) else "<missing>"
        a = actual[i] if i < len(actual) else "<missing>"
        mark = "OK" if e == a else "DIFF"
        log(f"[{i:02d}] {mark:4s} expected={e:28s} actual={a}")

    return same_order


def _get_robot_joint_names(robot: Any) -> list[str]:
    joint_names = None

    if hasattr(robot, "data") and hasattr(robot.data, "joint_names"):
        joint_names = _as_list(robot.data.joint_names)

    if joint_names is None and hasattr(robot, "joint_names"):
        joint_names = _as_list(robot.joint_names)

    if joint_names is None:
        log("[ERROR] Cannot find joint names.")
        log(f"dir(robot) = {dir(robot)}")
        if hasattr(robot, "data"):
            log(f"dir(robot.data) = {dir(robot.data)}")
        raise RuntimeError("Cannot find robot joint names.")

    return [str(x) for x in joint_names]


def _get_action_terms(action_manager: Any):
    terms = getattr(action_manager, "_terms", None)

    if terms is None:
        terms = getattr(action_manager, "terms", None)

    if callable(terms):
        terms = terms()

    if isinstance(terms, dict):
        return terms

    log("[ERROR] Cannot inspect action manager terms.")
    log(f"dir(action_manager) = {dir(action_manager)}")
    raise RuntimeError("Cannot inspect action manager terms.")


def _resolve_action_joint_order(action_manager: Any, robot_joint_names: list[str]) -> list[str]:
    terms = _get_action_terms(action_manager)
    action_joint_order: list[str] = []

    log("")
    log("=" * 100)
    log("ACTION MANAGER TERMS")
    log("=" * 100)

    for term_name, term in terms.items():
        joint_ids = getattr(term, "_joint_ids", None)
        if joint_ids is None:
            joint_ids = getattr(term, "joint_ids", None)

        joint_names = getattr(term, "_joint_names", None)
        if joint_names is None:
            joint_names = getattr(term, "joint_names", None)

        joint_ids = _as_list(joint_ids)
        joint_names = _as_list(joint_names)

        log("")
        log(f"[Action term] {term_name}")
        log(f"class       = {term.__class__.__name__}")
        log(f"joint_ids   = {joint_ids}")
        log(f"joint_names = {joint_names}")

        # Prefer the action term's own joint_names when available.
        # In Isaac Lab, joint_ids may be slice(None, None, None), meaning "all resolved joints".
        # Iterating over a slice directly will crash, so handle that explicitly.
        if joint_names is not None:
            resolved = [str(x) for x in joint_names]
        elif isinstance(joint_ids, slice):
            resolved = robot_joint_names[joint_ids]
        elif joint_ids is not None:
            resolved = [robot_joint_names[int(i)] for i in joint_ids]
        else:
            resolved = []
            log("[WARN] This action term has no visible joint_ids or joint_names.")

        log("resolved action order:")
        for i, name in enumerate(resolved):
            log(f"  [{i:02d}] {name}")

        action_joint_order.extend(resolved)

    return action_joint_order


def main() -> None:
    log("[JOINT-DBG] enter main")

    import gymnasium as gym

    log("[JOINT-DBG] before import g0_robot_lab.tasks")
    import g0_robot_lab.tasks  # noqa: F401
    log("[JOINT-DBG] after import g0_robot_lab.tasks")

    log("[JOINT-DBG] before import G0 joint lists")
    from g0_robot_lab.assets.robots.g0 import G0_JOINT_NAMES, G0_JOINT_SDK_NAMES
    log("[JOINT-DBG] after import G0 joint lists")

    log("[JOINT-DBG] before import G0RobotLabEnvCfg")
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg
    log("[JOINT-DBG] after import G0RobotLabEnvCfg")

    log("[JOINT-DBG] before env_cfg")
    env_cfg = G0RobotLabEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device

    log("[JOINT-DBG] before gym.make")
    env = gym.make(args_cli.task, cfg=env_cfg)
    log("[JOINT-DBG] after gym.make")

    try:
        log("[JOINT-DBG] before env.reset")
        env.reset()
        log("[JOINT-DBG] after env.reset")

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        action_manager = base_env.action_manager

        robot_joint_names = _get_robot_joint_names(robot)
        action_joint_order = _resolve_action_joint_order(action_manager, robot_joint_names)

        _print_order("G0_JOINT_NAMES from g0.py", list(G0_JOINT_NAMES))
        _print_order("G0_JOINT_SDK_NAMES from g0.py", list(G0_JOINT_SDK_NAMES))
        _print_order("robot runtime joint_names", robot_joint_names)
        _print_order("action manager resolved action order", action_joint_order)

        _compare_order(
            "COMPARE: G0_JOINT_NAMES vs robot runtime joint_names",
            list(G0_JOINT_NAMES),
            robot_joint_names,
        )

        action_matches_sdk = _compare_order(
            "COMPARE: G0_JOINT_SDK_NAMES vs action manager resolved action order",
            list(G0_JOINT_SDK_NAMES),
            action_joint_order,
        )

        log("")
        log("=" * 100)
        log("SUMMARY")
        log("=" * 100)

        if action_matches_sdk:
            log("[OK] action manager order == G0_JOINT_SDK_NAMES")
        else:
            log("[WARN] action manager order != G0_JOINT_SDK_NAMES")
            log("[WARN] 如果 velocity_env_cfg.py 仍是 joint_names=[\".*\"]，这个结果是正常的。")

        if args_cli.expect_action_sdk_order and not action_matches_sdk:
            raise RuntimeError("Action joint order does not match G0_JOINT_SDK_NAMES.")

    finally:
        log("[JOINT-DBG] before env.close")
        env.close()
        log("[JOINT-DBG] after env.close")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("")
        log("=" * 100)
        log("[ERROR] Exception in debug_runtime_joint_order.py")
        log("=" * 100)
        log(traceback.format_exc())
    finally:
        log("[JOINT-DBG] before simulation_app.close")
        simulation_app.close()
