#!/usr/bin/env python3
"""Validate that the G0 sim2sim scaffolding is present and internally consistent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.sim2sim import g0_sim2sim_config as cfg
except ModuleNotFoundError:
    import g0_sim2sim_config as cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mujoco/g0.xml", help="MuJoCo XML model to validate if mujoco is installed.")
    return parser.parse_args()


def check(condition: bool, ok: str, fail: str, failures: list[str]) -> None:
    if condition:
        print(f"OK: {ok}")
    else:
        print(f"FAIL: {fail}")
        failures.append(fail)


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    root = Path(__file__).resolve().parents[2]
    model = root / args.model

    print("G0 sim2sim setup validation")
    check((root / "scripts" / "sim2sim").is_dir(), "scripts/sim2sim exists", "scripts/sim2sim missing", failures)
    check(model.exists(), f"{args.model} exists", f"{args.model} missing", failures)
    check(cfg.get_action_dim() == 22, "joint/action dimension is 22", "joint/action dimension is not 22", failures)
    check(
        cfg.get_default_joint_pos_array().shape == (22,),
        "default_joint_pos has shape (22,)",
        "default_joint_pos shape is not (22,)",
        failures,
    )
    check(abs(cfg.ACTION_SCALE - 0.12) < 1e-12, "action scale is 0.12", "action scale is not 0.12", failures)
    check(abs(cfg.CONTROL_DT - 0.02) < 1e-12, "control_dt is 0.02", "control_dt is not 0.02", failures)

    try:
        from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface

        interface = G0MuJoCoInterface(model)
        check(len(interface.joint_indices) == 22, "MuJoCo model has all 22 joints", "MuJoCo joint check failed", failures)
    except ModuleNotFoundError as exc:
        print(f"WARNING: MuJoCo not installed, skipping XML load check: {exc}")
    except Exception as exc:
        print(f"FAIL: MuJoCo model validation failed: {exc}")
        failures.append(str(exc))

    print("")
    if failures:
        print("Summary: FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Summary: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
