#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "g0_robot_lab"
RSL_RL_SCRIPT_DIR = REPO_ROOT / "scripts" / "rsl_rl"
for path in (REPO_ROOT, SOURCE_ROOT, RSL_RL_SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import cli_args  # noqa: E402
from _rollout_core import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_TASK,
    run_policy_rollout_validation,
)

parser = argparse.ArgumentParser(description="Validate G0 policy rollout safety inside Isaac Lab.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--root-z", type=float, default=0.233)
parser.add_argument("--json-out", type=Path, default=None)
parser.add_argument("--no-json", action="store_true", default=False)
parser.add_argument("--effort-ratio-threshold", type=float, default=0.9)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(checkpoint=str(DEFAULT_CHECKPOINT))
args_cli = parser.parse_args()

os.environ["G0_ALLOW_HARDWARE"] = "0"


def main() -> int:
    from _rollout_io import print_summary

    json_path = None if args_cli.no_json else args_cli.json_out
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    payload = run_policy_rollout_validation(
        task=args_cli.task,
        checkpoint=Path(args_cli.checkpoint),
        steps=args_cli.steps,
        num_envs=args_cli.num_envs,
        seed=args_cli.seed,
        root_z=args_cli.root_z,
        effort_ratio_threshold=args_cli.effort_ratio_threshold,
        json_out=json_path,
        write_json=not args_cli.no_json,
        device=args_cli.device,
        command=sys.argv[:],
    )
    print_summary(payload)
    simulation_app.close()
    return payload["result"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
