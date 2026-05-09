#!/usr/bin/env python3
import argparse
import sys

print("[PROBE] file entered", flush=True)
print("[PROBE] argv =", sys.argv, flush=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print("[PROBE] before AppLauncher", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[PROBE] after AppLauncher", flush=True)

try:
    print("[PROBE] before import gymnasium", flush=True)
    import gymnasium as gym
    print("[PROBE] after import gymnasium", flush=True)

    print("[PROBE] before import g0_robot_lab.tasks", flush=True)
    import g0_robot_lab.tasks  # noqa: F401
    print("[PROBE] after import g0_robot_lab.tasks", flush=True)

finally:
    print("[PROBE] before close", flush=True)
    simulation_app.close()
    print("[PROBE] after close", flush=True)
