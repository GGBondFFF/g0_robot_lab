#!/usr/bin/env python3
"""Dump MuJoCo root-frame observations under controlled root states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface  # noqa: E402
from scripts.sim2sim.root_frame_samples import controlled_samples, quat_wxyz_from_rpy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mujoco/g0.xml", help="Path to MuJoCo XML model.")
    parser.add_argument("--output", default="logs/sim2sim/root_frame/mujoco_controlled_root_state.npz", help="Output .npz path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    interface = G0MuJoCoInterface(args.model)
    rows: dict[str, list[Any]] = {
        "sample_name": [],
        "commanded_rpy": [],
        "commanded_ang_vel": [],
        "root_quat": [],
        "projected_gravity": [],
        "base_ang_vel": [],
        "root_pos": [],
        "root_height": [],
    }
    samples = controlled_samples()
    for sample in samples:
        interface.reset()
        quat = quat_wxyz_from_rpy(*sample.rpy)
        interface.data.qpos[0:3] = np.asarray([0.0, 0.0, 0.23], dtype=np.float64)
        interface.data.qpos[3:7] = quat
        interface.data.qvel[0:3] = 0.0
        interface.data.qvel[3:6] = np.asarray(sample.ang_vel, dtype=np.float64)
        interface.mujoco.mj_forward(interface.model, interface.data)
        root_pos, root_quat = interface.get_root_pose()
        if root_pos is None or root_quat is None:
            raise RuntimeError("MuJoCo model does not expose a free root pose")

        rows["sample_name"].append(sample.name)
        rows["commanded_rpy"].append(np.asarray(sample.rpy, dtype=np.float64))
        rows["commanded_ang_vel"].append(np.asarray(sample.ang_vel, dtype=np.float64))
        rows["root_quat"].append(root_quat)
        rows["projected_gravity"].append(interface.get_projected_gravity())
        rows["base_ang_vel"].append(interface.get_base_ang_vel())
        rows["root_pos"].append(root_pos)
        rows["root_height"].append(float(root_pos[2]))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **{key: np.asarray(value) for key, value in rows.items()})
    print(f"Saved MuJoCo controlled root state: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
