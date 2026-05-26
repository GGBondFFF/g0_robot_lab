"""Offline dump of one DDS MotorControlFrame produced by the policy.

Pure offline — does not need plant_state_bridge / policy_control_bridge /
isaac_gui_plant to be running. We:
  1. Load the same TOML the live policy uses.
  2. Build a synthetic "standing at default pose, IMU level, zero joint vel"
     MotorStateFrame and a configurable RobotControlFrame (velocity command).
  3. Run history_length cold-start ticks so the ObsBuilder history is full.
  4. Encode one MotorControlFrame and print it as a 22-row × 5-column table.
  5. Also write the same table + the raw 464-byte hex to a file in docs/.

Usage:
    conda activate g0_isaaclab
    python tools/dump_dds_action.py
    python tools/dump_dds_action.py --vx 0.5 --vy 0.0 --wz 0.0 --out docs/walk_forward.log
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "python"))

import wire_frames as wf
from joint_mapping import MOTOR_ID_TO_JOINT_NAME, build_ordering
import policy_inference as pi


def build_standing_state(ordering) -> wf.MotorStateFrame:
    ms = wf.MotorStateFrame(timestamp_ns=time.time_ns(), sequence_id=1)
    for slot, m in enumerate(ms.motors):
        m.isvalid = 1
        m.pos = float(ordering.default_pos_wire[slot]) * 180.0 / math.pi
        m.dq = 0.0
        m.tau = 0.0
    # IMU: upright, no rotation. plant_state_bridge convention is
    # imu.acc = -projected_gravity_b * 9.81; upright -> proj_g_b = [0,0,-1] ->
    # acc = [0, 0, +9.81].
    ms.imu = wf.ImuData(
        imu_valid=1, mag_valid=0, imu_timestamp=0, mag_timestamp=0,
        acc=(0.0, 0.0, 9.81), gyro=(0.0, 0.0, 0.0), mag=(0.0, 0.0, 0.0),
    )
    return ms


def build_robot_cmd(vx: float, vy: float, wz: float) -> wf.RobotControlFrame:
    rc = wf.RobotControlFrame(timestamp_ns=time.time_ns(), sequence_id=1)
    rc.body.lin_x = vx
    rc.body.lin_y = vy
    rc.body.ang_z = wz
    return rc


def format_table(frame: wf.MotorControlFrame) -> str:
    lines = []
    lines.append("motor_id | joint_name              "
                 "|     pos(deg) |    dq(deg/s) | kp(N*m/deg) "
                 "| kd(N*m/(deg/s)) | tau(N*m)")
    lines.append("-" * 118)
    for slot in range(22):
        m = frame.motors[slot]
        lines.append(
            f"   {slot+1:>2d}    | {MOTOR_ID_TO_JOINT_NAME[slot]:23s} "
            f"|  {m.pos:+11.4f} |  {m.dq:+11.4f} |  {m.kp:9.6f} "
            f"|    {m.kd:10.7f} |  {m.tau:+7.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(REPO / "config" / "policy_inference.toml"))
    ap.add_argument("--vx", type=float, default=0.0,
                    help="body.lin_x in [-1,+1] (mapped per [command_mapping])")
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0,
                    help="body.ang_z [-1,+1] (mapped per [command_mapping])")
    ap.add_argument("--ticks", type=int, default=None,
                    help="warm-up ticks before dumping (default: history_length)")
    ap.add_argument("--dry-run", action="store_true",
                    help="zero policy output instead of loading policy.onnx")
    ap.add_argument("--out", default=None,
                    help="output file path (default: docs/dds_action_<cmd>.log)")
    args = ap.parse_args()

    cfg = pi.load_cfg(Path(args.config), args.dry_run)
    ordering = build_ordering(
        cfg.g0_module, cfg.urdf_strategy, cfg.urdf_explicit)

    print(f"[dump] policy = {cfg.policy_path}")
    print(f"[dump] emit_pd_gains_on_wire = {cfg.emit_pd_gains_on_wire}")
    print(f"[dump] action_scale = {cfg.action_scale}  "
          f"use_default_offset = {cfg.use_default_offset}")
    print(f"[dump] cmd_in (normalised [-1,+1]) = "
          f"vx={args.vx}  vy={args.vy}  wz={args.wz}")

    builder = pi.ObsBuilder(ordering, cfg)
    policy = pi.Policy(cfg.policy_path, dry_run=args.dry_run)

    ms = build_standing_state(ordering)
    rc = build_robot_cmd(args.vx, args.vy, args.wz)

    warmup = args.ticks if args.ticks is not None else cfg.history_length
    for _ in range(warmup):
        obs = builder.step(ms, rc)
        action = policy.act(obs)
        builder.update_last_action(action)

    # One more tick whose action we will dump.
    obs = builder.step(ms, rc)
    action = policy.act(obs)
    print(f"[dump] |action| = {float(np.linalg.norm(action)):.4f}  "
          f"max|a| = {float(np.max(np.abs(action))):.4f}")

    cmd_mapped = builder._vel_cmd(rc)
    print(f"[dump] vel_cmd after mapping (m/s, rad/s) = {cmd_mapped.tolist()}")

    frame = pi.encode_action(action, ordering, cfg, seq_id=1)
    payload = frame.encode()
    assert len(payload) == wf.MC_FRAME_BYTES

    table = format_table(frame)
    print()
    print(table)
    print()
    print(f"[dump] payload bytes = {len(payload)}  "
          f"(MotorControlFrame: 24-byte header + 22*20-byte motors)")

    if args.out is None:
        tag = f"vx{args.vx:+.2f}_vy{args.vy:+.2f}_wz{args.wz:+.2f}"
        args.out = str(REPO / "docs" / f"dds_action_{tag}.log")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write("# DDS MotorControlFrame dumped from policy_inference.\n")
        f.write(f"# policy             : {cfg.policy_path}\n")
        f.write(f"# action_scale       : {cfg.action_scale}\n")
        f.write(f"# use_default_offset : {cfg.use_default_offset}\n")
        f.write(f"# emit_pd_gains_wire : {cfg.emit_pd_gains_on_wire}\n")
        f.write(f"# urdf_order_strategy: {cfg.urdf_strategy}\n")
        f.write(f"# cmd_in [-1,+1]     : vx={args.vx} vy={args.vy} wz={args.wz}\n")
        f.write(f"# vel_cmd mapped     : {cmd_mapped.tolist()}\n")
        f.write(f"# |action| / max|a|  : "
                f"{float(np.linalg.norm(action)):.4f} / "
                f"{float(np.max(np.abs(action))):.4f}\n")
        f.write(f"# warmup_ticks       : {warmup}\n")
        f.write(f"# wire bytes         : {len(payload)}\n")
        f.write("#\n")
        f.write("# Wire units: pos=deg  dq=deg/s  kp=N*m/deg  "
                "kd=N*m/(deg/s)  tau=N*m\n")
        f.write("# Motor id 1..22 maps to joint names per "
                "cpp/include/joint_mapping.hpp.\n#\n")
        f.write(table)
        f.write("\n\n# raw 464-byte payload (hex):\n")
        hex_dump = payload.hex()
        for i in range(0, len(hex_dump), 64):
            f.write(hex_dump[i:i + 64] + "\n")

    print(f"[dump] wrote {out_path}")


if __name__ == "__main__":
    main()
