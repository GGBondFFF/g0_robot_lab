"""Isaac Lab GUI plant for the policy_test stack.

This is the "physical robot" side of the two-process sim2real loop:

    plant_state_bridge  <==DDS==>  policy_control_bridge
            ^                              ^
            |UDS                           |UDS
            v                              v
       isaac_gui_plant.py           policy_inference.py
       (this file — GUI)            (policy.onnx)

Forked from test/single_joint_test/isaac_gui_receiver.py. Key differences:

  * Gravity ON, root NOT fixed -> the robot actually stands on the ground.
  * Joints initialise *at* G0_DEFAULT_JOINT_POS (the test stand pose), not
    at zero -> no startup transient before the policy takes over.
  * Subscribes to plant_state_bridge's motor_control_to_plant UDS for joint
    targets, AND publishes MotorStateFrame to motor_state_from_plant UDS at
    a configurable decimation (default: every 4 sim steps -> 50 Hz to match
    training).
  * Isaac PD with g0.py stiffness/damping is authoritative (Option A); wire
    kp/kd/dq/tau are read but the plant ignores them — the policy sends
    pos-only commands.

The DDS topics are sim-only (g0_sim/*). The bridges hard-fail on any topic
starting with mc/, so this script can never accidentally drive real
hardware.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

from isaaclab.app import AppLauncher


# ── CLI / AppLauncher (must come before isaaclab.sim imports) ─────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_UDS_MC = "/tmp/g0_sim/motor_control_to_plant.sock"
_DEFAULT_UDS_MS = "/tmp/g0_sim/motor_state_from_plant.sock"
_DEFAULT_TOPIC_MC = "g0_sim/motor_control"
_DEFAULT_TOPIC_MS = "g0_sim/motor_state"

parser = argparse.ArgumentParser(description="G0 policy_test Isaac Lab plant.")
parser.add_argument("--uds-mc-out", default=_DEFAULT_UDS_MC,
                    help="UDS path on plant_state_bridge: bridge serves motor_control to us")
parser.add_argument("--uds-ms-in",  default=_DEFAULT_UDS_MS,
                    help="UDS path on plant_state_bridge: bridge accepts motor_state from us")
parser.add_argument("--topic-mc", default=_DEFAULT_TOPIC_MC,
                    help="DDS topic the bridge subscribes (must start with g0_sim/)")
parser.add_argument("--topic-ms", default=_DEFAULT_TOPIC_MS,
                    help="DDS topic the bridge publishes (must start with g0_sim/)")
parser.add_argument("--bridge-bin", default=None,
                    help="Path to plant_state_bridge (default: <this_dir>/cpp/build/plant_state_bridge)")
parser.add_argument("--no-spawn-bridge", action="store_true",
                    help="Don't auto-spawn the bridge; expect it to be running externally")
parser.add_argument("--sim-dt", type=float, default=1.0 / 200.0,
                    help="Isaac sim step (default 0.005 = training value)")
parser.add_argument("--state-decimation", type=int, default=4,
                    help="Publish motor_state every N sim steps (default 4 -> 50 Hz)")
parser.add_argument("--log-every", type=int, default=100,
                    help="Status print every N sim steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

for tname in (args_cli.topic_mc, args_cli.topic_ms):
    if tname.startswith("mc/"):
        print(f"ERROR: refusing real-hardware topic '{tname}'", file=sys.stderr)
        sys.exit(4)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── heavy imports (need simulation_app alive) ─────────────────────────────
import socket          # noqa: E402
import threading       # noqa: E402
import torch           # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

sys.path.insert(0, os.path.join(_THIS_DIR, "python"))
import wire_frames as wf  # noqa: E402
from joint_mapping import MOTOR_ID_TO_JOINT_NAME  # noqa: E402

from g0_robot_lab.assets.robots.g0.g0 import G0_CFG, G0_DEFAULT_JOINT_POS  # noqa: E402

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


# ── UDS reader: motor_control commands from the policy ────────────────────
class MotorControlReader:
    """Background thread: connect to UDS, decode MotorControlFrame, expose latest."""

    def __init__(self, uds_path: str):
        self._uds = uds_path
        self._lock = threading.Lock()
        self._latest = None  # wf.MotorControlFrame
        self._running = True
        self.frames_received = 0
        self.frames_dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="plant-mc-reader")

    def start(self): self._thread.start()
    def stop(self): self._running = False

    def get_latest(self):
        with self._lock:
            return self._latest

    def _connect(self):
        for _ in range(80):  # ~20s
            if not self._running:
                return None
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self._uds)
                s.settimeout(2.0)
                return s
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                time.sleep(0.25)
        return None

    def _run(self):
        s = self._connect()
        if s is None:
            print(f"[plant] motor_control UDS connect failed: {self._uds}")
            return
        print(f"[plant] motor_control UDS connected: {self._uds}")
        buf = b""
        size = wf.MC_FRAME_BYTES
        while self._running:
            try:
                chunk = s.recv(8192)
            except socket.timeout:
                continue
            except OSError as e:
                print(f"[plant] motor_control recv error: {e}")
                break
            if not chunk:
                print("[plant] motor_control UDS closed by peer")
                break
            buf += chunk
            while len(buf) >= size:
                f_bytes, buf = buf[:size], buf[size:]
                frame = wf.decode_motor_control(f_bytes)
                if frame is None:
                    self.frames_dropped += 1
                    continue
                with self._lock:
                    self._latest = frame
                self.frames_received += 1
        try: s.close()
        except OSError: pass


# ── UDS writer: motor_state feedback to the policy ────────────────────────
class MotorStateWriter:
    def __init__(self, uds_path: str):
        self._uds = uds_path
        self._sock = None
        self.frames_sent = 0

    def connect(self):
        for _ in range(80):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self._uds)
                s.setblocking(True)
                self._sock = s
                print(f"[plant] motor_state UDS connected: {self._uds}")
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                time.sleep(0.25)
        print(f"[plant] motor_state UDS connect failed: {self._uds}")

    def send(self, frame: wf.MotorStateFrame):
        if self._sock is None:
            return
        try:
            self._sock.sendall(frame.encode())
            self.frames_sent += 1
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[plant] motor_state send failed: {e}; dropping writer")
            try: self._sock.close()
            except OSError: pass
            self._sock = None


# ── bridge subprocess helper ──────────────────────────────────────────────
def _resolve_bridge_bin() -> str:
    if args_cli.bridge_bin:
        return args_cli.bridge_bin
    return os.path.join(_THIS_DIR, "cpp", "build", "plant_state_bridge")


def maybe_spawn_bridge():
    if args_cli.no_spawn_bridge:
        return None
    bridge = _resolve_bridge_bin()
    if not os.path.isfile(bridge) or not os.access(bridge, os.X_OK):
        print(f"[plant] bridge binary not found / not exec: {bridge}")
        sys.exit(6)
    log_path = "/tmp/g0_sim/plant_bridge.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(
        [bridge,
         "--topic-mc", args_cli.topic_mc,
         "--topic-ms", args_cli.topic_ms,
         "--uds-mc-out", args_cli.uds_mc_out,
         "--uds-ms-in",  args_cli.uds_ms_in],
        stdout=log_fp, stderr=subprocess.STDOUT,
    )
    print(f"[plant] spawned plant_state_bridge pid={proc.pid} (log: {log_path})")
    # Wait for both UDS sockets to appear.
    for _ in range(40):
        if (os.path.exists(args_cli.uds_mc_out)
                and os.path.exists(args_cli.uds_ms_in)):
            return proc
        if proc.poll() is not None:
            print(f"[plant] bridge exited early (rc={proc.returncode}); see {log_path}")
            sys.exit(7)
        time.sleep(0.1)
    print(f"[plant] bridge did not create UDS within 4s; see {log_path}")
    return proc


# ── scene ─────────────────────────────────────────────────────────────────
def build_scene() -> Articulation:
    cfg = G0_CFG.replace(prim_path="/World/G0")
    # Real plant: gravity ON, no root pin. Spawn slightly above ground so the
    # feet don't penetrate at t=0; physics settles within a few hundred steps.
    cfg.spawn.rigid_props.disable_gravity = False
    cfg.spawn.articulation_props.fix_root_link = False
    cfg.init_state.pos = (0.0, 0.0, 0.24)
    # Start AT the default pose so there's no startup transient before the
    # policy takes over.
    cfg.init_state.joint_pos = dict(G0_DEFAULT_JOINT_POS)
    cfg.init_state.joint_vel = {".*": 0.0}

    sim_utils.GroundPlaneCfg().func("/World/groundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.95)).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2000.0))

    return Articulation(cfg)


def _build_wire_to_jidx(joint_names) -> list:
    """For each motor_id (1..22), return the articulation joint index."""
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    out = []
    for jname in MOTOR_ID_TO_JOINT_NAME:
        if jname not in name_to_idx:
            raise RuntimeError(f"articulation missing joint '{jname}'")
        out.append(name_to_idx[jname])
    return out


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.9))

    robot = build_scene()
    bridge_proc = maybe_spawn_bridge()

    sim.reset()
    print(f"[plant] articulation joint_names ({len(robot.joint_names)}): {robot.joint_names}")

    wire_to_jidx = _build_wire_to_jidx(robot.joint_names)
    device = sim.device
    n_joints = len(robot.joint_names)
    reorder = torch.tensor(wire_to_jidx, device=device, dtype=torch.long)

    # Confirm Isaac PD has g0.py gains (Option A requirement).
    for name, act in robot.actuators.items():
        kp_max = float(act.stiffness.max().item())
        kd_max = float(act.damping.max().item())
        print(f"[plant] actuator '{name}': kp.max={kp_max:.4f}  kd.max={kd_max:.4f}")

    # Targets in articulation order. Initial = default pose.
    pos_target_rad = torch.zeros((1, n_joints), device=device)
    for jname, jpos in G0_DEFAULT_JOINT_POS.items():
        if jname in robot.joint_names:
            pos_target_rad[0, robot.joint_names.index(jname)] = float(jpos)
    dq_target_rps = torch.zeros((1, n_joints), device=device)
    tau_ff        = torch.zeros((1, n_joints), device=device)

    robot.set_joint_position_target(pos_target_rad)
    robot.set_joint_velocity_target(dq_target_rps)
    robot.set_joint_effort_target(tau_ff)

    reader = MotorControlReader(args_cli.uds_mc_out)
    reader.start()
    writer = MotorStateWriter(args_cli.uds_ms_in)
    writer.connect()

    last_seq = None
    sim_step = 0
    state_seq = 0
    t_last_log = time.time()
    cmd_count = 0

    try:
        while simulation_app.is_running():
            # 1) Latest motor_control command (pos-only on the wire; reorder
            #    wire -> articulation, deg -> rad).
            mc = reader.get_latest()
            if mc is not None and mc.sequence_id != last_seq:
                last_seq = mc.sequence_id
                pos_wire_deg = torch.tensor(
                    [m.pos for m in mc.motors], device=device, dtype=torch.float32)
                pos_target_rad[0].index_copy_(0, reorder, pos_wire_deg * DEG2RAD)
                # Policy sends 0 for dq/tau/kp/kd under Option A; pass them
                # through anyway in case a future agent uses feedforward.
                dq_wire_dps = torch.tensor(
                    [m.dq for m in mc.motors], device=device, dtype=torch.float32)
                tau_wire = torch.tensor(
                    [m.tau for m in mc.motors], device=device, dtype=torch.float32)
                dq_target_rps[0].index_copy_(0, reorder, dq_wire_dps * DEG2RAD)
                tau_ff[0]       .index_copy_(0, reorder, tau_wire)
                cmd_count += 1

            robot.set_joint_position_target(pos_target_rad)
            robot.set_joint_velocity_target(dq_target_rps)
            robot.set_joint_effort_target(tau_ff)
            robot.write_data_to_sim()
            sim.step()
            robot.update(args_cli.sim_dt)
            sim_step += 1

            # 2) Publish motor_state at the configured decimation.
            if sim_step % args_cli.state_decimation == 0:
                state_seq += 1
                # Read articulation state.
                joint_pos_rad = robot.data.joint_pos[0].detach().cpu().numpy()
                joint_vel_rps = robot.data.joint_vel[0].detach().cpu().numpy()
                # projected_gravity_b: gravity vector expressed in body frame
                # (unit-length, typically ~[0,0,-1] when upright).
                proj_g = robot.data.projected_gravity_b[0].detach().cpu().numpy()
                ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy()

                # Reorder articulation -> wire (motor_id 1..22).
                state = wf.MotorStateFrame(
                    timestamp_ns=time.time_ns(),
                    sequence_id=state_seq,
                )
                for slot, jidx in enumerate(wire_to_jidx):
                    m = state.motors[slot]
                    m.isvalid = 1
                    m.pos = float(joint_pos_rad[jidx]) * RAD2DEG
                    m.dq  = float(joint_vel_rps[jidx]) * RAD2DEG
                    m.tau = 0.0
                    m.status = 0
                    m.fpc_temper = 0.0
                    m.pcb_temper = 0.0

                # Pack IMU. We use the convention that acc holds gravity in
                # body frame at 9.81 magnitude so policy_inference's
                # _projected_gravity_from_imu (which does -acc/|acc|) yields
                # exactly Isaac's projected_gravity_b. gyro carries the body
                # frame angular velocity (rad/s) the policy reads directly.
                imu_acc = (-float(proj_g[0]) * 9.81,
                           -float(proj_g[1]) * 9.81,
                           -float(proj_g[2]) * 9.81)
                state.imu = wf.ImuData(
                    imu_valid=1, mag_valid=0,
                    imu_timestamp=int(time.time_ns() // 1_000_000) & 0xFFFFFFFF,
                    mag_timestamp=0,
                    acc=imu_acc,
                    gyro=(float(ang_vel_b[0]),
                          float(ang_vel_b[1]),
                          float(ang_vel_b[2])),
                    mag=(0.0, 0.0, 0.0),
                )
                writer.send(state)

            if sim_step % args_cli.log_every == 0:
                now = time.time()
                rate = args_cli.log_every / max(now - t_last_log, 1e-6)
                t_last_log = now
                print(f"[plant] sim_step={sim_step}  cmd_rx={reader.frames_received}  "
                      f"cmd_applied={cmd_count}  state_tx={writer.frames_sent}  "
                      f"loop~{rate:.1f}Hz")
    finally:
        reader.stop()
        if bridge_proc is not None and bridge_proc.poll() is None:
            print("[plant] terminating bridge subprocess")
            bridge_proc.terminate()
            try:
                bridge_proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                bridge_proc.kill()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
