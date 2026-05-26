"""Isaac Lab GUI receiver for the virtual single-joint DDS test.

Pipeline (run order):
    terminal 1a:  ./cpp/build/dds_to_uds_bridge        (subscribes DDS, serves UDS)
    terminal 1b:  python isaac_gui_receiver.py         (this file — owns the GUI)
    terminal 2:   ./cpp/build/single_joint_sender ...  (publishes DDS frames)

By default this script auto-spawns the bridge as a subprocess; pass
``--no-spawn-bridge`` to manage it yourself.

Frame contract on the UDS:
    24-byte header + 22 * 20-byte motor slots = 464 bytes.
    pos in deg, dq in deg/s, tau in N*m,
    kp in N*m/deg, kd in N*m/(deg/s)   (firmware PD-gain units).

Control law (option A in the design notes):
    * Isaac Lab's built-in PD is active using the stiffness/damping from
      g0.py — this is the authoritative position controller.
    * The joint position target starts at G0_DEFAULT_JOINT_POS so the robot
      drives from the URDF zero pose into the standing pose at startup.
    * Each DDS frame overrides position/velocity targets (deg→rad) per slot
      and adds frame.tau as feedforward effort. The frame's kp/kd are read
      for diagnostics but NOT used to update Isaac PD gains in sim — the
      sender already mirrors g0.py via the codegen, so they should agree.

The robot is spawned floating (disable_gravity=True) and pinned at the root
(fix_root_link=True) with all joints at 0 rad. Without DDS traffic, Isaac
PD pulls every joint into G0_DEFAULT_JOINT_POS. A single-joint sender frame
(--motor-id N --pos X) overrides slot N's target so joint N moves to X while
the rest hold the standing pose.

This script must NEVER be wired to real hardware. The DDS topic is
g0_sim/motor_control_virtual, never mc/motor_control.
"""
from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import subprocess
import sys
import threading
import time

from isaaclab.app import AppLauncher

# ── CLI / AppLauncher (must come before isaaclab.sim and friends) ──────────
_DEFAULT_UDS = "/tmp/g0_sim/motor_control_virtual.sock"
_DEFAULT_TOPIC = "g0_sim/motor_control_virtual"
_FRAME_BYTES = 464
_FRAME_MAGIC = 0x47305349  # 'G0SI'

parser = argparse.ArgumentParser(description="G0 virtual single-joint DDS test receiver.")
parser.add_argument("--uds", default=_DEFAULT_UDS, help="UDS path produced by the bridge")
parser.add_argument("--topic", default=_DEFAULT_TOPIC, help="DDS topic for the bridge to subscribe")
parser.add_argument("--bridge-bin", default=None,
                    help="Path to dds_to_uds_bridge (default: <this_dir>/cpp/build/dds_to_uds_bridge)")
parser.add_argument("--no-spawn-bridge", action="store_true",
                    help="Do not auto-spawn the bridge; expect it to be running externally")
parser.add_argument("--sim-dt", type=float, default=1.0 / 200.0)
parser.add_argument("--log-every", type=int, default=50,
                    help="Print receiver status every N applied frames")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.topic.startswith("mc/"):
    print(f"ERROR: refusing real-hardware topic '{args_cli.topic}'", file=sys.stderr)
    sys.exit(4)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Heavy imports (require simulation_app alive) ────────────────────────────
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

# joint_mapping lives in the sibling python/ dir; add to path.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "python"))
import joint_mapping as jm  # noqa: E402

from g0_robot_lab.assets.robots.g0.g0 import G0_CFG, G0_DEFAULT_JOINT_POS  # noqa: E402


# ── UDS reader thread ───────────────────────────────────────────────────────
class UdsFrameReader:
    """Background thread: connect to UDS, decode frames, expose latest one."""

    def __init__(self, uds_path: str):
        self.uds_path = uds_path
        self._lock = threading.Lock()
        self._latest = None  # (seq, ts_ns, pos_deg[22], dq[22], kp[22], kd[22], tau[22])
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="uds-reader")
        self.frames_received = 0
        self.frames_dropped = 0  # bad magic

    def start(self):
        self._thread.start()

    def stop(self):
        self._running = False

    def get(self):
        with self._lock:
            return self._latest

    def _connect(self):
        for _ in range(80):  # up to ~20s
            if not self._running:
                return None
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self.uds_path)
                s.settimeout(2.0)
                return s
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                time.sleep(0.25)
        return None

    def _run(self):
        s = self._connect()
        if s is None:
            print(f"[receiver] could not connect to UDS {self.uds_path} after 20s")
            return
        print(f"[receiver] UDS connected: {self.uds_path}")
        buf = b""
        while self._running:
            try:
                chunk = s.recv(8192)
            except socket.timeout:
                continue
            except OSError as e:
                print(f"[receiver] UDS recv error: {e}")
                break
            if not chunk:
                print("[receiver] UDS closed by peer")
                break
            buf += chunk
            while len(buf) >= _FRAME_BYTES:
                frame, buf = buf[:_FRAME_BYTES], buf[_FRAME_BYTES:]
                self._decode(frame)
        try:
            s.close()
        except OSError:
            pass

    def _decode(self, frame: bytes):
        magic = struct.unpack_from("<I", frame, 0)[0]
        if magic != _FRAME_MAGIC:
            self.frames_dropped += 1
            return
        _ver, mc = struct.unpack_from("<HH", frame, 4)
        ts_ns, seq = struct.unpack_from("<QQ", frame, 8)
        pos = [0.0] * 22
        dq = [0.0] * 22
        kp = [0.0] * 22
        kd = [0.0] * 22
        tau = [0.0] * 22
        for i in range(22):
            off = 24 + i * 20
            p, d, kpv, kdv, t = struct.unpack_from("<5f", frame, off)
            pos[i], dq[i], kp[i], kd[i], tau[i] = p, d, kpv, kdv, t
        with self._lock:
            self._latest = (seq, ts_ns, pos, dq, kp, kd, tau)
        self.frames_received += 1


# ── Bridge subprocess helper ────────────────────────────────────────────────
def _resolve_bridge_bin() -> str:
    if args_cli.bridge_bin:
        return args_cli.bridge_bin
    return os.path.join(_HERE, "cpp", "build", "dds_to_uds_bridge")


def maybe_spawn_bridge():
    if args_cli.no_spawn_bridge:
        return None
    bridge = _resolve_bridge_bin()
    if not os.path.isfile(bridge) or not os.access(bridge, os.X_OK):
        print(f"[receiver] bridge binary not found / not exec: {bridge}")
        print("           build it (cd cpp && mkdir build && cd build && cmake .. && make) "
              "or pass --no-spawn-bridge.")
        sys.exit(6)
    log_path = "/tmp/g0_sim/bridge.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fp = open(log_path, "w")
    proc = subprocess.Popen(
        [bridge, "--topic", args_cli.topic, "--uds", args_cli.uds],
        stdout=log_fp, stderr=subprocess.STDOUT,
    )
    print(f"[receiver] spawned bridge pid={proc.pid} (log: {log_path})")
    # Wait until socket appears.
    for _ in range(40):
        if os.path.exists(args_cli.uds):
            return proc
        if proc.poll() is not None:
            print(f"[receiver] bridge exited early (rc={proc.returncode}); see {log_path}")
            sys.exit(7)
        time.sleep(0.1)
    print(f"[receiver] bridge did not create UDS within 4s; see {log_path}")
    return proc


# ── Scene ───────────────────────────────────────────────────────────────────
def build_scene() -> Articulation:
    cfg = G0_CFG.replace(prim_path="/World/G0")
    # Float in the air, no gravity, root link pinned in space. With a free
    # root and no gravity, the reaction torque from a single joint command
    # would slowly spin the whole body (angular-momentum conservation),
    # masking the joint's actual motion. Pinning the root isolates joint
    # kinematics, which is exactly what this test needs.
    cfg.spawn.rigid_props.disable_gravity = True
    cfg.spawn.articulation_props.fix_root_link = True
    cfg.init_state.pos = (0.0, 0.0, 0.9)
    # Start all joints at zero so the transition into G0_DEFAULT_JOINT_POS
    # via Isaac PD is visually obvious. The target pose is applied below via
    # set_joint_position_target after sim.reset().
    cfg.init_state.joint_pos = {".*": 0.0}
    cfg.init_state.joint_vel = {".*": 0.0}

    # Keep Isaac Lab's built-in PD with the stiffness/damping from g0.py.
    # That is the authoritative position controller; the main loop only
    # overrides position/velocity targets and adds frame.tau as feedforward.
    # Effort limit is not touched: g0.py reflects the real motor (rated
    # ~0.5 N*m) and the trained policy was tuned to that envelope.

    sim_utils.GroundPlaneCfg().func("/World/groundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.95)).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2000.0))

    robot = Articulation(cfg)
    return robot


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.9))

    robot = build_scene()

    # Bridge before sim.reset so it's up when we start the reader.
    bridge_proc = maybe_spawn_bridge()

    sim.reset()
    print(f"[receiver] articulation joint_names ({len(robot.joint_names)}): {robot.joint_names}")

    # Resolve and cache motor_id (1..22) -> articulation joint index.
    motor_id_to_jidx = jm.validate_against_articulation(robot.joint_names)
    slot_to_jidx = [motor_id_to_jidx[mid] for mid in range(1, 23)]
    print(f"[receiver] mapping OK. slot_to_jidx = {slot_to_jidx}")

    # Diagnostic: confirm Isaac PD picked up g0.py's stiffness/damping. With
    # option-A control (Isaac PD authoritative), these MUST be non-zero or
    # the robot will not hold its pose.
    for name, act in robot.actuators.items():
        kp_max = float(act.stiffness.max().item())
        kd_max = float(act.damping.max().item())
        eff_max = float(act.effort_limit.max().item()) if act.effort_limit is not None else float("nan")
        print(f"[receiver] actuator '{name}': stiffness.max={kp_max:.4f} "
              f"damping.max={kd_max:.4f} effort_limit.max={eff_max:.4f}")

    reader = UdsFrameReader(args_cli.uds)
    reader.start()

    n_joints = len(robot.joint_names)
    device = sim.device

    # Per-joint command tensors in ISAAC units (radians, rad/s, N*m).
    # Initial position target = G0_DEFAULT_JOINT_POS, mapped from joint name
    # to articulation index. This is what Isaac PD pulls toward when no DDS
    # frame has arrived yet (so the robot transitions from the URDF zero
    # pose into the standing pose at startup).
    pos_target_rad = torch.zeros((1, n_joints), device=device)
    for jname, jpos in G0_DEFAULT_JOINT_POS.items():
        if jname in robot.joint_names:
            pos_target_rad[0, robot.joint_names.index(jname)] = float(jpos)
        else:
            print(f"[receiver] WARN: G0_DEFAULT_JOINT_POS has '{jname}' but the "
                  f"articulation does not.")
    dq_target_rps = torch.zeros((1, n_joints), device=device)
    tau_ff        = torch.zeros((1, n_joints), device=device)

    # slot_to_jidx reorder, as a tensor for index_copy_.
    reorder_idx = torch.tensor(slot_to_jidx, device=device, dtype=torch.long)
    deg_to_rad = math.pi / 180.0

    # Push the initial standing-pose target so the very first sim step sees
    # Isaac PD already pulling toward the default pose.
    robot.set_joint_position_target(pos_target_rad)
    robot.set_joint_velocity_target(dq_target_rps)
    robot.set_joint_effort_target(tau_ff)

    last_seq = None
    applied = 0
    last_log_t = time.time()

    try:
        while simulation_app.is_running():
            latest = reader.get()
            if latest is not None and latest[0] != last_seq:
                seq, ts_ns, pos_s, dq_s, kp_s, kd_s, tau_s = latest
                last_seq = seq
                # Wire is in deg / deg/s / N*m; Isaac targets want rad / rad/s.
                pos_wire = torch.tensor(pos_s, device=device, dtype=torch.float32) * deg_to_rad
                dq_wire  = torch.tensor(dq_s,  device=device, dtype=torch.float32) * deg_to_rad
                tau_wire = torch.tensor(tau_s, device=device, dtype=torch.float32)
                pos_target_rad[0].index_copy_(0, reorder_idx, pos_wire)
                dq_target_rps[0] .index_copy_(0, reorder_idx, dq_wire)
                tau_ff[0]        .index_copy_(0, reorder_idx, tau_wire)
                applied += 1
                if applied % args_cli.log_every == 1:
                    nz = [(slot + 1, jm.motor_id_to_joint_name(slot + 1),
                           f"pos={pos_s[slot]:.2f}deg",
                           f"tau={tau_s[slot]:.2f}",
                           f"kp={kp_s[slot]:.1f}",
                           f"kd={kd_s[slot]:.2f}")
                          for slot in range(22)
                          if abs(pos_s[slot]) + abs(dq_s[slot]) + abs(tau_s[slot])
                             + abs(kp_s[slot]) + abs(kd_s[slot]) > 1e-6]
                    now = time.time()
                    rate = args_cli.log_every / max(now - last_log_t, 1e-6)
                    last_log_t = now
                    print(f"[receiver] applied={applied} seq={seq} rate~{rate:.1f}Hz "
                          f"nonzero={nz}")

            # Isaac PD (with g0.py stiffness/damping) does the position
            # control internally based on these targets. tau_ff is added on
            # top via the actuator effort path each step.
            robot.set_joint_position_target(pos_target_rad)
            robot.set_joint_velocity_target(dq_target_rps)
            robot.set_joint_effort_target(tau_ff)
            robot.write_data_to_sim()
            sim.step()
            robot.update(args_cli.sim_dt)
    finally:
        reader.stop()
        if bridge_proc is not None and bridge_proc.poll() is None:
            print("[receiver] terminating bridge subprocess")
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
