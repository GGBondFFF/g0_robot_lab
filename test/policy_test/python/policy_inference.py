"""policy_inference.py — RL velocity policy runner for the policy_test stack.

Reads policy_inference.toml, connects to the three UDS sockets opened by
policy_control_bridge, builds 77-dim observations (× 5 history) from the
incoming RobotControl + MotorState frames, runs torch.jit policy.pt, and
publishes MotorControlFrame at the configured rate (50 Hz to match training).

This is the sim2real-style "robot brain" process. The plant side runs in a
separate Isaac Lab process behind plant_state_bridge.

Run inside the g0_isaaclab conda env (matches isaac-sim's Python and has
numpy/torch + tomllib in 3.11+):

    conda activate g0_isaaclab
    python policy_inference.py --config ../config/policy_inference.toml

CLI flags:
    --config <path>   TOML config (default: ../config/policy_inference.toml)
    --dry-run         override [runtime].dry_run = true; bypass policy.pt and
                      hold default pose. Useful for IO smoke-tests before
                      verifying obs alignment.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wire_frames as wf
from joint_mapping import build_ordering, JointOrdering
from uds_client import FrameReader, FrameWriter, connect_with_retry


# Python 3.11+ has stdlib tomllib; isaac-sim's python typically ships it.
try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


DEG2RAD = math.pi / 180.0


# ─── config loading ───────────────────────────────────────────────────────
@dataclass
class Cfg:
    policy_path: str
    g0_module: str
    rate_hz: float
    action_scale: float
    use_default_offset: bool
    action_clip: float
    emit_pd_gains_on_wire: bool
    uds_rc: str
    uds_ms: str
    uds_mc: str
    connect_timeout_s: float
    connect_retry_s: float
    scale_ang_vel: float
    scale_joint_vel: float
    history_length: int
    history_order: str
    gait_period_s: float
    cmd_vx: tuple
    cmd_vy: tuple
    cmd_wz: tuple
    urdf_strategy: str
    urdf_explicit: list
    dry_run: bool
    log_every: int


def load_cfg(path: Path, force_dry_run: bool) -> Cfg:
    with open(path, "rb") as f:
        t = tomllib.load(f)
    paths = t["paths"]
    ctrl  = t["control"]
    uds   = t["uds"]
    scl   = t["obs"]["scales"]
    hist  = t["obs"]["history"]
    gait  = t["obs"]["gait"]
    cm    = t["command_mapping"]
    jo    = t["joint_order"]
    rt    = t.get("runtime", {})
    return Cfg(
        policy_path=paths["policy"],
        g0_module=paths["g0_module"],
        rate_hz=float(ctrl["rate_hz"]),
        action_scale=float(ctrl["action_scale"]),
        use_default_offset=bool(ctrl["use_default_offset"]),
        action_clip=float(ctrl.get("action_clip", -1.0)),
        emit_pd_gains_on_wire=bool(ctrl.get("emit_pd_gains_on_wire", True)),
        uds_rc=uds["robot_control"],
        uds_ms=uds["motor_state"],
        uds_mc=uds["motor_control"],
        connect_timeout_s=float(uds.get("connect_timeout_s", 10.0)),
        connect_retry_s=float(uds.get("connect_retry_s", 0.2)),
        scale_ang_vel=float(scl["base_ang_vel"]),
        scale_joint_vel=float(scl["joint_vel"]),
        history_length=int(hist["length"]),
        history_order=str(hist.get("order", "back")),
        gait_period_s=float(gait["period_s"]),
        cmd_vx=tuple(cm["lin_vel_x"]),
        cmd_vy=tuple(cm["lin_vel_y"]),
        cmd_wz=tuple(cm["ang_vel_z"]),
        urdf_strategy=str(jo.get("urdf_order_strategy", "alphabetical")),
        urdf_explicit=list(jo.get("urdf_order_explicit", [])),
        dry_run=bool(force_dry_run or rt.get("dry_run", False)),
        log_every=int(rt.get("log_every", 50)),
    )


# ─── command mapping ([-1,+1] -> asymmetric m/s, rad/s) ───────────────────
def map_axis(value: float, neg_max: float, pos_max: float) -> float:
    """Clamp value to [-1, +1] then map into [neg_max, pos_max] piecewise.

    Positive half: 0..1  -> 0..pos_max
    Negative half: -1..0 -> neg_max..0  (neg_max is itself negative)
    """
    v = max(-1.0, min(1.0, value))
    if v >= 0.0:
        return v * pos_max
    return (-v) * neg_max  # v<0, neg_max<0 -> negative result


# ─── observation builder ──────────────────────────────────────────────────
class ObsBuilder:
    """Build 385-dim observation in Isaac Lab's term-major history layout.

    velocity_env_cfg.py has concatenate_terms=True + history_length=5. In
    Isaac Lab that produces, per term, a tensor of shape (history_length,
    term_dim) flattened to (history_length * term_dim), then concatenates
    all term-tensors in declaration order:

       base_ang_vel    history(5) * 3   = 15   (scale 0.2)
       projected_grav  history(5) * 3   = 15
       velocity_cmds   history(5) * 3   = 15
       joint_pos_rel   history(5) * 22  = 110  (URDF order)
       joint_vel_rel   history(5) * 22  = 110  (URDF order, scale 0.05)
       last_action     history(5) * 22  = 110  (SDK order — what policy emitted)
       gait_phase      history(5) * 2   = 10
       ────────────────────────────────── 385

    Within each term's history, the order is oldest -> newest (history_order
    = 'back'), the Isaac Lab default. Set [obs.history].order = 'front' in
    the TOML to flip if a future training run uses the opposite convention.
    """

    TERM_DIMS = (("base_ang_vel", 3),
                 ("projected_gravity", 3),
                 ("velocity_commands", 3),
                 ("joint_pos_rel", 22),
                 ("joint_vel_rel", 22),
                 ("last_action", 22),
                 ("gait_phase", 2))
    TOTAL_PER_FRAME = sum(d for _, d in TERM_DIMS)  # 77

    def __init__(self, ordering: JointOrdering, cfg: Cfg):
        self._o = ordering
        self._cfg = cfg
        self._H = cfg.history_length
        # One ring buffer per term, shape (H, dim). Initialise to zero (same
        # behaviour as Isaac Lab's ObservationManager when env resets).
        self._hist = {name: np.zeros((self._H, dim), dtype=np.float32)
                      for name, dim in self.TERM_DIMS}
        self._step_counter = 0
        self._step_dt = 1.0 / cfg.rate_hz
        self._last_action_sdk = np.zeros(22, dtype=np.float32)

    @property
    def total_obs_dim(self) -> int:
        return self.TOTAL_PER_FRAME * self._H  # 385 at H=5

    @property
    def last_action_sdk(self) -> np.ndarray:
        return self._last_action_sdk

    def update_last_action(self, action_sdk: np.ndarray) -> None:
        self._last_action_sdk = action_sdk.astype(np.float32, copy=True)

    def _gait_phase(self) -> np.ndarray:
        t = (self._step_counter * self._step_dt) % self._cfg.gait_period_s
        phase = t / self._cfg.gait_period_s * 2.0 * math.pi
        return np.asarray([math.sin(phase), math.cos(phase)], dtype=np.float32)

    def _projected_gravity_from_imu(self, imu: wf.ImuData) -> np.ndarray:
        """Best-effort projected gravity from accelerometer (m/s^2 in body frame).

        Isaac trains with projected_gravity = R_world->body @ [0,0,-1]. The
        plant_state_bridge / isaac_gui_plant.py is wired to pack
        -projected_gravity_b * 9.81 into imu.acc, so -acc/|acc| recovers
        Isaac's projected_gravity_b exactly.
        """
        ax, ay, az = imu.acc
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-6:
            return np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
        return np.asarray([-ax / norm, -ay / norm, -az / norm], dtype=np.float32)

    def _vel_cmd(self, rc: Optional[wf.RobotControlFrame]) -> np.ndarray:
        if rc is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray([
            map_axis(rc.body.lin_x, *self._cfg.cmd_vx),
            map_axis(rc.body.lin_y, *self._cfg.cmd_vy),
            map_axis(rc.body.ang_z, *self._cfg.cmd_wz),
        ], dtype=np.float32)

    def _push_term(self, name: str, value: np.ndarray) -> None:
        buf = self._hist[name]
        if self._cfg.history_order == "back":
            buf[:-1] = buf[1:]
            buf[-1] = value
        else:  # "front" — newest first
            buf[1:] = buf[:-1]
            buf[0] = value

    def step(self,
             ms: wf.MotorStateFrame,
             rc: Optional[wf.RobotControlFrame]) -> np.ndarray:
        # Per-term values for THIS tick.
        gyro = np.asarray(ms.imu.gyro, dtype=np.float32) * self._cfg.scale_ang_vel
        grav = self._projected_gravity_from_imu(ms.imu)
        cmd  = self._vel_cmd(rc)

        pos_wire_rad = np.asarray(
            [m.pos for m in ms.motors], dtype=np.float32) * DEG2RAD
        vel_wire_rad = np.asarray(
            [m.dq for m in ms.motors], dtype=np.float32) * DEG2RAD
        joint_pos_rel = pos_wire_rad[self._o.wire_to_urdf] - self._o.default_pos_urdf
        joint_vel_scl = vel_wire_rad[self._o.wire_to_urdf] * self._cfg.scale_joint_vel

        gait = self._gait_phase()

        # Push each term into its own history buffer.
        self._push_term("base_ang_vel",      gyro)
        self._push_term("projected_gravity", grav)
        self._push_term("velocity_commands", cmd)
        self._push_term("joint_pos_rel",     joint_pos_rel)
        self._push_term("joint_vel_rel",     joint_vel_scl)
        self._push_term("last_action",       self._last_action_sdk)
        self._push_term("gait_phase",        gait)

        # Term-major concatenation: each term's (H, dim) flattened row-major
        # (history slowest, term-dim fastest), then all terms concatenated in
        # declaration order.
        out = np.concatenate(
            [self._hist[name].reshape(-1) for name, _ in self.TERM_DIMS])
        assert out.shape == (self.total_obs_dim,), out.shape

        self._step_counter += 1
        return out


# ─── policy wrapper ───────────────────────────────────────────────────────
class Policy:
    """Auto-detects format by file extension.

    .onnx -> onnxruntime InferenceSession (sim2real-deployment standard)
    .pt   -> torch.jit.load
    """

    def __init__(self, path: str, dry_run: bool):
        self._dry_run = dry_run
        self._kind = None
        if dry_run:
            print(f"[policy] DRY-RUN: skipping policy load ({path})")
            return

        ext = Path(path).suffix.lower()
        if ext == ".onnx":
            import onnxruntime as ort
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in ort.get_available_providers()
                         else ["CPUExecutionProvider"])
            self._session = ort.InferenceSession(path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._kind = "onnx"
            print(f"[policy] loaded onnx {path}  providers={providers}  "
                  f"input='{self._input_name}'  output='{self._output_name}'")
        elif ext == ".pt":
            import torch
            self._torch = torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = torch.jit.load(path, map_location=self._device)
            self._model.eval()
            self._kind = "pt"
            print(f"[policy] loaded torch.jit {path} on {self._device}")
        else:
            raise ValueError(f"unsupported policy extension '{ext}': {path}")

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self._dry_run:
            return np.zeros(22, dtype=np.float32)
        if self._kind == "onnx":
            inp = obs.reshape(1, -1).astype(np.float32)
            out = self._session.run([self._output_name], {self._input_name: inp})[0]
            return out.reshape(-1).astype(np.float32)
        # torch.jit
        t = self._torch.from_numpy(obs).to(self._device).unsqueeze(0)
        with self._torch.inference_mode():
            out = self._model(t)
        return out.squeeze(0).detach().cpu().numpy().astype(np.float32)


# ─── action -> MotorControlFrame ──────────────────────────────────────────
def encode_action(action_sdk: np.ndarray,
                  ordering: JointOrdering,
                  cfg: Cfg,
                  seq_id: int) -> wf.MotorControlFrame:
    # Clip raw action if requested.
    raw = action_sdk
    if cfg.action_clip > 0:
        raw = np.clip(raw, -cfg.action_clip, cfg.action_clip)

    if cfg.use_default_offset:
        joint_target_sdk = ordering.default_pos_sdk + cfg.action_scale * raw
    else:
        joint_target_sdk = cfg.action_scale * raw

    # SDK order -> wire order, rad -> deg.
    joint_target_wire_rad = joint_target_sdk[ordering.sdk_to_wire]
    joint_target_wire_deg = joint_target_wire_rad / DEG2RAD

    frame = wf.MotorControlFrame(
        timestamp_ns=time.time_ns(),
        sequence_id=seq_id & 0xFFFFFFFFFFFFFFFF,
    )
    # Populate ALL 5 fields per motor so the wire frame is deployment-ready
    # for real-firmware PD (which reads kp/kd from each packet). dq and tau
    # are zero — the velocity policy emits position-only deltas with no
    # feed-forward velocity / torque.
    #
    # In sim, isaac_gui_plant.py ignores wire kp/kd because Isaac PD is
    # authoritative (Option A) — but emitting them keeps the bytes on the
    # wire identical to what a real-hardware sender would publish.
    use_gains = cfg.emit_pd_gains_on_wire
    for i, deg in enumerate(joint_target_wire_deg):
        m = frame.motors[i]
        m.pos = float(deg)
        m.dq = 0.0
        m.tau = 0.0
        m.kp = float(ordering.kp_wire[i]) if use_gains else 0.0
        m.kd = float(ordering.kd_wire[i]) if use_gains else 0.0
    return frame


# ─── main loop ────────────────────────────────────────────────────────────
def run(cfg: Cfg) -> None:
    ordering = build_ordering(
        cfg.g0_module,
        urdf_order_strategy=cfg.urdf_strategy,
        urdf_order_explicit=cfg.urdf_explicit,
    )
    print(f"[policy] joint order strategy: {cfg.urdf_strategy}")
    print(f"[policy] urdf order: {ordering.urdf_names}")

    sock_rc = connect_with_retry(cfg.uds_rc, cfg.connect_timeout_s,
                                 cfg.connect_retry_s, "rc")
    sock_ms = connect_with_retry(cfg.uds_ms, cfg.connect_timeout_s,
                                 cfg.connect_retry_s, "ms")
    sock_mc = connect_with_retry(cfg.uds_mc, cfg.connect_timeout_s,
                                 cfg.connect_retry_s, "mc")

    rc_reader = FrameReader(sock_rc, wf.RC_FRAME_BYTES, "rc")
    ms_reader = FrameReader(sock_ms, wf.MS_FRAME_BYTES, "ms")
    mc_writer = FrameWriter(sock_mc, "mc")

    builder = ObsBuilder(ordering, cfg)
    policy = Policy(cfg.policy_path, cfg.dry_run)

    period_s = 1.0 / cfg.rate_hz
    next_tick = time.monotonic()
    tick = 0
    seq = 0
    last_rc: Optional[wf.RobotControlFrame] = None
    last_ms: Optional[wf.MotorStateFrame] = None

    print(f"[policy] tick rate {cfg.rate_hz:.1f} Hz; dry_run={cfg.dry_run}")
    while True:
        # Drain latest frames (non-blocking).
        rc_bytes = rc_reader.drain_latest()
        if rc_bytes is not None:
            rc = wf.decode_robot_control(rc_bytes)
            if rc is not None:
                last_rc = rc
        ms_bytes = ms_reader.drain_latest()
        if ms_bytes is not None:
            ms = wf.decode_motor_state(ms_bytes)
            if ms is not None:
                last_ms = ms

        if last_ms is not None:
            obs = builder.step(last_ms, last_rc)
            action = policy.act(obs)
            builder.update_last_action(action)
            frame = encode_action(action, ordering, cfg, seq)
            mc_writer.send(frame.encode())
            seq += 1
            if tick % cfg.log_every == 0:
                cmd = builder._vel_cmd(last_rc)
                print(f"[policy] tick={tick} seq={seq} "
                      f"cmd={cmd.tolist()} "
                      f"|action|={float(np.linalg.norm(action)):.3f} "
                      f"rc_seen={last_rc is not None}")

        tick += 1
        next_tick += period_s
        sleep = next_tick - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            # Fell behind — reset cadence rather than burst-catch-up.
            next_tick = time.monotonic()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(HERE.parent / "config" / "policy_inference.toml"))
    ap.add_argument("--dry-run", action="store_true",
                    help="hold default pose, skip policy.pt")
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config), args.dry_run)
    print(f"[policy] config: {args.config}")
    try:
        run(cfg)
    except KeyboardInterrupt:
        print("\n[policy] interrupted; exit.")


if __name__ == "__main__":
    main()
