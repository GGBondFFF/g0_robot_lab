"""End-to-end UDS integration test (no Isaac Lab GUI).

For each of the 4 canonical motor ids (1, 7, 14, 21) this test:
  1. spawns dds_to_uds_bridge on a private UDS path + virtual DDS topic,
  2. spawns single_joint_sender (--dry-run false) publishing a PD command,
  3. reads UDS frames via a raw SOCK_STREAM client,
  4. asserts only the target slot is non-zero, with the right sign and value.

Skips if the C++ binaries aren't built.

Run::

    /home/lz/miniconda3/envs/g0_isaaclab/bin/python -m pytest \
        test/single_joint_test/python/test_uds_integration.py -v
"""
from __future__ import annotations

import math
import os
import socket
import struct
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import joint_mapping as jm

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.normpath(os.path.join(_HERE, "..", "cpp", "build"))
_BRIDGE = os.path.join(_BUILD, "dds_to_uds_bridge")
_SENDER = os.path.join(_BUILD, "single_joint_sender")
_FRAME_BYTES = 464
_FRAME_MAGIC = 0x47305349


def _binaries_present() -> bool:
    return os.access(_BRIDGE, os.X_OK) and os.access(_SENDER, os.X_OK)


pytestmark = pytest.mark.skipif(
    not _binaries_present(),
    reason=f"C++ binaries not built: build with `cd cpp && mkdir -p build && cd build && cmake .. && make`",
)


def _decode_frame(buf: bytes):
    assert len(buf) == _FRAME_BYTES
    magic, ver, mc = struct.unpack_from("<IHH", buf, 0)
    assert magic == _FRAME_MAGIC, f"bad magic {hex(magic)}"
    assert ver == 1
    assert mc == 22
    ts_ns, seq = struct.unpack_from("<QQ", buf, 8)
    motors = []
    for i in range(22):
        motors.append(struct.unpack_from("<5f", buf, 24 + i * 20))
    return ts_ns, seq, motors


def _read_n_frames(sock_path: str, n: int, timeout_s: float):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_s)
    sock.connect(sock_path)
    buf = b""
    frames = []
    deadline = time.time() + timeout_s
    while len(frames) < n and time.time() < deadline:
        try:
            chunk = sock.recv(8192)
        except socket.timeout:
            break
        if not chunk:
            break
        buf += chunk
        while len(buf) >= _FRAME_BYTES:
            frames.append(_decode_frame(buf[:_FRAME_BYTES]))
            buf = buf[_FRAME_BYTES:]
    sock.close()
    return frames


@pytest.fixture
def bridge(tmp_path):
    uds = str(tmp_path / "uds.sock")
    topic = "g0_sim/motor_control_virtual_pytest"
    log = open(tmp_path / "bridge.log", "w")
    proc = subprocess.Popen(
        [_BRIDGE, "--uds", uds, "--topic", topic],
        stdout=log, stderr=subprocess.STDOUT,
    )
    # Wait for the socket to appear.
    deadline = time.time() + 5.0
    while time.time() < deadline and not os.path.exists(uds):
        if proc.poll() is not None:
            log.close()
            pytest.fail(f"bridge exited early: see {tmp_path}/bridge.log")
        time.sleep(0.1)
    # Give DDS discovery a moment.
    time.sleep(0.5)
    yield {"uds": uds, "topic": topic, "proc": proc, "log_path": str(tmp_path / "bridge.log")}
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


@pytest.mark.parametrize(
    "motor_id,rhr_pos_deg,expected_sign",
    [
        (1,  -10.0, +1),  # sim_sign(1) = -1; -1 * -10 = +10
        (7,  +10.0, +1),  # sim_sign(7) = +1; +1 * +10 = +10
        (14, +10.0, -1),  # sim_sign(14)= -1; -1 * +10 = -10
        (21, -10.0, -1),  # sim_sign(21)= +1; +1 * -10 = -10
    ],
)
def test_end_to_end_single_joint(bridge, motor_id, rhr_pos_deg, expected_sign):
    """Sender publishes a 1-motor PD command (CLI in degrees, matches real
    motor wire); bridge forwards over UDS; we verify the wire bytes carry
    the right sign and value, and that all other 21 slots are exactly zero."""
    sender = subprocess.Popen(
        [_SENDER,
         "--motor-id", str(motor_id),
         "--pos", str(rhr_pos_deg),
         "--kp", "30",
         "--kd", "1.5",
         "--rate-hz", "50",
         "--duration-sec", "2.0",
         "--dry-run", "false",
         "--topic", bridge["topic"]],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    try:
        frames = _read_n_frames(bridge["uds"], n=5, timeout_s=6.0)
    finally:
        sender.terminate()
        try:
            sender.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            sender.kill()

    assert frames, f"received no UDS frames; bridge log: {bridge['log_path']}"

    target_slot = motor_id - 1
    sim_sign = jm.sim_sign_for_motor(motor_id)
    expected_wire_deg = sim_sign * rhr_pos_deg
    assert math.copysign(1.0, expected_wire_deg) == float(expected_sign)

    # Use the last frame (most settled).
    _, _, motors = frames[-1]
    for slot, (pos, dq, kp, kd, tau) in enumerate(motors):
        if slot == target_slot:
            assert pos == pytest.approx(expected_wire_deg, abs=1e-3), (
                f"slot {slot} pos={pos} expected {expected_wire_deg}")
            assert kp == pytest.approx(30.0, abs=1e-3)
            assert kd == pytest.approx(1.5, abs=1e-3)
            assert dq == pytest.approx(0.0, abs=1e-5)
            assert tau == pytest.approx(0.0, abs=1e-5)
        else:
            assert pos == 0.0, f"slot {slot} pos must be 0, got {pos}"
            assert dq == 0.0
            assert kp == 0.0
            assert kd == 0.0
            assert tau == 0.0


def test_sender_refuses_real_hw_topic():
    proc = subprocess.run(
        [_SENDER, "--motor-id", "1", "--pos", "0.0",
         "--dry-run", "false", "--topic", "mc/motor_control"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=5.0,
    )
    assert proc.returncode == 4, f"expected exit 4 for mc/ topic, got {proc.returncode}"


def test_sender_refuses_non_sim_topic_for_publish():
    proc = subprocess.run(
        [_SENDER, "--motor-id", "1", "--pos", "0.0",
         "--dry-run", "false", "--topic", "random/topic"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=5.0,
    )
    # Should refuse: non-mc but also non-g0_sim/ prefix is not allowed for publish.
    assert proc.returncode == 4, f"expected exit 4 for non-g0_sim topic, got {proc.returncode}"


def test_bridge_refuses_real_hw_topic(tmp_path):
    proc = subprocess.run(
        [_BRIDGE, "--topic", "mc/motor_control",
         "--uds", str(tmp_path / "x.sock")],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=5.0,
    )
    assert proc.returncode == 4, f"expected exit 4 for mc/ topic, got {proc.returncode}"
