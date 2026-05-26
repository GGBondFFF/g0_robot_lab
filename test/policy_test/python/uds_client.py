"""Tiny UDS-stream client helpers for the policy_test bridges.

Each bridge is a SOCK_STREAM listener; the Python policy is the client. The
read socket is "latest-frame-wins" (we drain everything available and only
keep the newest); the write socket is a plain blocking send.

We deliberately avoid asyncio — the inference loop is a single-threaded
50 Hz tick, and asyncio overhead at that rate dwarfs the IO.
"""
from __future__ import annotations

import errno
import socket
import time
from typing import Callable, Optional


def connect_with_retry(path: str, timeout_s: float, retry_s: float,
                       label: str) -> socket.socket:
    """Block until we can connect to a UDS server, raising after timeout_s."""
    deadline = time.monotonic() + timeout_s
    last_err: Optional[BaseException] = None
    while time.monotonic() < deadline:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(path)
            print(f"[uds:{label}] connected: {path}")
            return s
        except OSError as e:
            last_err = e
            s.close()
            time.sleep(retry_s)
    raise TimeoutError(
        f"timed out connecting to {path} ({label}): {last_err}")


class FrameReader:
    """Reads exactly `frame_size` bytes per frame from a non-blocking UDS.

    drain_latest() returns the most recent complete frame seen since the
    previous call (older frames are discarded), or None if nothing new.
    """

    def __init__(self, sock: socket.socket, frame_size: int, label: str):
        sock.setblocking(False)
        self._sock = sock
        self._frame_size = frame_size
        self._label = label
        self._buf = bytearray()
        self._last_complete: Optional[bytes] = None

    def drain_latest(self) -> Optional[bytes]:
        # Pull everything pending.
        try:
            while True:
                chunk = self._sock.recv(self._frame_size * 8)
                if not chunk:
                    print(f"[uds:{self._label}] peer closed")
                    raise ConnectionError(f"uds {self._label} closed")
                self._buf.extend(chunk)
        except BlockingIOError:
            pass
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                raise

        # Consume whole frames; keep only the last one.
        n_consumed = 0
        while len(self._buf) - n_consumed >= self._frame_size:
            self._last_complete = bytes(
                self._buf[n_consumed : n_consumed + self._frame_size])
            n_consumed += self._frame_size
        if n_consumed:
            del self._buf[:n_consumed]
        out, self._last_complete = self._last_complete, None
        return out


class FrameWriter:
    """Tiny blocking writer — bridge consumer drains promptly."""

    def __init__(self, sock: socket.socket, label: str):
        sock.setblocking(True)
        self._sock = sock
        self._label = label

    def send(self, buf: bytes) -> None:
        try:
            self._sock.sendall(buf)
        except (BrokenPipeError, ConnectionResetError) as e:
            raise ConnectionError(f"uds {self._label} broken: {e}") from e
