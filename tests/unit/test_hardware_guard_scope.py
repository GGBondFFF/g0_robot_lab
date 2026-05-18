from __future__ import annotations

import socket

import pytest


pytestmark = pytest.mark.unit


def test_socket_class_is_not_globally_monkeypatched_in_unit_tests():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    assert sock.__class__.__name__ == "socket"
    sock.close()
