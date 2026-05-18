from __future__ import annotations

import os
import socket

import pytest


pytestmark = [pytest.mark.deployment_dryrun, pytest.mark.hardware_forbidden]


def test_hardware_opt_in_is_disabled_in_dryrun_tests(dryrun_required):
    assert os.environ["G0_ALLOW_HARDWARE"] == "0"


def test_socket_send_is_blocked_only_for_hardware_forbidden_tests(dryrun_required):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with pytest.raises(AssertionError, match="Real socket.sendto is forbidden"):
        sock.sendto(b"blocked", ("127.0.0.1", 9))
    sock.close()
