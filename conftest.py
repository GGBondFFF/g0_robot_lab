"""Repo-root pytest conftest.

Ensures the repository root is on ``sys.path`` so tests can ``import deploy``
under a bare ``pytest`` invocation (not just ``python -m pytest``). The
deployment soak test inserts the root itself; the unit tests rely on this.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
