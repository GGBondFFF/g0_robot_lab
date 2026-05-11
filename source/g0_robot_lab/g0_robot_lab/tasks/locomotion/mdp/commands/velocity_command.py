from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command config with curriculum limit ranges.

    This extends Isaac Lab's built-in UniformVelocityCommandCfg with an
    additional `limit_ranges` field. The normal `ranges` field defines the
    currently sampled command range, while `limit_ranges` defines the maximum
    range that curriculum is allowed to expand to.

    The actual curriculum logic will be implemented later in curriculums.py.
    """

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING