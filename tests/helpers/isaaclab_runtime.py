from __future__ import annotations

from typing import Any


def as_list(value: Any):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.cpu().tolist()
    if isinstance(value, range):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return value


def get_robot_joint_names(robot: Any) -> list[str]:
    joint_names = getattr(getattr(robot, "data", None), "joint_names", None)
    if joint_names is None:
        joint_names = getattr(robot, "joint_names", None)
    if joint_names is None:
        raise RuntimeError("Cannot find robot joint names.")
    return [str(name) for name in joint_names]


def get_action_terms(action_manager: Any):
    terms = getattr(action_manager, "_terms", None)
    if terms is None:
        terms = getattr(action_manager, "terms", None)
    if callable(terms):
        terms = terms()
    if not isinstance(terms, dict):
        raise RuntimeError("Cannot inspect action manager terms.")
    return terms


def resolve_action_joint_order(action_manager: Any, robot_joint_names: list[str]) -> list[str]:
    action_joint_order: list[str] = []
    for term in get_action_terms(action_manager).values():
        joint_ids = getattr(term, "_joint_ids", None)
        if joint_ids is None:
            joint_ids = getattr(term, "joint_ids", None)
        joint_names = getattr(term, "_joint_names", None)
        if joint_names is None:
            joint_names = getattr(term, "joint_names", None)

        joint_ids = as_list(joint_ids)
        joint_names = as_list(joint_names)
        if joint_names is not None:
            resolved = [str(name) for name in joint_names]
        elif isinstance(joint_ids, slice):
            resolved = robot_joint_names[joint_ids]
        elif joint_ids is not None:
            resolved = [robot_joint_names[int(index)] for index in joint_ids]
        else:
            resolved = []
        action_joint_order.extend(resolved)
    return action_joint_order
