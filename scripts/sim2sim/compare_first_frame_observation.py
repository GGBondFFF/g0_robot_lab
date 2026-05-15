#!/usr/bin/env python3
"""Compare Isaac and MuJoCo first-frame observation terms.

This report is intentionally term-by-term instead of only comparing flattened
policy observations. It separates interface-aligned terms from terms that still
depend on dynamics, frame conventions, or command generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from scripts.sim2sim import g0_sim2sim_config as cfg
except ModuleNotFoundError:
    import g0_sim2sim_config as cfg


TERM_SLICES = {
    "base_ang_vel_scaled": slice(0, 3),
    "projected_gravity": slice(3, 6),
    "velocity_commands": slice(6, 9),
    "joint_pos_rel": slice(9, 31),
    "joint_vel_rel_scaled": slice(31, 53),
    "last_action": slice(53, 75),
    "gait_phase": slice(75, 77),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", default="logs/sim2sim/isaac_zero_action_golden_io.npz", help="Isaac golden .npz.")
    parser.add_argument("--mujoco", default="logs/sim2sim/mujoco_zero_action_rollout.npz", help="MuJoCo rollout .npz.")
    parser.add_argument("--output", default="logs/sim2sim/first_frame_observation_report.md", help="Markdown output.")
    return parser.parse_args()


def first(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    if value.ndim == 0:
        return value
    return np.asarray(value[0])


def compare(a: np.ndarray | None, b: np.ndarray | None) -> tuple[str, str, str]:
    if a is None or b is None:
        missing = []
        if a is None:
            missing.append("Isaac")
        if b is None:
            missing.append("MuJoCo")
        return f"missing in {', '.join(missing)}", "n/a", "n/a"
    if np.asarray(a).shape != np.asarray(b).shape:
        return f"shape mismatch {np.asarray(a).shape} vs {np.asarray(b).shape}", "n/a", "n/a"
    if not np.issubdtype(np.asarray(a).dtype, np.number) or not np.issubdtype(np.asarray(b).dtype, np.number):
        return ("ok" if np.array_equal(a, b) else "value mismatch"), "n/a", "n/a"
    err = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return "ok", f"{float(np.nanmean(err)):.6g}", f"{float(np.nanmax(err)):.6g}"


def term_from_obs(obs: np.ndarray | None, term: str) -> np.ndarray | None:
    if obs is None:
        return None
    obs = np.asarray(obs)
    width = cfg.get_single_frame_observation_dim()
    if obs.shape[-1] == width:
        return obs[TERM_SLICES[term]]
    if obs.shape[-1] != cfg.get_policy_observation_dim():
        return None
    offset = 0
    for name, term_slice in TERM_SLICES.items():
        dim = term_slice.stop - term_slice.start
        term_width = dim * cfg.POLICY_HISTORY_LENGTH
        if name == term:
            term_history = obs[offset : offset + term_width]
            return term_history[-dim:]
        offset += term_width
    return None


def derived_terms(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray | None]:
    default = first(data, "default_joint_pos")
    action = first(data, "action")
    joint_pos = first(data, "joint_pos")
    joint_vel = first(data, "joint_vel")
    command = first(data, "command")
    obs = first(data, "obs")
    terms: dict[str, np.ndarray | None] = {
        "joint_names": first(data, "joint_names"),
        "default_joint_pos": default,
        "action": action,
        "target_joint_pos": first(data, "target_joint_pos"),
        "command": command,
        "joint_pos_rel_from_state": None if joint_pos is None or default is None else joint_pos - default,
        "joint_vel_rel_scaled_from_state": None if joint_vel is None else joint_vel * 0.05,
        "last_action_from_rollout": action,
        "base_ang_vel": first(data, "base_ang_vel"),
        "projected_gravity": first(data, "projected_gravity"),
        "obs_shape": None if "obs" not in data.files else np.asarray(np.asarray(data["obs"]).shape),
    }
    for term in TERM_SLICES:
        terms[f"obs_{term}"] = term_from_obs(obs, term)
    return terms


def main() -> int:
    args = parse_args()
    isaac_path = Path(args.isaac)
    mujoco_path = Path(args.mujoco)
    if not isaac_path.exists():
        raise FileNotFoundError(f"Isaac golden file does not exist: {isaac_path}")
    if not mujoco_path.exists():
        raise FileNotFoundError(f"MuJoCo rollout file does not exist: {mujoco_path}")

    isaac = np.load(isaac_path, allow_pickle=True)
    mujoco = np.load(mujoco_path, allow_pickle=True)
    isaac_terms = derived_terms(isaac)
    mujoco_terms = derived_terms(mujoco)

    rows = [
        "joint_names",
        "default_joint_pos",
        "action",
        "target_joint_pos",
        "joint_pos_rel_from_state",
        "joint_vel_rel_scaled_from_state",
        "command",
        "last_action_from_rollout",
        "base_ang_vel",
        "projected_gravity",
        "obs_shape",
        "obs_base_ang_vel_scaled",
        "obs_projected_gravity",
        "obs_velocity_commands",
        "obs_joint_pos_rel",
        "obs_joint_vel_rel_scaled",
        "obs_last_action",
        "obs_gait_phase",
    ]

    lines = [
        "# First-Frame Observation Term Report",
        "",
        f"- Isaac file: `{isaac_path}`",
        f"- MuJoCo file: `{mujoco_path}`",
        "",
        "| term | status | mean abs error | max abs error | note |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for key in rows:
        status, mean_abs, max_abs = compare(isaac_terms.get(key), mujoco_terms.get(key))
        note = ""
        if key in {"base_ang_vel", "projected_gravity", "obs_base_ang_vel_scaled", "obs_projected_gravity"}:
            note = "frame convention still under validation"
        elif key in {"joint_pos_rel_from_state", "joint_vel_rel_scaled_from_state", "obs_joint_pos_rel", "obs_joint_vel_rel_scaled"}:
            note = "depends on model dynamics and first-frame timing"
        elif key in {"command", "obs_velocity_commands"}:
            note = "Isaac command generator may differ from explicit MuJoCo command"
        elif key == "obs_gait_phase":
            note = "depends on exact sampled frame time/history"
        lines.append(f"| `{key}` | {status} | {mean_abs} | {max_abs} | {note} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `action` and `target_joint_pos` should be exact in zero-action tests.",
            "- State-derived joint/root differences can come from the current MuJoCo model dynamics and from first-frame timing.",
            "- `projected_gravity` and `base_ang_vel` are now recorded on the MuJoCo side, but their frame convention still needs dedicated validation.",
            "- This report should be used before interpreting any policy rollout quality.",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved first-frame observation report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
