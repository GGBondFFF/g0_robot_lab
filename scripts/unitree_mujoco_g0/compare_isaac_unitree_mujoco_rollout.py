#!/usr/bin/env python3
"""Compare Isaac, custom MuJoCo, and Unitree-style G0 rollouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sim2sim import g0_sim2sim_config as cfg  # noqa: E402

IDENTITY_KEYS = [
    "policy_path",
    "policy_filename",
    "policy_sha256",
    "checkpoint_run_folder",
    "task",
    "command",
    "steps",
    "action_dim",
    "obs_dim",
    "joint_names",
    "action_scale",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", required=True, help="Isaac golden .npz.")
    parser.add_argument("--custom", required=True, help="Current custom MuJoCo rollout .npz.")
    parser.add_argument("--unitree", required=True, help="Unitree-style G0 rollout .npz.")
    parser.add_argument("--output", required=True, help="Markdown report path.")
    return parser.parse_args()


def load(path: str) -> np.lib.npyio.NpzFile:
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Rollout file does not exist: {npz_path}")
    return np.load(npz_path, allow_pickle=True)


def _key(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    return np.asarray(data[key]) if key in data.files else None


def _root_height(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "root_height" in data.files:
        return np.asarray(data["root_height"], dtype=np.float64)
    root_pos = _key(data, "root_pos")
    if root_pos is None or root_pos.ndim < 2 or root_pos.shape[-1] < 3:
        return None
    return np.asarray(root_pos[..., 2], dtype=np.float64)


def _summary_value(data: np.lib.npyio.NpzFile, key: str) -> str:
    value = _key(data, key)
    if value is None:
        return "n/a"
    value = np.asarray(value, dtype=np.float64)
    return f"{float(np.nanmean(value)):.6g}/{float(np.nanmax(value)):.6g}"


def root_summary(data: np.lib.npyio.NpzFile) -> tuple[str, str, str, str]:
    height = _root_height(data)
    if height is None:
        return "n/a", "n/a", "n/a", "unknown"
    finite_height = height[np.isfinite(height)]
    if finite_height.size == 0:
        return "n/a", "n/a", "n/a", "unknown"
    min_h = float(np.nanmin(finite_height))
    final_h = float(finite_height[-1])
    collapsed = min_h < 0.12 or final_h < 0.12
    return f"{min_h:.6g}", f"{float(np.nanmean(finite_height)):.6g}", f"{final_h:.6g}", "yes" if collapsed else "no"


def array_error(reference: np.ndarray | None, candidate: np.ndarray | None) -> tuple[str, str]:
    if reference is None or candidate is None:
        return "n/a", "n/a"
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.ndim == 0 or candidate.ndim == 0:
        return "n/a", "n/a"
    min_len = min(reference.shape[0], candidate.shape[0])
    if min_len == 0:
        return "n/a", "n/a"
    ref = reference[:min_len]
    cand = candidate[:min_len]
    if ref.shape != cand.shape:
        trailing = min(ref.shape[-1], cand.shape[-1]) if ref.ndim > 1 and cand.ndim > 1 else None
        if trailing is None:
            return "shape", "shape"
        ref = ref[..., :trailing]
        cand = cand[..., :trailing]
    err = np.abs(ref - cand)
    return f"{float(np.nanmean(err)):.6g}", f"{float(np.nanmax(err)):.6g}"


def action_stats(data: np.lib.npyio.NpzFile) -> tuple[str, str]:
    action = _key(data, "action")
    if action is None:
        action = _key(data, "processed_action")
    if action is None:
        return "n/a", "n/a"
    action = np.asarray(action, dtype=np.float64)
    return f"{float(np.nanmax(np.abs(action))):.6g}", f"{float(np.nanmean(np.abs(action) >= 0.999)):.6g}"


def lowcmd_summary(data: np.lib.npyio.NpzFile) -> list[str]:
    rows: list[str] = []
    for key in ("lowcmd_q", "lowcmd_dq", "lowcmd_kp", "lowcmd_kd", "lowcmd_tau", "tau_cmd", "tau_cmd_clipped"):
        value = _key(data, key)
        if value is None:
            rows.append(f"| `{key}` | missing | n/a | n/a |")
            continue
        value = np.asarray(value, dtype=np.float64)
        rows.append(f"| `{key}` | {value.shape} | {float(np.nanmean(value)):.6g} | {float(np.nanmax(np.abs(value))):.6g} |")
    saturation = _key(data, "tau_saturation")
    if saturation is not None:
        rows.append(f"| `tau_saturation_ratio` | {saturation.shape} | {float(np.nanmean(saturation)):.6g} | {float(np.nanmax(saturation)):.6g} |")
    return rows


def obs_term_rows(isaac: np.lib.npyio.NpzFile, unitree: np.lib.npyio.NpzFile) -> list[str]:
    if "obs" not in isaac.files or "obs" not in unitree.files:
        return ["| obs | missing | n/a | n/a |"]
    isaac_obs = np.asarray(isaac["obs"])
    unitree_obs = np.asarray(unitree["obs"])
    min_len = min(isaac_obs.shape[0], unitree_obs.shape[0])
    rows: list[str] = []
    for term in cfg.POLICY_OBS_TERMS:
        left = np.asarray([cfg.split_policy_observation(obs)[term] for obs in isaac_obs[:min_len]])
        right = np.asarray([cfg.split_policy_observation(obs)[term] for obs in unitree_obs[:min_len]])
        err = np.abs(left - right)
        rows.append(f"| `{term}` | {left.shape} | {float(np.nanmean(err)):.6g} | {float(np.nanmax(err)):.6g} |")
    return rows


def metadata_value(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        return "missing"
    value = np.asarray(data[key])
    if key == "command" and value.ndim > 1 and value.shape[-1] == 3:
        first = value.reshape(-1, 3)[0]
        if np.allclose(value.reshape(-1, 3), first[None, :]):
            return str(first.tolist())
        return f"{value.shape}, first={first.tolist()}"
    if value.ndim == 0:
        return str(value.item())
    return str(value.tolist())


def main() -> int:
    args = parse_args()
    isaac = load(args.isaac)
    custom = load(args.custom)
    unitree = load(args.unitree)

    datasets = [("Isaac", isaac), ("custom MuJoCo", custom), ("unitree-style G0", unitree)]
    lines = [
        "# Isaac / Custom MuJoCo / Unitree-Style G0 Compare",
        "",
        f"- Isaac: `{args.isaac}`",
        f"- custom MuJoCo: `{args.custom}`",
        f"- unitree-style G0: `{args.unitree}`",
        "",
        "## Policy Identity",
        "",
        "| key | Isaac | custom MuJoCo | unitree-style G0 |",
        "| --- | --- | --- | --- |",
        *[
            f"| `{key}` | `{metadata_value(isaac, key)}` | `{metadata_value(custom, key)}` | `{metadata_value(unitree, key)}` |"
            for key in IDENTITY_KEYS
        ],
        "",
        "## Rollout Summary",
        "",
        "| rollout | root min | root mean | root final | collapsed | action max | action saturation | foot contacts max | torque saturation |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, data in datasets:
        min_h, mean_h, final_h, collapsed = root_summary(data)
        action_max, action_sat = action_stats(data)
        foot_contacts = _key(data, "foot_ground_contact_count")
        foot_contact_text = "n/a" if foot_contacts is None else f"{int(np.nanmax(foot_contacts))}"
        tau_sat = _key(data, "tau_saturation")
        tau_sat_text = "n/a" if tau_sat is None else f"{float(np.nanmean(tau_sat)):.6g}"
        lines.append(
            f"| {name} | {min_h} | {mean_h} | {final_h} | {collapsed} | {action_max} | {action_sat} | {foot_contact_text} | {tau_sat_text} |"
        )

    lines.extend(
        [
            "",
            "## Error Against Isaac",
            "",
            "| key | custom mean | custom max | unitree mean | unitree max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in ("joint_pos", "joint_vel", "target_joint_pos", "action", "projected_gravity", "base_ang_vel"):
        custom_mean, custom_max = array_error(_key(isaac, key), _key(custom, key))
        unitree_mean, unitree_max = array_error(_key(isaac, key), _key(unitree, key))
        lines.append(f"| `{key}` | {custom_mean} | {custom_max} | {unitree_mean} | {unitree_max} |")

    lines.extend(
        [
            "",
            "## Unitree LowCmd Fields",
            "",
            "| key | shape | mean | max abs |",
            "| --- | --- | ---: | ---: |",
            *lowcmd_summary(unitree),
            "",
            "## Unitree Observation Term Error Against Isaac",
            "",
            "| term | compared shape | mean abs error | max abs error |",
            "| --- | --- | ---: | ---: |",
            *obs_term_rows(isaac, unitree),
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved three-way compare report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
