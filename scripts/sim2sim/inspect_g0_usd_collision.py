#!/usr/bin/env python3
"""Inspect G0 USD/PhysX foot collision fidelity.

Run through Isaac Lab so USD and PhysX schemas are available:

```
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/inspect_g0_usd_collision.py --headless
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USD = REPO_ROOT / "source" / "g0_robot_lab" / "g0_robot_lab" / "assets" / "robots" / "g0" / "usd" / "g0.usd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", default=str(DEFAULT_USD), help="Path to g0.usd.")
    parser.add_argument("--headless", action="store_true", help="Start Isaac Sim headless.")
    return parser.parse_args()


def _api_names(prim) -> list[str]:
    return [str(name) for name in prim.GetAppliedSchemas()]


def _attr_value(prim, name: str):
    attr = prim.GetAttribute(name)
    if attr and attr.HasAuthoredValueOpinion():
        return attr.Get()
    return None


def log(message: str = "") -> None:
    sys.__stdout__.write(f"{message}\n")
    sys.__stdout__.flush()


def _prim_summary(prim) -> dict[str, object]:
    return {
        "path": str(prim.GetPath()),
        "type": prim.GetTypeName(),
        "apis": _api_names(prim),
        "collision_enabled": _attr_value(prim, "physics:collisionEnabled"),
        "approximation": _attr_value(prim, "physics:approximation"),
        "contact_offset": _attr_value(prim, "physxCollision:contactOffset"),
        "rest_offset": _attr_value(prim, "physxCollision:restOffset"),
        "mesh_asset": _attr_value(prim, "inputs:file") or _attr_value(prim, "file"),
    }


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app

    try:
        from pxr import Usd

        usd_path = Path(args.usd)
        if not usd_path.exists():
            print(f"ERROR: USD does not exist: {usd_path}", file=sys.stderr)
            return 2
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            print(f"ERROR: failed to open USD: {usd_path}", file=sys.stderr)
            return 2

        log("G0 USD/PhysX collision inspection")
        log(f"usd_path: {usd_path}")
        log("")
        for foot_name in ("l_foot_link", "r_foot_link"):
            log(f"foot_link: {foot_name}")
            foot_root = next((prim for prim in stage.Traverse() if prim.GetName() == foot_name), None)
            if foot_root is None:
                log("  prims: []")
                continue
            foot_prims = list(Usd.PrimRange(foot_root))
            for prim in foot_prims:
                summary = _prim_summary(prim)
                is_collision = any("Collision" in api for api in summary["apis"]) or summary["collision_enabled"] is not None
                maybe_mesh = summary["type"] in {"Mesh", "Xform", "Scope"} or is_collision
                if not maybe_mesh:
                    continue
                log(f"  prim: {summary['path']}")
                log(f"    type: {summary['type']}")
                log(f"    apis: {summary['apis']}")
                log(f"    has_collision_api: {any('CollisionAPI' in api for api in summary['apis'])}")
                log(f"    has_mesh_collision_api: {any('MeshCollisionAPI' in api for api in summary['apis'])}")
                log(f"    collision_enabled: {summary['collision_enabled']}")
                log(f"    approximation: {summary['approximation']}")
                log(f"    contact_offset: {summary['contact_offset']}")
                log(f"    rest_offset: {summary['rest_offset']}")
                log(f"    mesh_asset: {summary['mesh_asset']}")
            log("")

        log("notes:")
        log("  - URDF converter config uses collider_type: convex_hull.")
        log("  - URDF converter config uses collision_from_visuals: false.")
        log("  - URDF converter config uses self_collision: false.")
        log("  - If no extra foot collision prim is listed, the USD appears to use converted mesh collision, not an added sole box.")
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
