from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert G0 URDF to USD.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    project_root = Path(__file__).resolve().parents[2]

    asset_root = project_root / "source/g0_robot_lab/g0_robot_lab/assets/robots/g0"
    urdf_path = asset_root / "urdf/g0.urdf"
    usd_dir = asset_root / "usd"

    usd_dir.mkdir(parents=True, exist_ok=True)

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(usd_dir),
        usd_file_name="g0.usd",
        fix_base=False,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
            target_type="position",
        ),
    )

    converter = UrdfConverter(cfg)

    print("[INFO] G0 URDF converted successfully.")
    print(f"[INFO] URDF: {urdf_path}")
    print(f"[INFO] USD : {converter.usd_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()