"""Codegen: resolve G0_CFG stiffness/damping into a sender-side C++ table.

Writes cpp/include/pd_gains_generated.hpp. Re-run after changing g0.py.

Wire units (match the sender's --kp / --kd CLI flags and the receiver's
deg-domain PD law):
  kp : N*m / deg            = G0_CFG stiffness  (N*m/rad)        * (pi/180)
  kd : N*m / (deg/s)        = G0_CFG damping    (N*m/(rad/s))    * (pi/180)

Usage:
  python test/single_joint_test/python/codegen_pd_gains.py
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # .../g0_robot_lab/test/single_joint_test/python -> repo root


def _install_isaaclab_stubs() -> None:
    """Stub the isaaclab modules g0.py imports so the codegen can run without
    an IsaacLab install. We only need plain attribute access on G0_CFG; none of
    these stubs need to be functional beyond storing kwargs.
    """
    import types

    class _Bag:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

        def replace(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
            return self

    def _bag_factory(*_a, **kw):
        return _Bag(**kw)

    def _passthrough(obj):
        return obj

    mods = {}

    # isaaclab.sim - cfg builders used by g0.py
    sim_mod = types.ModuleType("isaaclab.sim")
    for name in (
        "UsdFileCfg",
        "RigidBodyPropertiesCfg",
        "ArticulationRootPropertiesCfg",
        "SimulationCfg",
        "GroundPlaneCfg",
        "DomeLightCfg",
    ):
        setattr(sim_mod, name, _bag_factory)
    mods["isaaclab"] = types.ModuleType("isaaclab")
    mods["isaaclab.sim"] = sim_mod
    mods["isaaclab"].sim = sim_mod

    # isaaclab.actuators.ImplicitActuatorCfg
    act_mod = types.ModuleType("isaaclab.actuators")
    act_mod.ImplicitActuatorCfg = _bag_factory
    mods["isaaclab.actuators"] = act_mod
    mods["isaaclab"].actuators = act_mod

    # isaaclab.assets.ArticulationCfg with nested InitialStateCfg
    assets_mod = types.ModuleType("isaaclab.assets")

    class _ArticulationCfg(_Bag):
        InitialStateCfg = staticmethod(_bag_factory)

    assets_mod.ArticulationCfg = _ArticulationCfg
    mods["isaaclab.assets"] = assets_mod
    mods["isaaclab"].assets = assets_mod

    # isaaclab.utils.configclass — no-op decorator
    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.configclass = _passthrough
    mods["isaaclab.utils"] = utils_mod
    mods["isaaclab"].utils = utils_mod

    for name, mod in mods.items():
        sys.modules.setdefault(name, mod)


def _load_g0_cfg():
    """Import g0.py without triggering g0_robot_lab/__init__.py (which pulls in
    isaaclab_tasks). We only need the leaf module's G0_CFG / actuators dict.
    """
    _install_isaaclab_stubs()

    import importlib.util

    g0_path = (
        REPO_ROOT
        / "source"
        / "g0_robot_lab"
        / "g0_robot_lab"
        / "assets"
        / "robots"
        / "g0"
        / "g0.py"
    )
    actuators_path = g0_path.parent / "g0_actuators.py"

    spec_a = importlib.util.spec_from_file_location("g0_actuators_inline", actuators_path)
    mod_a = importlib.util.module_from_spec(spec_a)
    sys.modules["g0_actuators_inline"] = mod_a
    spec_a.loader.exec_module(mod_a)
    # g0.py does `from . import g0_actuators`; satisfy that import.
    sys.modules.setdefault("g0_robot_lab.assets.robots.g0.g0_actuators", mod_a)

    # Stub out the relative-import parent chain so `from . import g0_actuators`
    # in g0.py resolves to our pre-loaded module.
    import types
    pkg = types.ModuleType("g0_robot_lab_inline_pkg")
    pkg.__path__ = []  # mark as package
    pkg.g0_actuators = mod_a
    sys.modules["g0_robot_lab_inline_pkg"] = pkg
    sys.modules["g0_robot_lab_inline_pkg.g0_actuators"] = mod_a

    spec_g = importlib.util.spec_from_file_location(
        "g0_robot_lab_inline_pkg.g0", g0_path
    )
    mod_g = importlib.util.module_from_spec(spec_g)
    sys.modules["g0_robot_lab_inline_pkg.g0"] = mod_g
    spec_g.loader.exec_module(mod_g)
    return mod_g.G0_CFG, mod_g.G0_DEFAULT_JOINT_POS


G0_CFG, G0_DEFAULT_JOINT_POS = _load_g0_cfg()

# Motor id (1..22) -> joint name. MUST stay in sync with joint_mapping.hpp.
MOTOR_ID_TO_JOINT = [
    "waist_yaw_joint", "waist_roll_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint", "l_elbow_pitch_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint", "r_elbow_pitch_joint",
    "l_hip_pitch_joint", "l_hip_roll_joint",
    "l_hip_yaw_joint", "l_knee_pitch_joint",
    "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint",
    "r_hip_yaw_joint", "r_knee_pitch_joint",
    "r_ankle_pitch_joint", "r_ankle_roll_joint",
]
assert len(MOTOR_ID_TO_JOINT) == 22

RAD_PER_DEG = math.pi / 180.0


def joint_in_group(joint_name: str, exprs) -> bool:
    for expr in exprs:
        if re.fullmatch(expr, joint_name) is not None:
            return True
    return False


def resolve_gain(joint_name: str, gain_spec) -> float:
    if isinstance(gain_spec, (int, float)):
        return float(gain_spec)
    if isinstance(gain_spec, dict):
        for pattern, value in gain_spec.items():
            if re.fullmatch(pattern, joint_name) is not None:
                return float(value)
        raise KeyError(f"no stiffness/damping pattern matched joint '{joint_name}'")
    raise TypeError(f"unsupported gain spec type: {type(gain_spec)!r}")


def main() -> None:
    kp_rad = [None] * 22
    kd_rad = [None] * 22
    source_group = [None] * 22

    for group_name, act in G0_CFG.actuators.items():
        for mid, jname in enumerate(MOTOR_ID_TO_JOINT, start=1):
            if not joint_in_group(jname, act.joint_names_expr):
                continue
            if kp_rad[mid - 1] is not None:
                raise RuntimeError(
                    f"joint '{jname}' matched by multiple actuator groups: "
                    f"{source_group[mid - 1]!r} and {group_name!r}"
                )
            kp_rad[mid - 1] = resolve_gain(jname, act.stiffness)
            kd_rad[mid - 1] = resolve_gain(jname, act.damping)
            source_group[mid - 1] = group_name

    missing = [MOTOR_ID_TO_JOINT[i] for i, v in enumerate(kp_rad) if v is None]
    if missing:
        raise RuntimeError(f"no actuator group covers: {missing}")

    # Default standing pose, in degrees, indexed by motor_id - 1.
    # G0_DEFAULT_JOINT_POS in g0.py is keyed by joint name and stored in
    # radians; we convert to degrees here so the sender can write it straight
    # to the wire (which is deg-native).
    default_pos_deg = [None] * 22
    for mid, jname in enumerate(MOTOR_ID_TO_JOINT, start=1):
        if jname not in G0_DEFAULT_JOINT_POS:
            raise RuntimeError(f"G0_DEFAULT_JOINT_POS missing joint '{jname}'")
        default_pos_deg[mid - 1] = float(G0_DEFAULT_JOINT_POS[jname]) / RAD_PER_DEG

    out = HERE.parent / "cpp" / "include" / "pd_gains_generated.hpp"
    lines = [
        "// AUTO-GENERATED by test/single_joint_test/python/codegen_pd_gains.py.",
        "// Do not edit by hand. Re-run the codegen script after changing g0.py.",
        "//",
        "// Source: G0_CFG.actuators stiffness/damping and G0_DEFAULT_JOINT_POS,",
        "// converted to wire units:",
        "//   kp           : N*m / deg",
        "//   kd           : N*m / (deg/s)",
        "//   default_pos  : deg",
        "#ifndef G0_SJT_PD_GAINS_GENERATED_HPP",
        "#define G0_SJT_PD_GAINS_GENERATED_HPP",
        "",
        "#include <array>",
        "",
        "namespace g0_sjt {",
        "",
        "struct PdGain { float kp; float kd; };",
        "",
        "static constexpr std::array<PdGain, 22> kPdGainsByMotorId = {{",
    ]
    for mid, jname in enumerate(MOTOR_ID_TO_JOINT, start=1):
        kp_deg = kp_rad[mid - 1] * RAD_PER_DEG
        kd_deg = kd_rad[mid - 1] * RAD_PER_DEG
        lines.append(
            f"    {{ {kp_deg:.8f}f, {kd_deg:.8f}f }},  "
            f"// id={mid:2d}  {jname}  ({source_group[mid - 1]})"
        )
    lines += [
        "}};",
        "",
        "// G0_DEFAULT_JOINT_POS in degrees, indexed by (motor_id - 1).",
        "// Sender uses this when --baseline default so non-target slots hold the",
        "// standing pose instead of being commanded to zero.",
        "static constexpr std::array<float, 22> kDefaultPosByMotorIdDeg = {{",
    ]
    for mid, jname in enumerate(MOTOR_ID_TO_JOINT, start=1):
        lines.append(
            f"    {default_pos_deg[mid - 1]:+.8f}f,  // id={mid:2d}  {jname}"
        )
    lines += [
        "}};",
        "",
        "}  // namespace g0_sjt",
        "",
        "#endif  // G0_SJT_PD_GAINS_GENERATED_HPP",
        "",
    ]
    out.write_text("\n".join(lines))
    print(f"wrote {out}  (22 gain entries + 22 default-pose entries)")


if __name__ == "__main__":
    main()
