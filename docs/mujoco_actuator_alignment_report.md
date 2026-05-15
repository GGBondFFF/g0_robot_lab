# G0 MuJoCo Actuator Alignment Report

## Context

- Branch: `structure/mujoco-sim2sim-layout`
- Test time: `2026-05-15 15:48:50 CST`
- Goal: align the first-pass MuJoCo actuator parameters with Isaac Lab `G0_CFG` without changing formal foot collision geometry.

This pass does not tune parameters for stability. It maps the parameters that already exist in the Isaac G0 configuration into the generated MuJoCo model and records the remaining non-equivalent items.

## Source Parameters

The alignment is generated from:

- `G0_JOINT_SDK_NAMES`
- `G0_RIGHT_ANGLE_SERVO_JOINT_NAMES`
- `G0_STANDARD_SERVO_JOINT_NAMES`
- `G0_CFG.actuators`
- `g0_actuators.py`

The helper in `scripts/sim2sim/g0_sim2sim_config.py` parses the G0 source file instead of importing `G0_CFG` directly. Direct import requires Isaac Sim/pxr, while the sim2sim reports need to run in ordinary Python.

## Mapping

Current MuJoCo mapping:

| Isaac / hardware parameter | MuJoCo field | Status |
| --- | --- | --- |
| `stiffness` | position actuator `kp` | mapped directly |
| `effort_limit_sim` | position actuator `forcerange` | mapped as symmetric `[-limit, limit]` |
| `damping` | joint `damping` | first-pass approximation |
| `armature` | joint `armature` | mapped directly |
| URDF joint limits | position actuator `ctrlrange` | copied from generated joint `range` |
| `velocity_limit_sim` | report only | not yet an equivalent MuJoCo enforcement |

Important caveat: Isaac `ImplicitActuatorCfg` damping is an implicit drive damping term. Mapping it to MJCF joint `damping` makes the value explicit in MuJoCo, but it is not proof of exact PhysX/MuJoCo drive equivalence.

## Alignment Table

Command:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/export_g0_actuator_alignment_table.py
```

Output:

```text
logs/sim2sim/g0_actuator_alignment_table.md
```

Result:

```text
aligned rows: 22/22
```

Examples:

```text
l_hip_pitch_joint: standard, kp=4, forcerange=-0.5 0.5, damping=0.18, armature=0.0015
l_knee_pitch_joint: right_angle, kp=4, forcerange=-0.5833333333 0.5833333333, damping=0.26, armature=0.002041666667
r_elbow_pitch_joint: right_angle, kp=2, forcerange=-0.5833333333 0.5833333333, damping=0.08, armature=0.002041666667
```

The complete per-joint table is intentionally written under `logs/sim2sim/` because it is a generated validation artifact.

## Validation Results

Commands:

```bash
TERM=xterm conda run -n g0_isaaclab python -m pytest tests/sim2sim -q
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/generate_g0_mujoco_from_urdf.py
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/validate_sim2sim_setup.py --model mujoco/g0.xml
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/inspect_mujoco_collision.py --model mujoco/g0.xml --steps 1
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --steps 100 \
  --command 0.0 0.0 0.0 \
  --zero-action \
  --record-rollout logs/sim2sim/mujoco_zero_action_rollout.npz
```

Results:

```text
pytest: 11 passed
validate: Summary OK, all 22 joints found
actuator alignment: 22/22 rows aligned
foot_ground_contacts: 3
non_ground_self_contacts: 0
base_link_torso_link_contact: False
zero-action rollout: saved successfully
QACC warning: not observed in this 100-step zero-action run
```

Zero-action bridge:

```text
target_equals_default: True
max_target_default_abs_err: 0.0
max_abs_action: 0.0
joint_pos finite: True
```

The latest Isaac/MuJoCo zero-action compare still shows exact interface alignment:

```text
action mean/max abs error: 0 / 0
target_joint_pos mean/max abs error: 0 / 0
```

## Collision Geometry

Formal foot collision geometry was not changed.

Current `mujoco/g0.xml` still contains:

```text
l_foot_link: mesh geom using l_foot_link.STL
r_foot_link: mesh geom using r_foot_link.STL
```

No foot box, capsule, or sphere was added to the formal model.

Self-collision filtering remains active:

```text
ground: contype=1, conaffinity=2
robot:  contype=2, conaffinity=1
```

This preserves robot-ground contact and disables robot-robot internal contact, matching Isaac Lab `enabled_self_collisions=False`.

## Remaining Non-Equivalent Items

- `velocity_limit_sim` is recorded but not enforced with a verified equivalent MuJoCo actuator mechanism.
- Isaac implicit actuator damping and MuJoCo joint damping are only a first-pass approximation.
- MuJoCo position actuator dynamics are not proven equivalent to PhysX implicit actuators.
- Contact solver parameters remain MuJoCo defaults.
- Foot contact still uses URDF-derived mesh collision; fidelity to PhysX cooked convex hull behavior needs more inspection.
- Friction, mass/inertia sanity, and root initialization still need dedicated validation.
- `projected_gravity` and `base_ang_vel` frame equivalence still need controlled diagnostics.

## Next Step

Keep the actuator table as a regression artifact, then isolate the next source of mismatch with one change at a time:

1. Add diagnostics for velocity-limit behavior rather than assuming `velocity_limit_sim` is equivalent.
2. Compare MuJoCo and Isaac first-frame joint accelerations/contact forces under zero action.
3. Tune only solver/contact/friction parameters that can be traced to Isaac/PhysX semantics or measured hardware assumptions.
