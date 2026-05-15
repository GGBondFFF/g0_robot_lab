# G0 Contact Collision Fidelity Report

## Context

- Branch: `structure/mujoco-sim2sim-layout`
- Test time: `2026-05-15 15:02:01 CST`
- Goal: inspect contact/collision fidelity across URDF, Isaac USD/PhysX, and MuJoCo without changing the formal `mujoco/g0.xml` foot contact geometry.

Formal sim2sim should reproduce the collision semantics used by Isaac Lab through `G0_CFG` and the converted `g0.usd`. Do not add arbitrary foot box contacts to the formal model just to stabilize MuJoCo.

## URDF Foot Collision Summary

Command:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/inspect_g0_urdf_for_mujoco.py
```

General URDF result:

```text
link_count: 23
joint_count: 22
movable_joint_count: 22
mesh_count: 46
root_links: ['base_link']
foot_links: ['l_foot_link', 'r_foot_link']
missing_policy_joints: []
extra_movable_joints: []
missing_meshes: []
links_missing_inertial: []
movable_without_limit: []
```

`l_foot_link`:

```text
collision_count: 1
collision type: mesh
origin xyz/rpy: 0 0 0 / 0 0 0
collision mesh: ../meshes/l_foot_link.STL
mesh exists: True
visual mesh same as collision mesh: True
bbox min: (-0.0396614, -0.0160466, -0.0172086)
bbox max: (0.0354315, 0.0163397, 0.0120777)
vertex_count: 34782
face_count: 11594
```

`r_foot_link`:

```text
collision_count: 1
collision type: mesh
origin xyz/rpy: 0 0 0 / 0 0 0
collision mesh: ../meshes/r_foot_link.STL
mesh exists: True
visual mesh same as collision mesh: True
bbox min: (-0.0396614, -0.0163397, -0.0172086)
bbox max: (0.0354315, 0.0160466, 0.0120777)
vertex_count: 35064
face_count: 11688
```

Conclusion: the URDF formal foot collision is the STL mesh itself, not a hand-authored flat box sole.

## USD / PhysX Foot Collision Summary

Command:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/inspect_g0_usd_collision.py --headless
```

Converter config:

```text
collider_type: convex_hull
collision_from_visuals: false
self_collision: false
```

Visible foot prims in `g0.usd`:

```text
/g0/l_foot_link
/g0/l_foot_link/visuals
/g0/l_foot_link/collisions
/g0/r_foot_link
/g0/r_foot_link/visuals
/g0/r_foot_link/collisions
```

The visible foot link prims are Xforms. The link prims have `PhysicsRigidBodyAPI` and `PhysicsMassAPI`. The visible `visuals` and `collisions` Xforms do not show an extra foot box or extra authored collision mesh child in the USD traversal.

No additional simplified foot collision prim was found by the USD inspection script. Based on the converter config, the intended Isaac/PhysX collision is the URDF collision mesh converted with convex hull approximation, with self-collision disabled.

Some low-level PhysX collision details such as runtime cooked convex hull data, contact offset, and rest offset were not authored on the visible foot Xforms and may be applied by PhysX defaults/runtime cooking rather than explicit USD attributes.

## MuJoCo Foot Collision Summary

Command:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/inspect_mujoco_collision.py --model mujoco/g0.xml --steps 1
```

XML foot geoms:

```text
l_foot_link: one geom, type=mesh, mesh=l_foot_link
r_foot_link: one geom, type=mesh, mesh=r_foot_link
```

Compiled MuJoCo foot geoms:

```text
l_foot_link geom:
  type_id: 7
  mesh: l_foot_link
  friction: [1.0, 0.005, 0.0001]
  contype: 1
  conaffinity: 1
  solref: [0.02, 1.0]
  solimp: [0.9, 0.95, 0.001, 0.5, 2.0]

r_foot_link geom:
  type_id: 7
  mesh: r_foot_link
  friction: [1.0, 0.005, 0.0001]
  contype: 1
  conaffinity: 1
  solref: [0.02, 1.0]
  solimp: [0.9, 0.95, 0.001, 0.5, 2.0]
```

Initial/one-step contacts:

```text
ncon: 4
foot_ground_contacts: 3
non_ground_self_contacts: 1
negative_distance_contacts: 4
max_contact_force_norm: 44.5347
```

Contacts observed:

```text
ground <-> l_foot_link, dist=-0.00249075
ground <-> l_foot_link, dist=-0.00249075
ground <-> r_foot_link, dist=-0.00249075
base_link <-> torso_link, dist=-0.00158629, force_norm=44.5347
```

Conclusion: MuJoCo currently uses mesh foot geoms, which is closer to the URDF source than a hand-authored foot box. However, MuJoCo also currently allows internal self-collision by default.

## Three-Way Fidelity Assessment

URDF:

- foot collision is STL mesh
- visual mesh and collision mesh are the same foot STL

USD/PhysX:

- converter config uses URDF collision mesh, `collision_from_visuals: false`
- converter config uses `collider_type: convex_hull`
- converter config disables self-collision
- no extra simplified foot box was found

MuJoCo:

- foot collision is mesh geom using the foot mesh names
- friction/solver/contact parameters are MuJoCo defaults unless explicitly authored
- self-collision is effectively enabled through default `contype=1`, `conaffinity=1`

Main mismatch found in this pass:

```text
Isaac self-collision disabled
MuJoCo internal self-collision currently possible
```

This mismatch is directly observed through the initial `base_link <-> torso_link` contact.

## Self-Collision Semantics

Isaac Lab `G0_CFG` sets:

```text
enabled_self_collisions = False
```

MuJoCo has no single equivalent flag in the current `g0.xml`. With default `contype=1` and `conaffinity=1`, internal robot geoms can collide unless excluded or filtered.

Observed self-contact:

```text
base_link <-> torso_link
dist=-0.00158629
force_norm=44.5347
```

This is a strong candidate contributor to the current MuJoCo QACC instability because it creates a non-foot internal penetration/contact that Isaac Lab intentionally disables.

Potential MuJoCo equivalents for Isaac self-collision disabled:

- use collision filtering so robot geoms do not collide with other robot geoms
- add explicit `<exclude>` pairs for robot body pairs
- set robot geom contype/conaffinity to avoid robot-robot contacts while preserving robot-ground contacts

This should be done carefully in a dedicated model update. It should not be mixed with arbitrary foot sole box changes.

## QACC Instability: Likely Source Ranking

Most likely:

1. Internal self-collision mismatch: MuJoCo shows `base_link <-> torso_link` contact while Isaac disables self-collisions.
2. Actuator/solver mismatch: current MuJoCo position actuators use placeholder gains and limits, not Isaac `G0_CFG` actuator parameters.
3. Initial contact/penetration: foot-ground contacts start with negative distance around `-0.00249 m`.
4. Contact solver differences: Isaac uses PhysX convex hull collision; MuJoCo mesh collision/convex handling and solver defaults are different.
5. Foot collision geometry fidelity: both use foot mesh sources, but Isaac converter uses convex hull; MuJoCo compiled collision behavior must be confirmed.
6. Mass/inertia differences: MuJoCo is URDF-derived, but Isaac uses converted USD/PhysX cooked properties and should be checked numerically.

## Should Formal `mujoco/g0.xml` Be Modified?

Do not add arbitrary simplified foot boxes to the formal `mujoco/g0.xml` at this point.

Recommended formal-model changes to consider first:

- reproduce Isaac's self-collision disabled semantics in MuJoCo
- align actuator PD/force/velocity parameters
- inspect initial root height and penetration
- compare convex hull behavior between Isaac/PhysX and MuJoCo

## Is A Debug-Only Contact Model Needed?

Not yet. The current evidence points first to self-collision and actuator/contact parameter mismatch. If a simplified contact experiment becomes useful, create a separate file:

```text
mujoco/g0_contact_debug.xml
```

and document it as debug-only. Do not use that file as the formal sim2sim model.

## Why Not Add A Foot Box To Formal `g0.xml`?

Because Isaac Lab currently uses the URDF-derived USD asset from `G0_CFG`, and the URDF foot collision is the foot STL mesh. A hand-authored MuJoCo foot box would change the contact patch and could make MuJoCo behavior look better while moving away from the actual Isaac baseline.

The formal sim2sim path should first reproduce the Isaac collision semantics. Debug simplifications are allowed only in separate files with clear labels.

## Next Steps

1. Add a MuJoCo self-collision filtering experiment that preserves robot-ground contacts.
2. Keep the experiment separate or clearly document the formal semantics if applied to `g0.xml`.
3. Align position actuator gains, torque limits, and velocity limits with `G0_CFG`.
4. Re-run zero-action rollout and contact inspection after self-collision filtering.
5. Compare initial root height and foot penetration against Isaac/PhysX.
6. Only then consider debug-only simplified foot contact in `mujoco/g0_contact_debug.xml`.

## Self-Collision Filtering Update

Update time:

```text
2026-05-15 15:33:23 CST
```

The formal `mujoco/g0.xml` was regenerated from URDF with Isaac-style robot self-collision filtering. Foot collision geometry was not changed:

```text
l_foot_link: type=mesh, mesh=l_foot_link
r_foot_link: type=mesh, mesh=r_foot_link
```

No foot box, capsule, or sphere was added.

Filtering rule implemented in `scripts/sim2sim/generate_g0_mujoco_from_urdf.py`:

```text
ground geoms: contype=1, conaffinity=2
robot geoms:  contype=2, conaffinity=1
```

MuJoCo contact filtering uses:

```text
(contype1 & conaffinity2) || (contype2 & conaffinity1)
```

Therefore:

```text
ground-robot: enabled
robot-robot: disabled
```

Contact inspection after filtering:

```text
foot_ground_contacts: 3
non_ground_self_contacts: 0
negative_distance_contacts: 3
max_contact_force_norm: 2.27171
base_link_torso_link_contact: False
isaac_style_self_collision_disabled: True
```

Before filtering, the same inspection showed:

```text
foot_ground_contacts: 3
non_ground_self_contacts: 1
base_link_torso_link_contact: True
max_contact_force_norm: 44.5347
```

So the self-collision mismatch with Isaac Lab is resolved at the contact-filtering level.

Zero-action rollout after filtering:

```text
Saved MuJoCo rollout: logs/sim2sim/mujoco_zero_action_rollout.npz
Finished 100 MuJoCo control steps.
WARNING: Nan, Inf or huge value in QACC at DOF 3. The simulation is unstable. Time = 0.0450.
```

The QACC warning did not disappear. It changed from the previous internal-contact case, but remains present. With robot self-collision removed, the next likely causes are:

1. placeholder actuator PD/force/velocity settings
2. foot-ground initial penetration around `-0.00249 m`
3. MuJoCo solver/contact parameters versus PhysX convex hull behavior
4. mass/inertia and root initialization differences

The action bridge remains exact:

```text
target_joint_pos == default_joint_pos
max_target_default_abs_err: 0.0
action compare error: 0 / 0
target_joint_pos compare error: 0 / 0
```

Recommendation: keep the formal foot mesh collision unchanged, and next align actuator parameters and initial contact/root height before considering any debug-only contact model.
