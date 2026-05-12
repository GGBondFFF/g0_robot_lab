# G0 Initial Geometry Debug

## Purpose

`effort_scale=2.0` reduced the ankle torque ratio below long-term saturation, but G0 still fell in almost the same way. That means the next debug target is not PPO or reward tuning. The useful questions are now:

- Is the COM projection sensible relative to the feet?
- Are the feet actually in contact at reset?
- Is the foot collision patch flat and large enough?
- Is root height creating initial suspension or penetration?
- Are mass and inertia close to the expected robot?

## Command

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/inspect_g0_initial_geometry.py \
  --task G0-Velocity-v0 \
  --headless \
  --root_z 0.23 \
  --hip 0.20 \
  --knee 0.34 \
  --ankle 0.14
```

The script is debug-only. It disables reset randomization, velocity command, and base/bad-orientation termination, then prints initial geometry and mass/contact properties.

## Initial Geometry Output

Candidate:

```text
root_z=0.23
hip=0.20
knee=0.34
ankle=0.14
settle_steps=1
```

Root:

```text
root position=(-0.00172, 0.00010, 0.23578)
root orientation wxyz=(0.99996, 0.00261, 0.00647, -0.00492)
root_roll_deg=0.29589
root_pitch_deg=0.74235
root_yaw_deg=-0.56235
```

Mass and COM:

```text
total robot mass=1.352610 kg
whole-body COM=(-0.00202, 0.00005, 0.26771)
support center=(0.00792, 0.00158)
com_dx=-0.00994
com_dy=-0.00153
```

The COM projection is near the foot center at reset, so the first-frame COM projection is not obviously outside the support center. However, the runtime total mass is far below the expected about `1.9 kg`.

Feet:

```text
l_foot_link world position=(0.00758, 0.03941, 0.01897)
r_foot_link world position=(0.00825, -0.03625, 0.01885)
l_foot_link lowest collision/mesh point z=0.00081
r_foot_link lowest collision/mesh point z=0.00133
left/right foot contact force z=0.00000, 0.00000
left/right contact point count=0, 0
```

Both feet are slightly above the ground by about `0.8-1.3 mm` at this inspected state, and the contact sensor reports no foot contact after one small refresh step. This points to an initial contact/height issue: the robot is close to the ground but not actually carrying weight at reset.

## Link Mass/Inertia Notes

Runtime link masses were available from PhysX. The largest body is:

```text
torso_link mass=0.506639 kg
```

Each foot is only about:

```text
l_foot_link mass=0.026917 kg
r_foot_link mass=0.026918 kg
```

The script warning was:

```text
- total_mass 1.3526 kg is not close to expected 1.9 kg
- both foot contact forces are near zero
```

No zero link mass was observed. Several very small links have very small inertia diagonals, which may be acceptable for tiny links, but should be compared against the original CAD/URDF export because the total mass mismatch is large enough to matter.

## Foot Collision/Contact Patch

URDF foot collision:

```xml
<link name="l_foot_link">
  ...
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <mesh filename="../meshes/l_foot_link.STL" />
    </geometry>
  </collision>
</link>

<link name="r_foot_link">
  ...
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <mesh filename="../meshes/r_foot_link.STL" />
    </geometry>
  </collision>
</link>
```

USD conversion config:

```yaml
collider_type: convex_hull
collision_from_visuals: false
self_collision: false
```

Foot mesh local bboxes:

```text
l_foot_link min=(-0.03966, -0.01605, -0.01721) max=(0.03543, 0.01634, 0.01208)
r_foot_link min=(-0.03966, -0.01634, -0.01721) max=(0.03543, 0.01605, 0.01208)
```

This means each foot collision is a mesh converted to a convex hull. The nominal footprint is about `75 mm x 32 mm`, but the effective contact patch may be curved or narrow depending on the STL shape and convex hull. There is no explicit flat sole box collider in the URDF.

## Findings

Current most important findings:

- The runtime total mass is `1.3526 kg`, not close to the expected `1.9 kg`.
- The feet are nearly touching but initially report zero contact force.
- The foot collision is STL mesh/convex hull, not an explicit flat sole contact surface.
- The initial COM projection is not wildly outside the support center, but during the fall it quickly moves far forward.

## Recommended Next Fixes

1. Verify the original CAD/URDF mass export against the expected `1.9 kg`. If the real G0 is 1.9 kg, fix mass/inertia in the asset source rather than compensating with rewards.
2. Inspect the foot STL shape visually or with a mesh tool. Confirm whether the bottom is flat. If it is rounded or too small, add a minimal thin box collider under each foot as a realistic sole contact patch.
3. Re-test root height after contact patch inspection. The current foot lowest points are about `1 mm` above ground, so try a geometry-only check around `root_z=0.228-0.229` before changing rewards.
4. Keep `g0_actuators.py` hardware constants fixed. Use only `--effort-scale` for temporary torque diagnosis.
