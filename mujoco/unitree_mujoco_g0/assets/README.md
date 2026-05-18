# G0 Unitree-Style Assets

The generated `../g0.xml` intentionally references the canonical G0 mesh files
under:

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/meshes/
```

Meshes are not duplicated here so that Isaac Lab and MuJoCo diagnostics keep a
single source of truth for visual/collision geometry while the Unitree-style
MJCF control and sensor layout stays isolated in `mujoco/unitree_mujoco_g0/`.
