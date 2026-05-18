# Controlled Root Frame Convention Diagnostics

- isaac: `logs/sim2sim/root_frame/isaac_controlled_root_state.npz`
- mujoco: `logs/sim2sim/root_frame/mujoco_controlled_root_state.npz`
- samples: `20`

## Summary

- quaternion order assessment: `wxyz`
- quaternion max abs error allowing sign flip: `3.74281e-08`
- xyzw reorder candidate max abs error: `1`
- projected_gravity max abs error: `2.32911e-08`
- projected_gravity aligned: `True`
- base_ang_vel max abs error: `5.5907e-08`
- base_ang_vel aligned: `True`
- root_height max abs error: `4.17233e-09`
- root_height note: root height matches controlled value closely

## Interpretation

- Isaac and MuJoCo root quaternions use the same `w, x, y, z` order for these controlled samples.
- `G0MuJoCoInterface.get_projected_gravity()` matches Isaac `projected_gravity_b` for controlled orientations.
- `G0MuJoCoInterface.get_base_ang_vel()` matches Isaac `root_ang_vel_b` for the controlled angular velocity samples.
- No actuator, collision, root-height, friction, or solver parameters were changed by this diagnostic.

## Per-Sample Errors

| sample | quat err | projected gravity err | base ang vel err | root height err |
| --- | ---: | ---: | ---: | ---: |
| `upright_w000` | 8.35466e-12 | 1.67093e-11 | 0 | 4.17233e-09 |
| `upright_w100` | 8.35466e-12 | 1.67093e-11 | 1.67093e-11 | 4.17233e-09 |
| `upright_w010` | 8.35466e-12 | 1.67093e-11 | 1.87022e-12 | 4.17233e-09 |
| `upright_w001` | 8.35466e-12 | 1.67093e-11 | 1.67093e-11 | 4.17233e-09 |
| `roll10_w000` | 3.74281e-08 | 2.32911e-08 | 0 | 4.17233e-09 |
| `roll10_w100` | 3.74281e-08 | 2.32911e-08 | 4.43346e-10 | 4.17233e-09 |
| `roll10_w010` | 3.74281e-08 | 2.32911e-08 | 2.32911e-08 | 4.17233e-09 |
| `roll10_w001` | 3.74281e-08 | 2.32911e-08 | 2.32911e-08 | 4.17233e-09 |
| `pitch10_w000` | 2.21765e-08 | 2.32911e-08 | 0 | 4.17233e-09 |
| `pitch10_w100` | 2.21765e-08 | 2.32911e-08 | 2.32911e-08 | 4.17233e-09 |
| `pitch10_w010` | 2.21765e-08 | 2.32911e-08 | 1.1521e-12 | 4.17233e-09 |
| `pitch10_w001` | 2.21765e-08 | 2.32911e-08 | 2.32911e-08 | 4.17233e-09 |
| `yaw30_w000` | 1.35678e-08 | 1.26333e-10 | 0 | 4.17233e-09 |
| `yaw30_w100` | 1.35678e-08 | 1.26333e-10 | 1.55436e-08 | 4.17233e-09 |
| `yaw30_w010` | 1.35678e-08 | 1.26333e-10 | 1.55436e-08 | 4.17233e-09 |
| `yaw30_w001` | 1.35678e-08 | 1.26333e-10 | 1.26333e-10 | 4.17233e-09 |
| `roll10_pitch10_yaw30_w000` | 1.95497e-08 | 9.56433e-09 | 0 | 4.17233e-09 |
| `roll10_pitch10_yaw30_w100` | 1.95497e-08 | 9.56433e-09 | 4.79901e-08 | 4.17233e-09 |
| `roll10_pitch10_yaw30_w010` | 1.95497e-08 | 9.56433e-09 | 5.5907e-08 | 4.17233e-09 |
| `roll10_pitch10_yaw30_w001` | 1.95497e-08 | 9.56433e-09 | 9.56433e-09 | 4.17233e-09 |
