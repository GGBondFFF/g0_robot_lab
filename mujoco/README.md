# MuJoCo G0 Assets

This directory is the MuJoCo side of the Isaac Lab to MuJoCo sim2sim workflow.

## Current Models

### `g0_interface_placeholder.xml`

`g0_interface_placeholder.xml` preserves the first interface scaffold. It is designed to validate software structure and interface alignment:

- all 22 `G0_JOINT_SDK_NAMES` joints exist
- all 22 position actuators exist
- the MuJoCo loader can build qpos/qvel/actuator indices
- zero-action and action bridge tests can run

It is not a dynamics model. Do not treat its rollout as hardware evidence.

Validate it with:

```bash
python scripts/sim2sim/validate_sim2sim_setup.py --model mujoco/g0_interface_placeholder.xml
```

### `g0.xml`

`g0.xml` is now a URDF-derived working model generated from:

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/urdf/g0.urdf
```

Generate or refresh it with:

```bash
python scripts/sim2sim/generate_g0_mujoco_from_urdf.py
```

It currently includes:

- URDF-derived body tree
- URDF-derived joint axes and limits
- URDF-derived inertial values
- URDF-derived mesh geoms
- added floating root
- added ground plane
- added 22 position actuators named exactly like `G0_JOINT_SDK_NAMES`

As of the first URDF migration pass, `g0.xml` loads successfully with the DeepMind MuJoCo Python API (`mujoco==3.8.1`) and exposes all 22 expected G0 SDK joints.

It is still not final dynamics. The latest zero-action rollout reports MuJoCo `QACC` instability, so current state/velocity rollout errors are expected and must not be interpreted as policy failure.

## What Still Needs To Be Calibrated

The following must be aligned with Isaac Lab before policy behavior is meaningful:

- body mass and inertia
- default pose
- actuator PD gains
- torque limits
- velocity limits
- damping and armature
- foot collision geometry
- contact patch
- friction
- root frame convention

Current zero-action rollout validates the action bridge and file pipeline, but joint/root motion differences against Isaac Lab are expected until these physical parameters are aligned.

## Suggested Future Workflow

1. Keep joint names exactly aligned with `G0_JOINT_SDK_NAMES`.
2. Keep actuator names equal to joint names or `<joint>_actuator`.
3. Replace raw mesh contact with validated foot contact geometry.
4. Align actuator PD, torque limits, velocity limits, damping, and armature.
5. Re-run `python scripts/sim2sim/validate_sim2sim_setup.py --model mujoco/g0.xml`.
6. Run zero-action MuJoCo rollout.
7. Compare against Isaac golden I/O before trusting policy rollout behavior.

Generated intermediate files can go under `mujoco/generated/`. External meshes or copied assets can go under `mujoco/assets/`.
