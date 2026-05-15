# MuJoCo G0 Assets

This directory is the MuJoCo side of the Isaac Lab to MuJoCo sim2sim workflow.

## Current `g0.xml` Status

`g0.xml` is a first editable scaffold. It is designed to validate software structure and interface alignment:

- all 22 `G0_JOINT_SDK_NAMES` joints exist
- all 22 position actuators exist
- the MuJoCo loader can build qpos/qvel/actuator indices
- zero-action and action bridge tests can run

It is not a final dynamics model. Do not treat its rollout as hardware evidence.

As of the first sim2sim validation pass, `g0.xml` loads successfully with the DeepMind MuJoCo Python API (`mujoco==3.8.1`) and exposes all 22 expected G0 SDK joints. The placeholder body layout was changed to a simple nested chain because MuJoCo does not allow a freejoint plus all hinge joints on the same body.

That change is only for interface validation. The nested chain is not the real G0 kinematic tree.

## What Still Needs To Be Converted Or Calibrated

The following must be aligned with Isaac Lab before policy behavior is meaningful:

- URDF/USD geometry
- body hierarchy
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

Current zero-action rollout validates the action bridge and file pipeline, but joint/root motion differences against Isaac Lab are expected until these physical parameters are replaced.

## Suggested Future Workflow

1. Convert or manually clean up `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/urdf/g0.urdf` into a full MJCF model.
2. Keep joint names exactly aligned with `G0_JOINT_SDK_NAMES`.
3. Keep actuator names equal to joint names or `<joint>_actuator`.
4. Re-run `python scripts/sim2sim/validate_sim2sim_setup.py`.
5. Run zero-action MuJoCo rollout.
6. Compare against Isaac golden I/O before trusting policy rollout behavior.

Generated intermediate files can go under `mujoco/generated/`. External meshes or copied assets can go under `mujoco/assets/`.
