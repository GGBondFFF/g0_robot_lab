# G0 Sim2sim Checklist

- [ ] `main` baseline train/play is runnable
- [ ] work is on `structure/mujoco-sim2sim-layout`
- [ ] `policy.pt` exported from Isaac Lab `play.py`
- [ ] Isaac golden I/O dumped with `dump_isaac_golden_io.py`
- [ ] MuJoCo model loads
- [ ] 22 joints found in MuJoCo
- [ ] joint order matches `G0_JOINT_SDK_NAMES`
- [ ] default pose matches `G0_DEFAULT_JOINT_POS`
- [ ] zero action gives default pose
- [ ] all-one action gives `default + 0.12`
- [ ] all-negative-one action gives `default - 0.12`
- [ ] action clipping verified
- [ ] first-frame observation compared
- [ ] `projected_gravity` frame convention verified
- [ ] `base_ang_vel` frame convention verified
- [ ] control dt and decimation match Isaac Lab
- [ ] actuator PD gains aligned
- [ ] torque and velocity limits aligned
- [ ] foot contact geometry aligned
- [ ] friction aligned
- [ ] short rollout compared
- [ ] failure causes documented

