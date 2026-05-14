# G0 Real Deployment Fall Handling Plan

## Pre-Deployment Checks

Before running any learned policy on the real G0:

- Check motor zero points.
- Check joint direction signs.
- Check action order against SDK order.
- Check left/right leg mirror relationships.
- Check default standing pose.
- Check mechanical joint limits.
- Check low-voltage protection.
- Check over-current protection.
- Check over-temperature protection.
- Check communication timeout behavior.

Do not change real motor torque limits to compensate for a weak policy.

## First Real-Robot Policy Bring-Up

Do not run the full policy immediately.

Use a staged action multiplier:

```text
0.0 -> 0.2 -> 0.4 -> 0.6 -> 0.8 -> 1.0
```

At each stage, observe:

- foot slip
- knee collapse
- hip shaking
- forward body pitch
- lateral body roll
- ankle chatter
- unexpected arm/waist motion
- motor current and temperature

At action multiplier `0.0`, the policy is loaded but contributes no motion. This checks software wiring without commanding RL motion.

## If G0 Falls Forward

Check:

- whether default hip/knee/ankle pose puts the center of mass too far forward;
- whether the base height target is too high or too low;
- whether foot collision geometry contacts the ground correctly;
- whether foot friction is too low;
- whether `hip_pitch`, `knee_pitch`, and `ankle_pitch` action signs match deployment;
- whether commanded forward velocity is too large;
- whether the policy is producing excessive forward swing-leg targets.

Do not solve this by raising actuator torque limits.

## If G0 Wobbles Laterally Or Side-Falls

Check:

- `hip_roll` sign;
- `ankle_roll` sign;
- left/right mirror mapping;
- whether lateral velocity command curriculum is too aggressive;
- whether feet-slide penalty is too weak;
- whether the stance width in the default pose is too narrow;
- whether lateral push curriculum was introduced gradually enough.

Reduce lateral velocity curriculum before changing hardware constants.

## If Knees Collapse

Do not directly increase torque limits.

Check:

- whether stiffness/damping matches the real servo low-level position loop;
- whether action scale is too large;
- whether action clipping is active in deployment;
- whether default knee angle puts the leg in a weak support posture;
- whether base height target asks the robot to stand too tall;
- whether the policy is saturating knee torque continuously.

For G0, continuous torque margin is small. A policy that depends on torque saturation is not deployable.

## If Feet Slip

Check:

- foot collision mesh;
- foot sole material;
- static and dynamic friction;
- whether gait reward induces too-fast foot exchange;
- whether `feet_slide` penalty is too weak;
- whether commanded yaw/lateral velocity is too high;
- whether contact state in simulation matches real foot contact.

Do not mask slipping by widening termination thresholds.

## If Actions Shake

Check:

- `action_rate` logs;
- policy std;
- raw policy action magnitude;
- deployed action clipping;
- deployment control frequency;
- Isaac Lab decimation and real control loop period;
- whether the real servo position loop can track the policy targets;
- whether observation filtering/normalization matches training.

The policy should not require high-frequency target jumps to stay upright.

## Safety Stop Logic

The real deployment controller should stop RL output immediately when:

- base roll exceeds a configured safe threshold;
- base pitch exceeds a configured safe threshold;
- any joint approaches mechanical limits;
- motor current exceeds safe limit;
- motor temperature exceeds safe limit;
- battery voltage is too low;
- communication times out;
- state estimator data becomes stale or invalid;
- the robot is already falling.

Safe stop behavior should:

- stop sending RL actions;
- optionally hold or move toward a tested safe pose if still upright;
- avoid fighting the fall with high-gain learned actions;
- log the last observations, actions, joint states, torques, and safety trigger.

Never continue executing RL actions during a fall.

## Post-Fall Debug Checklist

After any fall, record:

- command at fall time;
- action vector and target joint positions;
- raw joint positions and velocities;
- estimated base roll/pitch/yaw;
- motor currents and temperatures;
- foot contact state if available;
- whether safety stop triggered;
- whether the fall was forward, backward, lateral, or knee-collapse.

Then reproduce the same command in simulation with:

- the same action multiplier;
- the same command range;
- the same default pose;
- the same action order;
- the same control frequency.
