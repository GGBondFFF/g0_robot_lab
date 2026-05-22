// G0 motor_id <-> joint_name mapping for the virtual single-joint DDS test.
//
// This header is for the Isaac Lab virtual DDS test under
// test/single_joint_test only. It MUST NOT be used to drive real hardware.
//
// Must stay byte-for-byte equivalent to python/joint_mapping.py. The pytest
// suite cross-checks both tables.
#ifndef G0_SINGLE_JOINT_TEST_JOINT_MAPPING_HPP
#define G0_SINGLE_JOINT_TEST_JOINT_MAPPING_HPP

#include <array>
#include <stdexcept>
#include <string>

namespace g0_sjt {

struct JointEntry {
    int motor_id;
    const char* joint_name;
    int sim_sign_observed;  // +1 or -1
};

static const int kNumMotors = 22;

static const std::array<JointEntry, 22> kJointTable = {{
    { 1, "waist_yaw_joint",        -1},
    { 2, "waist_roll_joint",       -1},
    { 3, "l_shoulder_pitch_joint", -1},
    { 4, "l_shoulder_roll_joint",  -1},
    { 5, "l_shoulder_yaw_joint",   -1},
    { 6, "l_elbow_pitch_joint",    -1},
    { 7, "r_shoulder_pitch_joint", +1},
    { 8, "r_shoulder_roll_joint",  -1},
    { 9, "r_shoulder_yaw_joint",   -1},
    {10, "r_elbow_pitch_joint",    +1},
    {11, "l_hip_pitch_joint",      +1},
    {12, "l_hip_roll_joint",       -1},
    {13, "l_hip_yaw_joint",        -1},
    {14, "l_knee_pitch_joint",     -1},
    {15, "l_ankle_pitch_joint",    -1},
    {16, "l_ankle_roll_joint",     -1},
    {17, "r_hip_pitch_joint",      -1},
    {18, "r_hip_roll_joint",       -1},
    {19, "r_hip_yaw_joint",        -1},
    {20, "r_knee_pitch_joint",     +1},
    {21, "r_ankle_pitch_joint",    +1},
    {22, "r_ankle_roll_joint",     -1},
}};

inline const JointEntry& get_entry(int motor_id) {
    if (motor_id < 1 || motor_id > kNumMotors) {
        throw std::out_of_range("motor_id out of range (1..22)");
    }
    return kJointTable[static_cast<std::size_t>(motor_id - 1)];
}

inline std::string motor_id_to_joint_name(int motor_id) {
    return std::string(get_entry(motor_id).joint_name);
}

inline int sim_sign_for_motor(int motor_id) {
    return get_entry(motor_id).sim_sign_observed;
}

// Convert a right-hand-rule command into the sim's native-sign command.
inline double apply_sign(int motor_id, double desired_rhr_value) {
    return static_cast<double>(sim_sign_for_motor(motor_id)) * desired_rhr_value;
}

// Fill a 22-slot frame: only the target motor's slot gets the sign-converted
// command; all other slots are 0.0. Slot index = motor_id - 1.
inline std::array<double, 22> build_single_joint_frame(int motor_id,
                                                       double desired_rhr_value) {
    std::array<double, 22> frame;
    frame.fill(0.0);
    frame[static_cast<std::size_t>(motor_id - 1)] = apply_sign(motor_id, desired_rhr_value);
    return frame;
}

}  // namespace g0_sjt

#endif  // G0_SINGLE_JOINT_TEST_JOINT_MAPPING_HPP
