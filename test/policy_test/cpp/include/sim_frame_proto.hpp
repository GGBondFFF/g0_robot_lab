// Binary wire protocol for the policy_test UDS bridges.
//
// Three independent channels, each carrying one fixed-size frame type. All
// fields stay in DDS wire units (degrees for motor pos/dq, N*m for tau, raw
// IMU units, m/s for body linear velocity). Conversion to Isaac Lab radians
// happens on the Python side at the Isaac boundary, never on the wire.
//
// Frames are #pragma pack(1) and use distinct magic numbers so a mis-wired
// socket fails loudly instead of silently producing garbage.
//
// Channel summary (server = bridge listens, client = Python connects):
//
//   policy_control_bridge (sim2real "robot brain" side):
//     OUT robot_control_uds          -> RobotControlFrame  ('G0RC')
//     OUT motor_state_uds            -> MotorStateFrame    ('G0SS')
//     IN  motor_control_uds          <- MotorControlFrame  ('G0SC')
//
//   plant_state_bridge (Isaac Lab plant side):
//     OUT motor_control_uds          -> MotorControlFrame  ('G0SC')
//     IN  motor_state_uds            <- MotorStateFrame    ('G0SS')
//
// The MotorControlFrame layout is deliberately identical to
// test/single_joint_test/cpp/include/sim_frame_proto.hpp::SimFrame so the
// same Python decoder can handle both. Magic is 'G0SC' here (vs 'G0SI' there)
// to keep cross-test traffic obvious in logs.
#ifndef G0_POLICY_TEST_SIM_FRAME_PROTO_HPP
#define G0_POLICY_TEST_SIM_FRAME_PROTO_HPP

#include <cstdint>

namespace g0_pt {

constexpr int kNumMotors = 22;

#pragma pack(push, 1)

// ─── shared header (all frames) ────────────────────────────────────────────
struct FrameHeader {
    uint32_t magic;         // channel discriminator (see kFrameMagic*)
    uint16_t version;       // currently 1
    uint16_t payload_count; // 22 for motor frames, 1 for robot_control
    uint64_t timestamp_ns;  // bridge ingest time (system_clock when DDS sub),
                            //   or producer-supplied time when Python writes.
    uint64_t sequence_id;   // monotonic, extended from DDS uint16 by bridge.
};
static_assert(sizeof(FrameHeader) == 24, "FrameHeader must be 24 bytes");

// ─── MotorControl::Control frame (Python policy -> DDS sim) ────────────────
struct MotorCmdWire {
    float pos;  // degrees
    float dq;   // deg/s
    float kp;   // N*m / deg
    float kd;   // N*m / (deg/s)
    float tau;  // N*m
};
static_assert(sizeof(MotorCmdWire) == 20, "MotorCmdWire must be 20 bytes");

struct MotorControlFrame {
    FrameHeader header;
    MotorCmdWire motors[22];
};
static_assert(sizeof(MotorControlFrame) == 24 + 22 * 20,
              "MotorControlFrame must be 464 bytes");

// ─── MotorControl::State frame (Isaac plant -> Python policy) ──────────────
struct MotorStateWire {
    uint8_t isvalid;
    float pos;          // degrees
    float dq;           // deg/s
    float tau;          // N*m
    uint8_t status;
    float fpc_temper;   // degC
    float pcb_temper;   // degC
};
static_assert(sizeof(MotorStateWire) == 1 + 4 + 4 + 4 + 1 + 4 + 4,
              "MotorStateWire must be 22 bytes");

struct ImuDataWire {
    uint8_t imu_data_valid;
    uint8_t mag_data_valid;
    uint32_t imu_timestamp;
    uint32_t mag_timestamp;
    float acc_x,  acc_y,  acc_z;   // m/s^2
    float gyro_x, gyro_y, gyro_z;  // rad/s (matches Common::Velocity3D in IMU)
    float mag_x,  mag_y,  mag_z;
};
static_assert(sizeof(ImuDataWire) == 1 + 1 + 4 + 4 + 12 + 12 + 12,
              "ImuDataWire must be 46 bytes");

struct MotorStateFrame {
    FrameHeader header;
    MotorStateWire motors[22];
    ImuDataWire imu;
};
static_assert(sizeof(MotorStateFrame) == 24 + 22 * 22 + 46,
              "MotorStateFrame must be 554 bytes");

// ─── RobotControl::Control frame (operator -> Python policy) ───────────────
// Mirrors fake_control_agent.toml semantics. ArmState is intentionally
// dropped — the velocity policy doesn't read it.
struct GripperWire {
    uint8_t is_valid;
    float roll, pitch, yaw;            // attitude (deg as on the wire)
    float lin_x, lin_y, lin_z;         // linear_velocity (m/s)
    float opening;                     // 0..1
};
static_assert(sizeof(GripperWire) == 1 + 12 + 12 + 4,
              "GripperWire must be 29 bytes");

struct BodyStateWire {
    float roll, pitch, yaw;            // attitude (deg)
    float lin_x, lin_y, lin_z;         // m/s, normalized in fake agent to [-1,+1]
    float ang_x, ang_y, ang_z;         // rad/s (yaw_rate sits in ang_z)
    float height;
};
static_assert(sizeof(BodyStateWire) == 40, "BodyStateWire must be 40 bytes");

struct RobotControlFrame {
    FrameHeader header;
    uint8_t mode;            // ::Common::RobotMode enum value (0=HEXAPOD,1=QUADRUPED,2=DANCE)
    uint8_t has_left_gripper;
    uint8_t has_right_gripper;
    uint8_t reserved;
    BodyStateWire body;
    GripperWire left_gripper;   // ignored if has_left_gripper == 0
    GripperWire right_gripper;  // ignored if has_right_gripper == 0
};
static_assert(sizeof(RobotControlFrame) == 24 + 4 + 40 + 29 + 29,
              "RobotControlFrame must be 126 bytes");

#pragma pack(pop)

// ─── magic constants ──────────────────────────────────────────────────────
constexpr uint32_t kFrameMagicMotorControl = 0x47305343u; // 'G' '0' 'S' 'C'
constexpr uint32_t kFrameMagicMotorState   = 0x47305353u; // 'G' '0' 'S' 'S'
constexpr uint32_t kFrameMagicRobotControl = 0x47305243u; // 'G' '0' 'R' 'C'
constexpr uint16_t kFrameVersion = 1;

// ─── default UDS paths (all under /tmp/g0_sim/, mirrors single_joint_test) ─
constexpr const char* kUdsRobotControl = "/tmp/g0_sim/robot_control.sock";
constexpr const char* kUdsMotorStateToPolicy = "/tmp/g0_sim/motor_state_to_policy.sock";
constexpr const char* kUdsMotorControlFromPolicy = "/tmp/g0_sim/motor_control_from_policy.sock";
constexpr const char* kUdsMotorControlToPlant = "/tmp/g0_sim/motor_control_to_plant.sock";
constexpr const char* kUdsMotorStateFromPlant = "/tmp/g0_sim/motor_state_from_plant.sock";

// Sim-only DDS topic names (mirror motion-planner's mc/* and robot_control,
// prefixed with g0_sim/ so the bridges refuse anything starting with mc/).
constexpr const char* kDdsTopicMotorControl = "g0_sim/motor_control";
constexpr const char* kDdsTopicMotorState   = "g0_sim/motor_state";
constexpr const char* kDdsTopicRobotControl = "g0_sim/robot_control";

}  // namespace g0_pt

#endif  // G0_POLICY_TEST_SIM_FRAME_PROTO_HPP
