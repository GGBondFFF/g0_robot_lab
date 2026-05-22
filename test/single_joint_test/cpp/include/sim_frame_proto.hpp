// Binary wire protocol on the UDS channel between dds_to_uds_bridge (C++)
// and isaac_gui_receiver.py (Python).
//
// One fixed-size frame per message: 24-byte header + 22 * 20-byte motor slots
// = 464 bytes. Fields stay in DDS-wire units (degrees for pos / dq, N*m for
// tau) — the Python side converts deg -> rad at the Isaac Lab boundary.
#ifndef G0_SINGLE_JOINT_TEST_SIM_FRAME_PROTO_HPP
#define G0_SINGLE_JOINT_TEST_SIM_FRAME_PROTO_HPP

#include <cstdint>

namespace g0_sjt {

#pragma pack(push, 1)

struct SimFrameHeader {
    uint32_t magic;        // 0x47305349 = 'G' '0' 'S' 'I' (little-endian)
    uint16_t version;      // protocol version, currently 1
    uint16_t motor_count;  // always 22 for G0
    uint64_t timestamp_ns; // bridge-side ingest time, ns since steady_clock epoch
    uint64_t sequence_id;  // monotonic, extended from DDS uint16 by bridge
};
static_assert(sizeof(SimFrameHeader) == 24, "SimFrameHeader must be 24 bytes");

struct SimMotorWire {
    float pos;  // degrees    (DDS wire unit, matches motion-planner)
    float dq;   // deg/s
    float kp;
    float kd;
    float tau;  // N*m
};
static_assert(sizeof(SimMotorWire) == 20, "SimMotorWire must be 20 bytes");

struct SimFrame {
    SimFrameHeader header;
    SimMotorWire motors[22];
};
static_assert(sizeof(SimFrame) == 24 + 22 * 20, "SimFrame must be 464 bytes");

#pragma pack(pop)

constexpr uint32_t kSimFrameMagic = 0x47305349u;  // 'G0SI'
constexpr uint16_t kSimFrameVersion = 1;
constexpr int kSimFrameBytes = static_cast<int>(sizeof(SimFrame));
constexpr const char* kDefaultUdsPath = "/tmp/g0_sim/motor_control_virtual.sock";

}  // namespace g0_sjt

#endif  // G0_SINGLE_JOINT_TEST_SIM_FRAME_PROTO_HPP
