// policy_control_bridge — sim2real "robot brain" side bridge.
//
// Connects the Python policy process to DDS:
//
//   DDS  g0_sim/robot_control   --sub--> UDS broadcast  (RobotControlFrame)
//   DDS  g0_sim/motor_state     --sub--> UDS broadcast  (MotorStateFrame)
//   DDS  g0_sim/motor_control   <--pub-- UDS receive    (MotorControlFrame)
//
// All conversions are byte-faithful to the IDL — no unit changes, no sign
// flipping. The Python side is responsible for assembling observations and
// emitting motor_control frames at the policy's 50 Hz cadence.
//
// HARD-WIRED safety: refuses any DDS topic starting with "mc/" (real
// hardware) and any path outside /tmp/ if --uds-* is overridden.

#include "mbus_sim_io.hpp"
#include "sim_frame_proto.hpp"
#include "uds_socket.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

struct Args {
    std::string topic_robot_control = g0_pt::kDdsTopicRobotControl;
    std::string topic_motor_state   = g0_pt::kDdsTopicMotorState;
    std::string topic_motor_control = g0_pt::kDdsTopicMotorControl;
    std::string uds_robot_control     = g0_pt::kUdsRobotControl;
    std::string uds_motor_state_out   = g0_pt::kUdsMotorStateToPolicy;
    std::string uds_motor_control_in  = g0_pt::kUdsMotorControlFromPolicy;
    double poll_hz = 200.0;
    bool allow_real_hw = false;
};

void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "DDS topics (must start with g0_sim/):\n"
        "  --topic-rc <name>   default %s\n"
        "  --topic-ms <name>   default %s\n"
        "  --topic-mc <name>   default %s\n"
        "UDS paths:\n"
        "  --uds-rc       <path>   bridge -> python RobotControlFrame  (default %s)\n"
        "  --uds-ms-out   <path>   bridge -> python MotorStateFrame    (default %s)\n"
        "  --uds-mc-in    <path>   python -> bridge MotorControlFrame  (default %s)\n"
        "  --poll-hz      <float>  bridge inner loop rate              (default 200)\n",
        argv0,
        g0_pt::kDdsTopicRobotControl, g0_pt::kDdsTopicMotorState,
        g0_pt::kDdsTopicMotorControl,
        g0_pt::kUdsRobotControl, g0_pt::kUdsMotorStateToPolicy,
        g0_pt::kUdsMotorControlFromPolicy);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* n) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); return nullptr; }
            return argv[++i];
        };
        if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else if (k == "--topic-rc") { auto v = need("--topic-rc"); if (!v) return false; a.topic_robot_control = v; }
        else if (k == "--topic-ms") { auto v = need("--topic-ms"); if (!v) return false; a.topic_motor_state = v; }
        else if (k == "--topic-mc") { auto v = need("--topic-mc"); if (!v) return false; a.topic_motor_control = v; }
        else if (k == "--uds-rc") { auto v = need("--uds-rc"); if (!v) return false; a.uds_robot_control = v; }
        else if (k == "--uds-ms-out") { auto v = need("--uds-ms-out"); if (!v) return false; a.uds_motor_state_out = v; }
        else if (k == "--uds-mc-in") { auto v = need("--uds-mc-in"); if (!v) return false; a.uds_motor_control_in = v; }
        else if (k == "--poll-hz") { auto v = need("--poll-hz"); if (!v) return false; a.poll_hz = std::atof(v); }
        else if (k == "--allow-real-hw") { a.allow_real_hw = true; }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); return false; }
    }
    if (a.poll_hz <= 0.0) { std::fprintf(stderr, "--poll-hz must be > 0\n"); return false; }
    return true;
}

// ─── DDS -> wire-frame copiers ─────────────────────────────────────────────
void fill_robot_control_frame(const RobotControl::Control& src,
                              uint64_t bridge_seq,
                              g0_pt::RobotControlFrame& dst) {
    using namespace std::chrono;
    dst.header.magic = g0_pt::kFrameMagicRobotControl;
    dst.header.version = g0_pt::kFrameVersion;
    dst.header.payload_count = 1;
    dst.header.timestamp_ns = static_cast<uint64_t>(
        duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count());
    dst.header.sequence_id = bridge_seq;

    dst.mode = static_cast<uint8_t>(src.mode());
    const auto& body = src.body();
    dst.body.roll  = body.attitude().roll();
    dst.body.pitch = body.attitude().pitch();
    dst.body.yaw   = body.attitude().yaw();
    dst.body.lin_x = body.linear_velocity().x();
    dst.body.lin_y = body.linear_velocity().y();
    dst.body.lin_z = body.linear_velocity().z();
    dst.body.ang_x = body.angular_velocity().x();
    dst.body.ang_y = body.angular_velocity().y();
    dst.body.ang_z = body.angular_velocity().z();
    dst.body.height = body.height();

    auto pack_gripper = [](const RobotControl::GripperState& g,
                           g0_pt::GripperWire& w) {
        w.is_valid = 1;
        w.roll  = g.attitude().roll();
        w.pitch = g.attitude().pitch();
        w.yaw   = g.attitude().yaw();
        w.lin_x = g.linear_velocity().x();
        w.lin_y = g.linear_velocity().y();
        w.lin_z = g.linear_velocity().z();
        w.opening = g.opening();
    };
    std::memset(&dst.left_gripper, 0, sizeof(dst.left_gripper));
    std::memset(&dst.right_gripper, 0, sizeof(dst.right_gripper));
    dst.has_left_gripper = 0;
    dst.has_right_gripper = 0;
    dst.reserved = 0;
    if (src.left_gripper().has_value()) {
        dst.has_left_gripper = 1;
        pack_gripper(*src.left_gripper(), dst.left_gripper);
    }
    if (src.right_gripper().has_value()) {
        dst.has_right_gripper = 1;
        pack_gripper(*src.right_gripper(), dst.right_gripper);
    }
}

void fill_motor_state_frame(const MotorControl::State& src,
                            uint64_t bridge_seq,
                            g0_pt::MotorStateFrame& dst) {
    dst.header.magic = g0_pt::kFrameMagicMotorState;
    dst.header.version = g0_pt::kFrameVersion;
    dst.header.payload_count = g0_pt::kNumMotors;
    dst.header.timestamp_ns = src.source_timestamp_ns();
    dst.header.sequence_id = bridge_seq;

    const auto& m = src.motors();
    for (int i = 0; i < g0_pt::kNumMotors; ++i) {
        dst.motors[i].isvalid    = m[i].isvalid();
        dst.motors[i].pos        = m[i].pos();
        dst.motors[i].dq         = m[i].dq();
        dst.motors[i].tau        = m[i].tau();
        dst.motors[i].status     = m[i].status();
        dst.motors[i].fpc_temper = m[i].fpc_temper();
        dst.motors[i].pcb_temper = m[i].pcb_temper();
    }
    const auto& imu = src.imu();
    dst.imu.imu_data_valid = imu.imu_data_valid();
    dst.imu.mag_data_valid = imu.mag_data_valid();
    dst.imu.imu_timestamp  = imu.imu_timestamp();
    dst.imu.mag_timestamp  = imu.mag_timestamp();
    dst.imu.acc_x = imu.acc().x(); dst.imu.acc_y = imu.acc().y(); dst.imu.acc_z = imu.acc().z();
    dst.imu.gyro_x = imu.gyro().x(); dst.imu.gyro_y = imu.gyro().y(); dst.imu.gyro_z = imu.gyro().z();
    dst.imu.mag_x = imu.mag().x();   dst.imu.mag_y = imu.mag().y();   dst.imu.mag_z = imu.mag().z();
}

// ─── wire-frame -> DDS for python -> motor_control ────────────────────────
bool decode_motor_control_frame(const uint8_t* bytes,
                                MotorControl::Control& out) {
    g0_pt::MotorControlFrame f;
    std::memcpy(&f, bytes, sizeof(f));
    if (f.header.magic != g0_pt::kFrameMagicMotorControl) {
        std::fprintf(stderr,
            "[policy_bridge] motor_control frame magic mismatch: 0x%08x\n",
            f.header.magic);
        return false;
    }
    if (f.header.version != g0_pt::kFrameVersion) {
        std::fprintf(stderr,
            "[policy_bridge] motor_control frame version mismatch: %u\n",
            unsigned(f.header.version));
        return false;
    }
    if (f.header.payload_count != g0_pt::kNumMotors) {
        std::fprintf(stderr,
            "[policy_bridge] motor_control frame motor_count != 22: %u\n",
            unsigned(f.header.payload_count));
        return false;
    }
    out.timestamp_ns(f.header.timestamp_ns);
    out.sequence_id(static_cast<uint16_t>(f.header.sequence_id & 0xFFFFu));
    out.motor_count(g0_pt::kNumMotors);
    auto& motors = out.motors();
    for (int i = 0; i < g0_pt::kNumMotors; ++i) {
        motors[i].pos(f.motors[i].pos);
        motors[i].dq (f.motors[i].dq);
        motors[i].kp (f.motors[i].kp);
        motors[i].kd (f.motors[i].kd);
        motors[i].tau(f.motors[i].tau);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Args a;
    if (!parse(argc, argv, a)) { usage(argv[0]); return 2; }

    if (a.allow_real_hw) {
        std::printf("[policy_bridge] *** --allow-real-hw set: mc/ topics permitted ***\n");
        g0_pt::set_allow_real_hw_topics(true);
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    std::printf("[policy_bridge] topic_rc=%s  topic_ms=%s  topic_mc=%s\n",
                a.topic_robot_control.c_str(), a.topic_motor_state.c_str(),
                a.topic_motor_control.c_str());
    std::printf("[policy_bridge] uds_rc=%s\n", a.uds_robot_control.c_str());
    std::printf("[policy_bridge] uds_ms_out=%s\n", a.uds_motor_state_out.c_str());
    std::printf("[policy_bridge] uds_mc_in=%s\n", a.uds_motor_control_in.c_str());
    std::printf("[policy_bridge] poll_hz=%g\n", a.poll_hz);

    g0_pt::RobotControlSubscriber sub_rc(a.topic_robot_control);
    g0_pt::MotorStateSubscriber   sub_ms(a.topic_motor_state);
    g0_pt::MotorControlPublisher  pub_mc(a.topic_motor_control);

    g0_pt::UdsBroadcaster out_rc(a.uds_robot_control,    "policy_bridge:rc");
    g0_pt::UdsBroadcaster out_ms(a.uds_motor_state_out,  "policy_bridge:ms");
    g0_pt::UdsReceiver    in_mc (a.uds_motor_control_in, "policy_bridge:mc",
                                 sizeof(g0_pt::MotorControlFrame));

    const auto period = std::chrono::duration<double>(1.0 / a.poll_hz);
    auto next_tick = std::chrono::steady_clock::now();

    uint64_t seq_rc_out = 0, seq_ms_out = 0;
    uint64_t cnt_rc_in = 0, cnt_ms_in = 0, cnt_mc_out = 0;

    RobotControl::Control   rc_msg;
    MotorControl::State     ms_msg;
    g0_pt::RobotControlFrame rc_frame{};
    g0_pt::MotorStateFrame   ms_frame{};

    while (!g_stop.load()) {
        out_rc.accept_pending();
        out_ms.accept_pending();
        in_mc.accept_pending();

        // DDS -> UDS broadcasts.
        while (sub_rc.poll(rc_msg)) {
            ++cnt_rc_in;
            fill_robot_control_frame(rc_msg, ++seq_rc_out, rc_frame);
            out_rc.broadcast(&rc_frame, sizeof(rc_frame));
        }
        while (sub_ms.poll(ms_msg)) {
            ++cnt_ms_in;
            fill_motor_state_frame(ms_msg, ++seq_ms_out, ms_frame);
            out_ms.broadcast(&ms_frame, sizeof(ms_frame));
        }

        // UDS -> DDS publish.
        in_mc.drain([&](const uint8_t* bytes) {
            MotorControl::Control mc_out;
            if (decode_motor_control_frame(bytes, mc_out)) {
                pub_mc.publish(mc_out);
                ++cnt_mc_out;
            }
        });

        if ((cnt_rc_in + cnt_ms_in + cnt_mc_out) % 500 == 1) {
            std::printf("[policy_bridge] rc_in=%lu  ms_in=%lu  mc_out=%lu  "
                        "clients(rc/ms)=%zu/%zu  mc_uds_connected=%d\n",
                        (unsigned long)cnt_rc_in, (unsigned long)cnt_ms_in,
                        (unsigned long)cnt_mc_out,
                        out_rc.client_count(), out_ms.client_count(),
                        in_mc.connected() ? 1 : 0);
        }

        next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        std::this_thread::sleep_until(next_tick);
    }

    std::printf("\n[policy_bridge] stopping. rc_in=%lu ms_in=%lu mc_out=%lu\n",
                (unsigned long)cnt_rc_in, (unsigned long)cnt_ms_in,
                (unsigned long)cnt_mc_out);
    return 0;
}
