// plant_state_bridge — Isaac Lab plant side bridge.
//
// Connects the Isaac Lab process to DDS:
//
//   DDS  g0_sim/motor_control  --sub--> UDS broadcast  (MotorControlFrame)
//   DDS  g0_sim/motor_state    <--pub-- UDS receive    (MotorStateFrame)
//
// Mirrors policy_control_bridge but in the opposite direction. The Python
// plant side reads commanded motor frames out of the UDS, advances the Isaac
// sim, then writes a MotorStateFrame back so the policy can close the loop.

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
    std::string topic_motor_control = g0_pt::kDdsTopicMotorControl;
    std::string topic_motor_state   = g0_pt::kDdsTopicMotorState;
    std::string uds_motor_control_out = g0_pt::kUdsMotorControlToPlant;
    std::string uds_motor_state_in    = g0_pt::kUdsMotorStateFromPlant;
    double poll_hz = 200.0;
};

void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "DDS topics (must start with g0_sim/):\n"
        "  --topic-mc <name>   default %s\n"
        "  --topic-ms <name>   default %s\n"
        "UDS paths:\n"
        "  --uds-mc-out  <path>   bridge -> python MotorControlFrame  (default %s)\n"
        "  --uds-ms-in   <path>   python -> bridge MotorStateFrame    (default %s)\n"
        "  --poll-hz     <float>  bridge inner loop rate              (default 200)\n",
        argv0,
        g0_pt::kDdsTopicMotorControl, g0_pt::kDdsTopicMotorState,
        g0_pt::kUdsMotorControlToPlant, g0_pt::kUdsMotorStateFromPlant);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* n) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); return nullptr; }
            return argv[++i];
        };
        if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else if (k == "--topic-mc")    { auto v = need("--topic-mc");    if (!v) return false; a.topic_motor_control = v; }
        else if (k == "--topic-ms")    { auto v = need("--topic-ms");    if (!v) return false; a.topic_motor_state = v; }
        else if (k == "--uds-mc-out")  { auto v = need("--uds-mc-out");  if (!v) return false; a.uds_motor_control_out = v; }
        else if (k == "--uds-ms-in")   { auto v = need("--uds-ms-in");   if (!v) return false; a.uds_motor_state_in = v; }
        else if (k == "--poll-hz")     { auto v = need("--poll-hz");     if (!v) return false; a.poll_hz = std::atof(v); }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); return false; }
    }
    if (a.poll_hz <= 0.0) { std::fprintf(stderr, "--poll-hz must be > 0\n"); return false; }
    return true;
}

void fill_motor_control_frame(const MotorControl::Control& src,
                              uint64_t bridge_seq,
                              g0_pt::MotorControlFrame& dst) {
    dst.header.magic = g0_pt::kFrameMagicMotorControl;
    dst.header.version = g0_pt::kFrameVersion;
    dst.header.payload_count = g0_pt::kNumMotors;
    dst.header.timestamp_ns = src.timestamp_ns();
    dst.header.sequence_id = bridge_seq;
    const auto& m = src.motors();
    for (int i = 0; i < g0_pt::kNumMotors; ++i) {
        dst.motors[i].pos = m[i].pos();
        dst.motors[i].dq  = m[i].dq();
        dst.motors[i].kp  = m[i].kp();
        dst.motors[i].kd  = m[i].kd();
        dst.motors[i].tau = m[i].tau();
    }
}

bool decode_motor_state_frame(const uint8_t* bytes,
                              MotorControl::State& out) {
    g0_pt::MotorStateFrame f;
    std::memcpy(&f, bytes, sizeof(f));
    if (f.header.magic != g0_pt::kFrameMagicMotorState) {
        std::fprintf(stderr,
            "[plant_bridge] motor_state frame magic mismatch: 0x%08x\n",
            f.header.magic);
        return false;
    }
    if (f.header.version != g0_pt::kFrameVersion) {
        std::fprintf(stderr,
            "[plant_bridge] motor_state frame version mismatch: %u\n",
            unsigned(f.header.version));
        return false;
    }
    if (f.header.payload_count != g0_pt::kNumMotors) {
        std::fprintf(stderr,
            "[plant_bridge] motor_state frame motor_count != 22: %u\n",
            unsigned(f.header.payload_count));
        return false;
    }
    out.sequence_id(static_cast<uint16_t>(f.header.sequence_id & 0xFFFFu));
    out.motor_count(g0_pt::kNumMotors);
    out.source_timestamp_ns(f.header.timestamp_ns);

    auto& motors = out.motors();
    for (int i = 0; i < g0_pt::kNumMotors; ++i) {
        motors[i].isvalid(f.motors[i].isvalid);
        motors[i].pos(f.motors[i].pos);
        motors[i].dq(f.motors[i].dq);
        motors[i].tau(f.motors[i].tau);
        motors[i].status(f.motors[i].status);
        motors[i].fpc_temper(f.motors[i].fpc_temper);
        motors[i].pcb_temper(f.motors[i].pcb_temper);
    }

    auto& imu = out.imu();
    imu.imu_data_valid(f.imu.imu_data_valid);
    imu.mag_data_valid(f.imu.mag_data_valid);
    imu.imu_timestamp(f.imu.imu_timestamp);
    imu.mag_timestamp(f.imu.mag_timestamp);
    imu.acc(::Common::Accelerometer3D(f.imu.acc_x, f.imu.acc_y, f.imu.acc_z));
    imu.gyro(::Common::Velocity3D(f.imu.gyro_x, f.imu.gyro_y, f.imu.gyro_z));
    imu.mag(::Common::Magnetometer3D(f.imu.mag_x, f.imu.mag_y, f.imu.mag_z));
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Args a;
    if (!parse(argc, argv, a)) { usage(argv[0]); return 2; }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    std::printf("[plant_bridge] topic_mc=%s  topic_ms=%s\n",
                a.topic_motor_control.c_str(), a.topic_motor_state.c_str());
    std::printf("[plant_bridge] uds_mc_out=%s\n", a.uds_motor_control_out.c_str());
    std::printf("[plant_bridge] uds_ms_in=%s\n",  a.uds_motor_state_in.c_str());
    std::printf("[plant_bridge] poll_hz=%g\n", a.poll_hz);

    g0_pt::MotorControlSubscriber sub_mc(a.topic_motor_control);
    g0_pt::MotorStatePublisher    pub_ms(a.topic_motor_state);

    g0_pt::UdsBroadcaster out_mc(a.uds_motor_control_out, "plant_bridge:mc");
    g0_pt::UdsReceiver    in_ms (a.uds_motor_state_in,    "plant_bridge:ms",
                                 sizeof(g0_pt::MotorStateFrame));

    const auto period = std::chrono::duration<double>(1.0 / a.poll_hz);
    auto next_tick = std::chrono::steady_clock::now();

    uint64_t seq_mc_out = 0;
    uint64_t cnt_mc_in = 0, cnt_ms_out = 0;

    MotorControl::Control mc_msg;
    MotorControl::State   ms_msg;
    g0_pt::MotorControlFrame mc_frame{};

    while (!g_stop.load()) {
        out_mc.accept_pending();
        in_ms.accept_pending();

        // DDS -> UDS broadcast.
        while (sub_mc.poll(mc_msg)) {
            ++cnt_mc_in;
            fill_motor_control_frame(mc_msg, ++seq_mc_out, mc_frame);
            out_mc.broadcast(&mc_frame, sizeof(mc_frame));
        }

        // UDS -> DDS publish.
        in_ms.drain([&](const uint8_t* bytes) {
            if (decode_motor_state_frame(bytes, ms_msg)) {
                pub_ms.publish(ms_msg);
                ++cnt_ms_out;
            }
        });

        if ((cnt_mc_in + cnt_ms_out) % 500 == 1) {
            std::printf("[plant_bridge] mc_in=%lu  ms_out=%lu  "
                        "clients(mc)=%zu  ms_uds_connected=%d\n",
                        (unsigned long)cnt_mc_in, (unsigned long)cnt_ms_out,
                        out_mc.client_count(),
                        in_ms.connected() ? 1 : 0);
        }

        next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        std::this_thread::sleep_until(next_tick);
    }

    std::printf("\n[plant_bridge] stopping. mc_in=%lu ms_out=%lu\n",
                (unsigned long)cnt_mc_in, (unsigned long)cnt_ms_out);
    return 0;
}
