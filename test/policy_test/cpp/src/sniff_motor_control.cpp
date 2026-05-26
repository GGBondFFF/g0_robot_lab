// sniff_motor_control — passive DDS subscriber that dumps live
// MotorControl::Control frames as a human-readable table + raw hex.
//
// Runs alongside the policy_test stack without touching any UDS / bridge.
// Just subscribes to the same DDS topic the plant_state_bridge does, takes
// the configured number of frames, writes them to a log file, and exits.
//
// Usage:
//   sniff_motor_control [--topic g0_sim/motor_control]
//                       [--count 1]
//                       [--out <path>]
//                       [--skip 0]      # drop first N frames (warm-up)
//                       [--poll-hz 200]

#include "mbus_sim_io.hpp"
#include "sim_frame_proto.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

const char* kJointNames[22] = {
    "waist_yaw_joint", "waist_roll_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint", "l_elbow_pitch_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint", "r_elbow_pitch_joint",
    "l_hip_pitch_joint", "l_hip_roll_joint",
    "l_hip_yaw_joint", "l_knee_pitch_joint",
    "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint",
    "r_hip_yaw_joint", "r_knee_pitch_joint",
    "r_ankle_pitch_joint", "r_ankle_roll_joint",
};

struct Args {
    std::string topic = g0_pt::kDdsTopicMotorControl;
    std::string out;
    int count = 1;
    int skip = 0;
    double poll_hz = 200.0;
    bool allow_real_hw = false;
};

void usage(const char* a0) {
    std::fprintf(stderr,
        "Usage: %s [--topic <name>] [--count N] [--skip N] [--out <path>] [--poll-hz N]\n"
        "  Defaults: topic=%s  count=1  skip=0  poll-hz=200\n"
        "  --out default: ../docs/dds_action_live_<utc_ms>.log relative to this binary.\n",
        a0, g0_pt::kDdsTopicMotorControl);
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* n) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); return nullptr; }
            return argv[++i];
        };
        if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else if (k == "--topic")    { auto v = need("--topic");    if (!v) return false; a.topic = v; }
        else if (k == "--out")      { auto v = need("--out");      if (!v) return false; a.out = v; }
        else if (k == "--count")    { auto v = need("--count");    if (!v) return false; a.count = std::atoi(v); }
        else if (k == "--skip")     { auto v = need("--skip");     if (!v) return false; a.skip = std::atoi(v); }
        else if (k == "--poll-hz") { auto v = need("--poll-hz"); if (!v) return false; a.poll_hz = std::atof(v); }
        else if (k == "--allow-real-hw") { a.allow_real_hw = true; }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); return false; }
    }
    if (a.count <= 0) { std::fprintf(stderr, "--count must be > 0\n"); return false; }
    if (a.poll_hz <= 0.0) { std::fprintf(stderr, "--poll-hz must be > 0\n"); return false; }
    return true;
}

std::string format_table(const MotorControl::Control& ctrl) {
    std::ostringstream os;
    os << "# timestamp_ns = " << ctrl.timestamp_ns()
       << "   sequence_id = " << ctrl.sequence_id()
       << "   motor_count = " << ctrl.motor_count() << "\n";
    os << "motor_id | joint_name              "
       << "|     pos(deg) |    dq(deg/s) | kp(N*m/deg) "
       << "| kd(N*m/(deg/s)) | tau(N*m)\n";
    os << std::string(118, '-') << "\n";
    const auto& m = ctrl.motors();
    for (int i = 0; i < 22; ++i) {
        char line[256];
        std::snprintf(line, sizeof(line),
            "   %2d    | %-23s |  %+11.4f |  %+11.4f |  %9.6f |    %10.7f |  %+7.4f\n",
            i + 1, kJointNames[i],
            m[i].pos(), m[i].dq(), m[i].kp(), m[i].kd(), m[i].tau());
        os << line;
    }
    return os.str();
}

void encode_wire_hex(const MotorControl::Control& ctrl, std::ostringstream& os) {
    // Re-pack via our wire format so the hex matches what the policy emitted
    // pre-DDS-CDR (i.e. the in-memory g0_pt::MotorControlFrame). DDS uses
    // CDR on the wire — different from our internal frame — but the
    // semantic fields are identical, so this is the "policy-side wire".
    g0_pt::MotorControlFrame f;
    f.header.magic = g0_pt::kFrameMagicMotorControl;
    f.header.version = g0_pt::kFrameVersion;
    f.header.payload_count = 22;
    f.header.timestamp_ns = ctrl.timestamp_ns();
    f.header.sequence_id = ctrl.sequence_id();
    const auto& m = ctrl.motors();
    for (int i = 0; i < 22; ++i) {
        f.motors[i].pos = m[i].pos();
        f.motors[i].dq  = m[i].dq();
        f.motors[i].kp  = m[i].kp();
        f.motors[i].kd  = m[i].kd();
        f.motors[i].tau = m[i].tau();
    }
    const auto* bytes = reinterpret_cast<const unsigned char*>(&f);
    for (size_t i = 0; i < sizeof(f); ++i) {
        os << std::hex << std::setw(2) << std::setfill('0') << int(bytes[i]);
        if ((i + 1) % 32 == 0) os << "\n";
    }
    if (sizeof(f) % 32) os << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Args a;
    if (!parse(argc, argv, a)) { usage(argv[0]); return 2; }

    if (a.allow_real_hw) {
        std::printf("[sniff] *** --allow-real-hw set: mc/ topics permitted ***\n");
        g0_pt::set_allow_real_hw_topics(true);
    }

    if (a.out.empty()) {
        using namespace std::chrono;
        auto ms = duration_cast<milliseconds>(
            system_clock::now().time_since_epoch()).count();
        a.out = "../docs/dds_action_live_" + std::to_string(ms) + ".log";
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    std::printf("[sniff] topic = %s   count = %d   skip = %d   out = %s\n",
                a.topic.c_str(), a.count, a.skip, a.out.c_str());

    g0_pt::MotorControlSubscriber sub(a.topic);
    std::printf("[sniff] subscribed; waiting for frames...\n");

    std::ofstream fout(a.out);
    if (!fout.is_open()) {
        std::fprintf(stderr, "[sniff] could not open output: %s\n", a.out.c_str());
        return 3;
    }
    fout << "# Live DDS MotorControl::Control snapshot\n";
    fout << "# topic = " << a.topic << "\n";
    fout << "# skipped first " << a.skip << " frame(s), captured " << a.count
         << " frame(s) total.\n";
    fout << "# Wire units: pos=deg  dq=deg/s  kp=N*m/deg  "
            "kd=N*m/(deg/s)  tau=N*m\n";
    fout << "# Hex dump below each frame is the policy_test internal "
            "g0_pt::MotorControlFrame (464B), not the on-the-wire DDS CDR\n"
            "# (CDR adds a small header). The float fields per motor are "
            "identical to what the policy emitted.\n#\n";

    const auto period = std::chrono::duration<double>(1.0 / a.poll_hz);
    auto next_tick = std::chrono::steady_clock::now();

    MotorControl::Control ctrl;
    int seen = 0, captured = 0;

    while (!g_stop.load() && captured < a.count) {
        while (sub.poll(ctrl)) {
            ++seen;
            if (seen <= a.skip) continue;
            ++captured;
            std::string table = format_table(ctrl);
            std::ostringstream hex_os;
            encode_wire_hex(ctrl, hex_os);

            std::printf("\n[sniff] frame %d/%d  (sequence_id=%u)\n%s\n",
                        captured, a.count, unsigned(ctrl.sequence_id()),
                        table.c_str());

            fout << "\n# ───── frame " << captured << " / " << a.count
                 << " ─────\n";
            fout << table << "\n";
            fout << "# raw 464-byte g0_pt::MotorControlFrame (hex):\n";
            fout << hex_os.str();
            fout.flush();

            if (captured >= a.count) break;
        }
        next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        std::this_thread::sleep_until(next_tick);
    }

    std::printf("[sniff] captured %d frame(s); wrote %s\n", captured, a.out.c_str());
    return captured == a.count ? 0 : 1;
}
