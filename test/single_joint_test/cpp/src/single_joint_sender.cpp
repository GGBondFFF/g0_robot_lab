// Virtual single-joint DDS sender for the Isaac Lab test.
//
// Fills a full 22-slot MotorControl::Control frame matching motion-planner's
// wire format (5 fields per motor: pos, dq, kp, kd, tau). Only the target
// motor slot carries a non-zero command; the other 21 slots are zeroed.
//
// Wire units (matches the real motor and motion-planner's dds_executor):
//   pos:  degrees
//   dq:   degrees / second
//   tau:  N*m
//   kp:   N*m / deg            (torque per degree of position error)
//   kd:   N*m / (deg/s)        (torque per deg/s of velocity error)
// These are the firmware's PD-gain units; the Isaac Lab receiver computes
//   tau_out = kp*(pos_cmd_deg - pos_actual_deg)
//           + kd*(dq_cmd_dps  - dq_actual_dps)
//           + tau_ff
// directly in degrees, so the same numeric kp/kd you use on the real motor
// will produce the same closed-loop response in sim.
//
// NOTE: g0.py's per-joint stiffness/damping are in Isaac Lab units
//   (N*m/rad and N*m/(rad/s)). To get the equivalent deg-domain gain,
//   divide by (180/pi) ~= 57.296. Example: hip_pitch stiffness 4.0 N*m/rad
//   -> kp_deg = 4.0 / 57.296 ~= 0.0698 N*m/deg.
//
// CLI inputs use the SAME units as the wire — no rad/deg conversion happens
// here. (This sender talks the motor's native language.)
//
// Sign convention:
//   pos, dq, tau   are signed quantities about the joint axis. The CLI value
//                  is interpreted in the *right-hand-rule* (robot body
//                  frame, +X forward). It is multiplied by
//                  sim_sign_observed before going on the wire so that the
//                  observed motion in Isaac Lab matches RHR.
//   kp, kd         are scalar gains -> NOT sign-converted.
//
// Sender-side control flags (NOT part of the motor frame):
//   --duration-sec : 0 = run until Ctrl-C (default), >0 = stop after that long
//   --dry-run      : true = print only, no DDS publish (default false)
//   --rate-hz      : per-frame send rate
//   --full-print-every : how often to dump the full 22-row table
//   --topic        : DDS topic name (must start with g0_sim/ for publish)
//
// Safety gates:
//   * Any topic starting with "mc/" is rejected (real-hardware prefix).
//   * --dry-run false additionally requires the topic to start with "g0_sim/".

#include "joint_mapping.hpp"
#include "mbus_sim_client.hpp"

#include <idl_motor_control.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

namespace {

constexpr const char* kDefaultTopic = "g0_sim/motor_control_virtual";

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

struct Args {
    int motor_id = 0;
    // Right-hand-rule inputs for the target motor (non-target motors get 0).
    // ALL angles in degrees (matches real motor wire); torque in N*m.
    double pos_deg = 0.0;
    double dq_dps  = 0.0;
    double tau_nm  = 0.0;
    double kp = 0.0;
    double kd = 0.0;
    // Shortcut: --mode <pos|dq|tau> --value <float>
    std::string mode;
    bool mode_given = false;
    double value = 0.0;
    bool value_given = false;
    bool pos_given = false, dq_given = false, tau_given = false;

    double rate_hz = 50.0;
    double duration_sec = 0.0;  // 0 = run until Ctrl-C
    bool dry_run = false;
    std::string topic = kDefaultTopic;
    int full_print_every = 0;   // 0 = first frame only
};

void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s --motor-id <1..22> [--pos|--dq|--tau ... --kp ... --kd ...]\n"
        "\n"
        "Motor frame (right-hand-rule, in motor wire units — DEGREES):\n"
        "  --motor-id <int>            Target motor id (1..22), REQUIRED.\n"
        "  --pos <deg>                 Target position    (default 0)\n"
        "  --dq  <deg/s>               Target velocity    (default 0)\n"
        "  --tau <N*m>                 Feed-forward torque (default 0)\n"
        "  --kp  <N*m/deg>             Position gain      (default 0)\n"
        "  --kd  <N*m/(deg/s)>         Velocity gain      (default 0)\n"
        "Shortcut (mutually exclusive with the explicit field flag):\n"
        "  --mode {pos|dq|tau}         Which field to populate (with --value)\n"
        "  --value <float>             Value for the field named by --mode\n"
        "\n"
        "Sender control (NOT part of the motor frame):\n"
        "  --rate-hz <float>           Send rate (default 50)\n"
        "  --duration-sec <float>      0 = run until Ctrl-C (default), >0 = stop after\n"
        "  --dry-run {true|false}      Default false (publish to DDS). true = no DDS\n"
        "  --topic <name>              DDS topic (default %s).\n"
        "                                Must start with g0_sim/ when --dry-run false.\n"
        "  --full-print-every <N>      Re-dump the full 22-row table every N frames.\n"
        "                                0 = first frame only (default).\n",
        argv0, kDefaultTopic);
}

bool parse_bool(const std::string& s, bool& out) {
    if (s == "true" || s == "1")  { out = true;  return true; }
    if (s == "false" || s == "0") { out = false; return true; }
    return false;
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); return nullptr; }
            return argv[++i];
        };
        if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else if (k == "--motor-id")        { auto v = need("--motor-id");        if (!v) return false; a.motor_id = std::atoi(v); }
        else if (k == "--pos")             { auto v = need("--pos");             if (!v) return false; a.pos_deg = std::atof(v); a.pos_given = true; }
        else if (k == "--dq")              { auto v = need("--dq");              if (!v) return false; a.dq_dps  = std::atof(v); a.dq_given  = true; }
        else if (k == "--tau")             { auto v = need("--tau");             if (!v) return false; a.tau_nm  = std::atof(v); a.tau_given = true; }
        else if (k == "--kp")              { auto v = need("--kp");              if (!v) return false; a.kp = std::atof(v); }
        else if (k == "--kd")              { auto v = need("--kd");              if (!v) return false; a.kd = std::atof(v); }
        else if (k == "--mode")            { auto v = need("--mode");            if (!v) return false; a.mode = v; a.mode_given = true; }
        else if (k == "--value")           { auto v = need("--value");           if (!v) return false; a.value = std::atof(v); a.value_given = true; }
        else if (k == "--rate-hz")         { auto v = need("--rate-hz");         if (!v) return false; a.rate_hz = std::atof(v); }
        else if (k == "--duration-sec")    { auto v = need("--duration-sec");    if (!v) return false; a.duration_sec = std::atof(v); }
        else if (k == "--dry-run")         { auto v = need("--dry-run");         if (!v) return false; if (!parse_bool(v, a.dry_run)) { std::fprintf(stderr, "bad --dry-run\n"); return false; } }
        else if (k == "--topic")           { auto v = need("--topic");           if (!v) return false; a.topic = v; }
        else if (k == "--full-print-every"){ auto v = need("--full-print-every");if (!v) return false; a.full_print_every = std::atoi(v); }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); return false; }
    }
    if (a.motor_id < 1 || a.motor_id > 22) { std::fprintf(stderr, "--motor-id must be 1..22\n"); return false; }
    if (a.rate_hz <= 0 || a.duration_sec < 0) { std::fprintf(stderr, "--rate-hz must be >0, --duration-sec >=0\n"); return false; }

    if (a.mode_given != a.value_given) {
        std::fprintf(stderr, "--mode and --value must be specified together\n"); return false;
    }
    if (a.mode_given) {
        if (a.mode == "pos") {
            if (a.pos_given) { std::fprintf(stderr, "cannot give both --pos and (--mode pos --value)\n"); return false; }
            a.pos_deg = a.value; a.pos_given = true;
        } else if (a.mode == "dq") {
            if (a.dq_given) { std::fprintf(stderr, "cannot give both --dq and (--mode dq --value)\n"); return false; }
            a.dq_dps = a.value; a.dq_given = true;
        } else if (a.mode == "tau") {
            if (a.tau_given) { std::fprintf(stderr, "cannot give both --tau and (--mode tau --value)\n"); return false; }
            a.tau_nm = a.value; a.tau_given = true;
        } else {
            std::fprintf(stderr, "--mode must be pos|dq|tau\n"); return false;
        }
    }
    return true;
}

// Sign-convert vector quantities (pos, dq, tau). kp/kd are NOT sign-converted.
struct WireCmd {
    float pos_deg;
    float dq_dps;
    float tau_nm;
    float kp;
    float kd;
};

WireCmd to_wire(int motor_id, const Args& a) {
    const int s = g0_sjt::sim_sign_for_motor(motor_id);
    return WireCmd{
        static_cast<float>(s * a.pos_deg),
        static_cast<float>(s * a.dq_dps),
        static_cast<float>(s * a.tau_nm),
        static_cast<float>(a.kp),
        static_cast<float>(a.kd),
    };
}

void zero_motor(MotorControl::MotorCmd& m) {
    m.pos(0.0f); m.dq(0.0f); m.kp(0.0f); m.kd(0.0f); m.tau(0.0f);
}

void write_motor(MotorControl::MotorCmd& m, const WireCmd& w) {
    m.pos(w.pos_deg); m.dq(w.dq_dps); m.kp(w.kp); m.kd(w.kd); m.tau(w.tau_nm);
}

void build_frame(MotorControl::Control& ctrl, const Args& a, uint16_t seq) {
    auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    ctrl.timestamp_ns(static_cast<uint64_t>(now_ns));
    ctrl.sequence_id(seq);
    ctrl.motor_count(g0_sjt::kNumMotors);

    auto& motors = ctrl.motors();
    for (auto& m : motors) zero_motor(m);
    write_motor(motors[a.motor_id - 1], to_wire(a.motor_id, a));
}

void print_header(const Args& a) {
    const auto& e = g0_sjt::get_entry(a.motor_id);
    const WireCmd w = to_wire(a.motor_id, a);
    std::printf("=== single_joint_sender ===\n");
    std::printf("topic         : %s\n", a.topic.c_str());
    std::printf("dry_run       : %s\n", a.dry_run ? "true" : "false");
    std::printf("motor_id      : %d\n", a.motor_id);
    std::printf("joint_name    : %s\n", e.joint_name);
    std::printf("sim_sign      : %+d\n", e.sim_sign_observed);
    std::printf("rhr  in       : pos=%+.4f deg   dq=%+.4f deg/s   tau=%+.4f N*m   kp=%+.4f   kd=%+.4f\n",
        a.pos_deg, a.dq_dps, a.tau_nm, a.kp, a.kd);
    std::printf("wire (sent)   : pos=%+.4f deg   dq=%+.4f deg/s   tau=%+.4f N*m   kp=%+.4f   kd=%+.4f\n",
        w.pos_deg, w.dq_dps, w.tau_nm, w.kp, w.kd);
    std::printf("rate_hz       : %g\n", a.rate_hz);
    if (a.duration_sec == 0.0)
        std::printf("duration_sec  : 0 (run until Ctrl-C)\n");
    else
        std::printf("duration_sec  : %g\n", a.duration_sec);
    std::printf("===========================\n\n");
}

void print_full_frame(const MotorControl::Control& ctrl) {
    std::printf("[frame seq=%u ts_ns=%llu] (wire fields shown; deg, deg/s, N*m)\n",
        static_cast<unsigned>(ctrl.sequence_id()),
        static_cast<unsigned long long>(ctrl.timestamp_ns()));
    std::printf(" %-3s %-26s %-4s %10s %10s %10s %8s %8s\n",
        "id", "joint_name", "sgn", "pos", "dq", "tau", "kp", "kd");
    for (int mid = 1; mid <= g0_sjt::kNumMotors; ++mid) {
        const auto& e = g0_sjt::get_entry(mid);
        const auto& m = ctrl.motors()[mid - 1];
        std::printf(" %-3d %-26s %+4d %+10.4f %+10.4f %+10.4f %+8.4f %+8.4f\n",
            mid, e.joint_name, e.sim_sign_observed,
            m.pos(), m.dq(), m.tau(), m.kp(), m.kd());
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) { usage(argv[0]); return 2; }

    if (a.topic.rfind("mc/", 0) == 0) {
        std::fprintf(stderr,
            "ERROR: topic '%s' looks like a real-hardware topic (mc/...).\n"
            "       This test must use a virtual topic (default %s).\n",
            a.topic.c_str(), kDefaultTopic);
        return 4;
    }
    if (!a.dry_run && a.topic.rfind("g0_sim/", 0) != 0) {
        std::fprintf(stderr,
            "ERROR: --dry-run false requires the topic to start with 'g0_sim/'.\n"
            "       Got: '%s'. Default is %s.\n",
            a.topic.c_str(), kDefaultTopic);
        return 4;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    std::unique_ptr<g0_sjt::MbusSimPublisher> pub;
    if (!a.dry_run) {
        try {
            pub.reset(new g0_sjt::MbusSimPublisher(a.topic));
            std::printf("[sender] DDS publisher initialized for topic '%s'\n",
                        a.topic.c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ERROR: failed to init DDS publisher: %s\n", e.what());
            return 5;
        }
    }

    print_header(a);

    const bool infinite = (a.duration_sec == 0.0);
    const long total_frames = infinite ? 0 : static_cast<long>(a.rate_hz * a.duration_sec);
    const auto period = std::chrono::duration<double>(1.0 / a.rate_hz);
    auto next_tick = std::chrono::steady_clock::now();

    MotorControl::Control ctrl;
    long i = 0;
    while ((infinite || i < total_frames) && !g_stop.load()) {
        build_frame(ctrl, a, static_cast<uint16_t>(i & 0xFFFF));

        if (pub) pub->publish(ctrl);

        const bool full =
            (i == 0) ||
            (a.full_print_every > 0 && (i % a.full_print_every) == 0);
        if (full) {
            print_full_frame(ctrl);
        } else {
            const auto& m = ctrl.motors()[a.motor_id - 1];
            std::printf("[seq=%4u] slot %2d (%s) -> pos=%+.4f deg dq=%+.4f deg/s tau=%+.4f kp=%+.4f kd=%+.4f\n",
                static_cast<unsigned>(ctrl.sequence_id()),
                a.motor_id - 1,
                g0_sjt::motor_id_to_joint_name(a.motor_id).c_str(),
                m.pos(), m.dq(), m.tau(), m.kp(), m.kd());
        }

        next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        std::this_thread::sleep_until(next_tick);
        ++i;
    }

    std::printf("\nDone. %ld frames emitted (%s)%s.\n", i,
                pub ? "DDS published" : "dry-run, no DDS publish",
                g_stop.load() ? " [stopped by signal]" : "");
    return 0;
}
