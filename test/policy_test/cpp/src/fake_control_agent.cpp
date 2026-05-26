// fake_control_agent — TOML-driven RobotControl::Control publisher for the
// policy_test virtual stack.
//
// Mirrors motion-planner/test/fake_control_agent/main.cpp but only the
// config-file mode (no keyboard, no evdev), and locked onto the sim-only
// topic g0_sim/robot_control. The bridge in policy_control_bridge subscribes
// to this topic and forwards it to the Python policy over UDS.
//
// Usage:
//   fake_control_agent [--config <path>] [--topic g0_sim/robot_control]
//
// TOML schema is the same as motion-planner's fake_control_agent.toml:
//   [publish] rate_hz, loop_count
//   [body]    vx vy vz (m/s, normalized for the velocity policy),
//             roll pitch yaw (deg), yaw_rate (deg/s)
//   [left_gripper]  enabled opening vx vy vz roll pitch yaw
//   [right_gripper] enabled opening vx vy vz roll pitch yaw
//
// Wire field convention matches motion-planner exactly: roll/pitch/yaw are
// converted DEG -> RAD before being written into Common::AttitudeAngles, and
// yaw_rate is converted DEG/s -> RAD/s into BodyState.angular_velocity.z.

#include "mbus_sim_io.hpp"
#include "sim_frame_proto.hpp"

#include <toml++/toml.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

constexpr double kDeg2Rad = M_PI / 180.0;

std::atomic<bool> g_running{true};
void on_signal(int) { g_running.store(false); }

struct GripperCfg {
    bool  enabled = false;
    float opening = 0.0f;
    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    float roll = 0.0f, pitch = 0.0f, yaw = 0.0f;
};

struct Config {
    double rate_hz = 50.0;
    long   loop_count = -1;
    float vx = 0.0f, vy = 0.0f, vz = 0.0f;
    float roll = 0.0f, pitch = 0.0f, yaw = 0.0f;
    float yaw_rate = 0.0f;
    GripperCfg left_gripper;
    GripperCfg right_gripper;
};

GripperCfg parse_gripper(const toml::table* tbl) {
    GripperCfg g;
    if (!tbl) return g;
    g.enabled = (*tbl)["enabled"].value_or(false);
    g.opening = static_cast<float>((*tbl)["opening"].value_or(0.0));
    g.vx      = static_cast<float>((*tbl)["vx"].value_or(0.0));
    g.vy      = static_cast<float>((*tbl)["vy"].value_or(0.0));
    g.vz      = static_cast<float>((*tbl)["vz"].value_or(0.0));
    g.roll    = static_cast<float>((*tbl)["roll"].value_or(0.0));
    g.pitch   = static_cast<float>((*tbl)["pitch"].value_or(0.0));
    g.yaw     = static_cast<float>((*tbl)["yaw"].value_or(0.0));
    return g;
}

Config load_config(const std::string& path) {
    toml::table tbl;
    try {
        tbl = toml::parse_file(path);
    } catch (const toml::parse_error& e) {
        throw std::runtime_error(std::string("TOML parse error: ") + e.what());
    }
    Config cfg;
    if (auto* pub = tbl.get_as<toml::table>("publish")) {
        cfg.rate_hz    = (*pub)["rate_hz"].value_or(50.0);
        cfg.loop_count = static_cast<long>((*pub)["loop_count"].value_or(int64_t{-1}));
    }
    if (auto* body = tbl.get_as<toml::table>("body")) {
        cfg.vx       = static_cast<float>((*body)["vx"].value_or(0.0));
        cfg.vy       = static_cast<float>((*body)["vy"].value_or(0.0));
        cfg.vz       = static_cast<float>((*body)["vz"].value_or(0.0));
        cfg.roll     = static_cast<float>((*body)["roll"].value_or(0.0));
        cfg.pitch    = static_cast<float>((*body)["pitch"].value_or(0.0));
        cfg.yaw      = static_cast<float>((*body)["yaw"].value_or(0.0));
        cfg.yaw_rate = static_cast<float>((*body)["yaw_rate"].value_or(0.0));
    }
    cfg.left_gripper  = parse_gripper(tbl.get_as<toml::table>("left_gripper"));
    cfg.right_gripper = parse_gripper(tbl.get_as<toml::table>("right_gripper"));
    return cfg;
}

RobotControl::Control build_message(const Config& cfg) {
    Common::AttitudeAngles attitude(
        static_cast<float>(cfg.roll  * kDeg2Rad),
        static_cast<float>(cfg.pitch * kDeg2Rad),
        static_cast<float>(cfg.yaw   * kDeg2Rad));
    Common::Velocity3D linear_vel(cfg.vx, cfg.vy, cfg.vz);
    Common::Velocity3D angular_vel(
        0.0f, 0.0f, static_cast<float>(cfg.yaw_rate * kDeg2Rad));
    RobotControl::BodyState body(attitude, linear_vel, angular_vel, 0.0f);

    auto make_gripper = [](const GripperCfg& g)
        -> std::optional<RobotControl::GripperState>
    {
        if (!g.enabled) return std::nullopt;
        Common::AttitudeAngles att(
            static_cast<float>(g.roll  * kDeg2Rad),
            static_cast<float>(g.pitch * kDeg2Rad),
            static_cast<float>(g.yaw   * kDeg2Rad));
        Common::Velocity3D vel(g.vx, g.vy, g.vz);
        return RobotControl::GripperState(true, att, vel, g.opening);
    };
    return RobotControl::Control(
        Common::RobotMode::HEXAPOD, body,
        make_gripper(cfg.left_gripper),
        make_gripper(cfg.right_gripper),
        std::nullopt, std::nullopt);
}

void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [--config <path>] [--topic <name>]\n"
        "  --config <path>   TOML config (default: fake_control_agent.toml)\n"
        "  --topic  <name>   DDS topic, must start with g0_sim/\n"
        "                    (default: g0_pt::kDdsTopicRobotControl = g0_sim/robot_control)\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    std::string config_path = "fake_control_agent.toml";
    std::string topic_name  = g0_pt::kDdsTopicRobotControl;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* n) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); return nullptr; }
            return argv[++i];
        };
        if (k == "-h" || k == "--help") { usage(argv[0]); return 0; }
        else if (k == "-c" || k == "--config") { auto v = need("--config"); if (!v) return 2; config_path = v; }
        else if (k == "--topic") { auto v = need("--topic"); if (!v) return 2; topic_name = v; }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); usage(argv[0]); return 2; }
    }

    Config cfg;
    try {
        cfg = load_config(config_path);
        std::printf("[fake_control_agent] loaded %s\n", config_path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "[fake_control_agent] WARN: could not load '%s': %s\n"
            "  using zero defaults.\n", config_path.c_str(), e.what());
    }

    g0_pt::RobotControlPublisher pub(topic_name);  // refuses mc/ prefix
    RobotControl::Control msg = build_message(cfg);

    std::printf("[fake_control_agent] topic=%s  rate=%.2f Hz  loop=%ld\n",
                topic_name.c_str(), cfg.rate_hz, cfg.loop_count);
    std::printf("  body linear_vel : vx=%.3f vy=%.3f vz=%.3f (wire as-is)\n",
                cfg.vx, cfg.vy, cfg.vz);
    std::printf("  body attitude   : roll=%.2f pitch=%.2f yaw=%.2f deg\n",
                cfg.roll, cfg.pitch, cfg.yaw);
    std::printf("  yaw_rate        : %.2f deg/s  (-> %.4f rad/s on wire)\n",
                cfg.yaw_rate, cfg.yaw_rate * kDeg2Rad);
    if (cfg.left_gripper.enabled)
        std::printf("  left_gripper    : opening=%.2f\n", cfg.left_gripper.opening);
    if (cfg.right_gripper.enabled)
        std::printf("  right_gripper   : opening=%.2f\n", cfg.right_gripper.opening);

    const double period_s = (cfg.rate_hz > 0.0) ? 1.0 / cfg.rate_hz : 0.02;
    const auto period_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(period_s));
    const bool infinite = (cfg.loop_count < 0);
    long published = 0;
    auto next = std::chrono::steady_clock::now();

    while (g_running.load() && (infinite || published < cfg.loop_count)) {
        pub.publish(msg);
        ++published;
        next += period_ns;
        std::this_thread::sleep_until(next);
    }
    std::printf("[fake_control_agent] published %ld packets. exit.\n", published);
    return 0;
}
