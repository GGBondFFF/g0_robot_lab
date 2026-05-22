// DDS-to-UDS bridge for the Isaac Lab virtual single-joint test.
//
// Subscribes to a *virtual* MotorControl::Control DDS topic (default
// g0_sim/motor_control_virtual) and forwards each received frame, byte-for-
// byte intact in DDS wire units (degrees / N*m), over a Unix-domain
// SOCK_STREAM socket. Accepts multiple clients; broadcasts every frame to
// all of them. The Python receiver in this same test reads it.
//
// HARD-WIRED safety: refuses any topic name starting with "mc/" (real
// hardware prefix). Default UDS path is /tmp/g0_sim/motor_control_virtual.sock.

#include "joint_mapping.hpp"
#include "mbus_sim_client.hpp"
#include "sim_frame_proto.hpp"

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

struct BridgeArgs {
    std::string topic = "g0_sim/motor_control_virtual";
    std::string uds_path = g0_sjt::kDefaultUdsPath;
    double poll_hz = 200.0;   // poll faster than typical sender rate
};

void usage(const char* argv0) {
    std::fprintf(stderr,
        "Usage: %s [--topic <name>] [--uds <path>] [--poll-hz <float>]\n"
        "Defaults: topic=g0_sim/motor_control_virtual  uds=%s  poll-hz=200\n",
        argv0, g0_sjt::kDefaultUdsPath);
}

bool parse(int argc, char** argv, BridgeArgs& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const char* n) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", n); return nullptr; }
            return argv[++i];
        };
        if (k == "-h" || k == "--help") { usage(argv[0]); std::exit(0); }
        else if (k == "--topic")    { auto v = need("--topic");    if (!v) return false; a.topic = v; }
        else if (k == "--uds")      { auto v = need("--uds");      if (!v) return false; a.uds_path = v; }
        else if (k == "--poll-hz")  { auto v = need("--poll-hz");  if (!v) return false; a.poll_hz = std::atof(v); }
        else { std::fprintf(stderr, "unknown arg: %s\n", k.c_str()); return false; }
    }
    if (a.poll_hz <= 0) { std::fprintf(stderr, "--poll-hz must be > 0\n"); return false; }
    return true;
}

void mkdir_parent(const std::string& path) {
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos || pos == 0) return;
    std::string dir = path.substr(0, pos);
    ::mkdir(dir.c_str(), 0755);  // best-effort; ignore EEXIST
}

int create_listener(const std::string& uds_path) {
    ::unlink(uds_path.c_str());
    mkdir_parent(uds_path);
    int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) { std::perror("socket"); return -1; }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (uds_path.size() >= sizeof(addr.sun_path)) {
        std::fprintf(stderr, "uds path too long\n"); ::close(fd); return -1;
    }
    std::strncpy(addr.sun_path, uds_path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::perror("bind"); ::close(fd); return -1;
    }
    if (::listen(fd, 4) < 0) { std::perror("listen"); ::close(fd); return -1; }
    return fd;
}

void to_sim_frame(const MotorControl::Control& ctrl, uint64_t bridge_seq,
                  g0_sjt::SimFrame& out) {
    out.header.magic = g0_sjt::kSimFrameMagic;
    out.header.version = g0_sjt::kSimFrameVersion;
    out.header.motor_count = g0_sjt::kNumMotors;
    out.header.timestamp_ns = ctrl.timestamp_ns();
    out.header.sequence_id = bridge_seq;
    const auto& motors = ctrl.motors();
    for (int i = 0; i < g0_sjt::kNumMotors; ++i) {
        out.motors[i].pos = motors[i].pos();
        out.motors[i].dq  = motors[i].dq();
        out.motors[i].kp  = motors[i].kp();
        out.motors[i].kd  = motors[i].kd();
        out.motors[i].tau = motors[i].tau();
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Line-buffer stdout so log lines flush promptly even when redirected.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    BridgeArgs a;
    if (!parse(argc, argv, a)) { usage(argv[0]); return 2; }
    if (a.topic.rfind("mc/", 0) == 0) {
        std::fprintf(stderr,
            "ERROR: refusing real-hardware topic '%s'. Must start with g0_sim/.\n",
            a.topic.c_str());
        return 4;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    std::printf("[bridge] DDS topic : %s\n", a.topic.c_str());
    std::printf("[bridge] UDS path  : %s\n", a.uds_path.c_str());
    std::printf("[bridge] poll_hz   : %g\n", a.poll_hz);

    int listen_fd = create_listener(a.uds_path);
    if (listen_fd < 0) return 5;

    g0_sjt::MbusSimSubscriber sub(a.topic);
    std::printf("[bridge] subscribed; waiting for frames...  (Ctrl-C to stop)\n");

    std::vector<int> clients;
    const auto period = std::chrono::duration<double>(1.0 / a.poll_hz);
    auto next_tick = std::chrono::steady_clock::now();
    uint64_t bridge_seq = 0;
    uint64_t frames_in = 0;
    uint64_t frames_out = 0;

    MotorControl::Control ctrl;
    g0_sjt::SimFrame frame{};

    while (!g_stop.load()) {
        // Accept new clients (non-blocking).
        for (;;) {
            int cfd = ::accept4(listen_fd, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
            if (cfd < 0) break;
            std::printf("[bridge] client connected (fd=%d)\n", cfd);
            clients.push_back(cfd);
        }

        // Drain DDS samples.
        while (sub.poll(ctrl)) {
            ++frames_in;
            to_sim_frame(ctrl, ++bridge_seq, frame);

            // Broadcast.
            for (auto it = clients.begin(); it != clients.end();) {
                ssize_t n = ::send(*it, &frame, sizeof(frame), MSG_NOSIGNAL);
                if (n != static_cast<ssize_t>(sizeof(frame))) {
                    std::printf("[bridge] client fd=%d dropped (send=%zd errno=%d)\n",
                                *it, n, errno);
                    ::close(*it);
                    it = clients.erase(it);
                } else {
                    ++frames_out;
                    ++it;
                }
            }
            if (frames_in % 200 == 1) {
                std::printf("[bridge] in=%lu out=%lu clients=%zu (last seq=%u)\n",
                            (unsigned long)frames_in, (unsigned long)frames_out,
                            clients.size(), (unsigned)ctrl.sequence_id());
            }
        }

        next_tick += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
        std::this_thread::sleep_until(next_tick);
    }

    std::printf("\n[bridge] stopping. frames_in=%lu frames_out=%lu\n",
                (unsigned long)frames_in, (unsigned long)frames_out);
    for (int fd : clients) ::close(fd);
    ::close(listen_fd);
    ::unlink(a.uds_path.c_str());
    return 0;
}
