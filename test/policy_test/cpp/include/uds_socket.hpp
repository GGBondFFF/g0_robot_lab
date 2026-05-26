// Small Unix-domain socket helpers shared by both policy_test bridges.
//
// Each helper class owns one AF_UNIX SOCK_STREAM listener on a fixed path.
// Two flavors:
//   UdsBroadcaster — server side. Accepts multiple non-blocking clients,
//                    broadcasts one fixed-size frame to every connected fd.
//                    Drops clients that fail to drain.
//   UdsReceiver    — server side. Accepts a single client (latest one wins);
//                    reads fixed-size frames non-blocking and hands them to a
//                    callback.
//
// Both are intentionally header-only and synchronous — the bridges drive them
// from one poll loop. Designed to match the cadence of single_joint_test's
// dds_to_uds_bridge so the same Python-side decoder code can be reused.
#ifndef G0_POLICY_TEST_UDS_SOCKET_HPP
#define G0_POLICY_TEST_UDS_SOCKET_HPP

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace g0_pt {

inline void mkdir_parent(const std::string& path) {
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos || pos == 0) return;
    ::mkdir(path.substr(0, pos).c_str(), 0755);  // best-effort; ignore EEXIST
}

inline int make_uds_listener(const std::string& path) {
    ::unlink(path.c_str());
    mkdir_parent(path);
    int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd < 0) throw std::runtime_error(std::string("socket: ") + std::strerror(errno));
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        ::close(fd);
        throw std::runtime_error("uds path too long: " + path);
    }
    std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        throw std::runtime_error(std::string("bind ") + path + ": " + std::strerror(errno));
    }
    if (::listen(fd, 4) < 0) {
        ::close(fd);
        throw std::runtime_error(std::string("listen: ") + std::strerror(errno));
    }
    return fd;
}

// ─── Broadcaster: bridge -> N clients ─────────────────────────────────────
class UdsBroadcaster {
public:
    UdsBroadcaster(std::string path, const char* tag)
        : path_(std::move(path)), tag_(tag) {
        listen_fd_ = make_uds_listener(path_);
    }
    ~UdsBroadcaster() {
        for (int fd : clients_) ::close(fd);
        if (listen_fd_ >= 0) ::close(listen_fd_);
        ::unlink(path_.c_str());
    }
    UdsBroadcaster(const UdsBroadcaster&) = delete;
    UdsBroadcaster& operator=(const UdsBroadcaster&) = delete;

    void accept_pending() {
        for (;;) {
            int cfd = ::accept4(listen_fd_, nullptr, nullptr,
                                SOCK_NONBLOCK | SOCK_CLOEXEC);
            if (cfd < 0) break;
            std::printf("[%s] client connected (fd=%d) on %s\n",
                        tag_, cfd, path_.c_str());
            clients_.push_back(cfd);
        }
    }

    // Sends `size` bytes; partial writes drop the client.
    void broadcast(const void* data, size_t size) {
        for (auto it = clients_.begin(); it != clients_.end();) {
            ssize_t n = ::send(*it, data, size, MSG_NOSIGNAL);
            if (n != static_cast<ssize_t>(size)) {
                std::printf("[%s] client fd=%d dropped (send=%zd errno=%d)\n",
                            tag_, *it, n, errno);
                ::close(*it);
                it = clients_.erase(it);
            } else {
                ++it;
            }
        }
    }

    size_t client_count() const { return clients_.size(); }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    const char* tag_;
    int listen_fd_ = -1;
    std::vector<int> clients_;
};

// ─── Receiver: bridge <- 1 client (latest wins) ───────────────────────────
class UdsReceiver {
public:
    UdsReceiver(std::string path, const char* tag, size_t frame_size)
        : path_(std::move(path)), tag_(tag), frame_size_(frame_size),
          buffer_(frame_size) {
        listen_fd_ = make_uds_listener(path_);
    }
    ~UdsReceiver() {
        if (client_fd_ >= 0) ::close(client_fd_);
        if (listen_fd_ >= 0) ::close(listen_fd_);
        ::unlink(path_.c_str());
    }
    UdsReceiver(const UdsReceiver&) = delete;
    UdsReceiver& operator=(const UdsReceiver&) = delete;

    void accept_pending() {
        for (;;) {
            int cfd = ::accept4(listen_fd_, nullptr, nullptr,
                                SOCK_NONBLOCK | SOCK_CLOEXEC);
            if (cfd < 0) break;
            if (client_fd_ >= 0) {
                std::printf("[%s] replacing client fd=%d -> %d on %s\n",
                            tag_, client_fd_, cfd, path_.c_str());
                ::close(client_fd_);
            } else {
                std::printf("[%s] client connected (fd=%d) on %s\n",
                            tag_, cfd, path_.c_str());
            }
            client_fd_ = cfd;
            buffer_filled_ = 0;
        }
    }

    // Reads as many whole frames as are available. For each, calls fn(ptr).
    // Returns the number of frames delivered this call.
    template <typename Fn>
    int drain(Fn&& fn) {
        if (client_fd_ < 0) return 0;
        int delivered = 0;
        for (;;) {
            ssize_t want = static_cast<ssize_t>(frame_size_ - buffer_filled_);
            ssize_t n = ::recv(client_fd_, buffer_.data() + buffer_filled_,
                               static_cast<size_t>(want), 0);
            if (n > 0) {
                buffer_filled_ += static_cast<size_t>(n);
                if (buffer_filled_ == frame_size_) {
                    fn(buffer_.data());
                    ++delivered;
                    buffer_filled_ = 0;
                }
                continue;
            }
            if (n == 0) {
                std::printf("[%s] client fd=%d closed on %s\n",
                            tag_, client_fd_, path_.c_str());
                ::close(client_fd_);
                client_fd_ = -1;
                buffer_filled_ = 0;
                break;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            std::printf("[%s] recv errno=%d on fd=%d; dropping\n",
                        tag_, errno, client_fd_);
            ::close(client_fd_);
            client_fd_ = -1;
            buffer_filled_ = 0;
            break;
        }
        return delivered;
    }

    bool connected() const { return client_fd_ >= 0; }
    const std::string& path() const { return path_; }

private:
    std::string path_;
    const char* tag_;
    size_t frame_size_;
    std::vector<uint8_t> buffer_;
    size_t buffer_filled_ = 0;
    int listen_fd_ = -1;
    int client_fd_ = -1;
};

}  // namespace g0_pt

#endif  // G0_POLICY_TEST_UDS_SOCKET_HPP
