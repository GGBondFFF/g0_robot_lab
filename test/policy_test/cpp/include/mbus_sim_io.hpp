// Minimal mbus DDS pub/sub wrappers for the policy_test sim bridges.
//
// Mirrors motion-planner/src/dds_client.cpp behaviour but talks only to the
// *virtual* g0_sim/* topic family. All wrappers refuse any topic name that
// starts with "mc/" (real-hardware prefix) so a mis-set CLI flag can never
// reach the real robot.
#ifndef G0_POLICY_TEST_MBUS_SIM_IO_HPP
#define G0_POLICY_TEST_MBUS_SIM_IO_HPP

#include <mbus_wrapper.hpp>
#include <idl_motor_control.hpp>
#include <idl_robot_control.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace g0_pt {

// Global opt-in to allow mc/ (real-hardware) topics. Default: false. Set
// to true via set_allow_real_hw_topics() in main() AFTER an explicit CLI
// confirmation flag — never enable silently.
inline bool& _allow_real_hw_flag() { static bool f = false; return f; }
inline void set_allow_real_hw_topics(bool allow) { _allow_real_hw_flag() = allow; }

inline void guard_sim_topic(const std::string& topic_name) {
    if (topic_name.rfind("mc/", 0) == 0 && !_allow_real_hw_flag()) {
        throw std::runtime_error(
            "refusing real-hardware topic prefix mc/ (pass --allow-real-hw "
            "to override): " + topic_name);
    }
}

// ─── MotorControl::Control ────────────────────────────────────────────────
class MotorControlPublisher {
public:
    explicit MotorControlPublisher(const std::string& topic_name) {
        guard_sim_topic(topic_name);
        const mbus::TopicConfig cfg{topic_name, "mbus::MotorControl_Control",
                                    mbus::TopicRule::Pub};
        ctx_ = std::make_unique<mbus::MbusContext>(0);
        topic_ = ctx_->register_topic<MotorControl::Control>(cfg);
    }
    void publish(const MotorControl::Control& msg) { topic_->write(msg); }

private:
    std::unique_ptr<mbus::MbusContext> ctx_;
    std::shared_ptr<mbus::MbusTopic<MotorControl::Control>> topic_;
};

class MotorControlSubscriber {
public:
    explicit MotorControlSubscriber(const std::string& topic_name) {
        guard_sim_topic(topic_name);
        const mbus::TopicConfig cfg{topic_name, "mbus::MotorControl_Control",
                                    mbus::TopicRule::Sub};
        ctx_ = std::make_unique<mbus::MbusContext>(0);
        topic_ = ctx_->register_topic<MotorControl::Control>(cfg);
    }
    bool poll(MotorControl::Control& out) {
        std::vector<mbus::DdsData<MotorControl::Control>> samples;
        auto rc = topic_->take(samples, 1U);
        if (rc != DDS_RETCODE_OK) {
            throw std::runtime_error(
                "mbus take(MotorControl::Control) failed (rc=" +
                std::to_string(rc) + ")");
        }
        if (samples.empty()) return false;
        out = samples.back().data;
        return true;
    }

private:
    std::unique_ptr<mbus::MbusContext> ctx_;
    std::shared_ptr<mbus::MbusTopic<MotorControl::Control>> topic_;
};

// ─── MotorControl::State ──────────────────────────────────────────────────
class MotorStatePublisher {
public:
    explicit MotorStatePublisher(const std::string& topic_name) {
        guard_sim_topic(topic_name);
        const mbus::TopicConfig cfg{topic_name, "mbus::MotorControl_State",
                                    mbus::TopicRule::Pub};
        ctx_ = std::make_unique<mbus::MbusContext>(0);
        topic_ = ctx_->register_topic<MotorControl::State>(cfg);
    }
    void publish(const MotorControl::State& msg) { topic_->write(msg); }

private:
    std::unique_ptr<mbus::MbusContext> ctx_;
    std::shared_ptr<mbus::MbusTopic<MotorControl::State>> topic_;
};

class MotorStateSubscriber {
public:
    explicit MotorStateSubscriber(const std::string& topic_name) {
        guard_sim_topic(topic_name);
        const mbus::TopicConfig cfg{topic_name, "mbus::MotorControl_State",
                                    mbus::TopicRule::Sub};
        ctx_ = std::make_unique<mbus::MbusContext>(0);
        topic_ = ctx_->register_topic<MotorControl::State>(cfg);
    }
    bool poll(MotorControl::State& out) {
        std::vector<mbus::DdsData<MotorControl::State>> samples;
        auto rc = topic_->take(samples, 1U);
        if (rc != DDS_RETCODE_OK) {
            throw std::runtime_error(
                "mbus take(MotorControl::State) failed (rc=" +
                std::to_string(rc) + ")");
        }
        if (samples.empty()) return false;
        out = samples.back().data;
        return true;
    }

private:
    std::unique_ptr<mbus::MbusContext> ctx_;
    std::shared_ptr<mbus::MbusTopic<MotorControl::State>> topic_;
};

// ─── RobotControl::Control ────────────────────────────────────────────────
class RobotControlPublisher {
public:
    explicit RobotControlPublisher(const std::string& topic_name) {
        guard_sim_topic(topic_name);
        const mbus::TopicConfig cfg{topic_name, "mbus::RobotControl_Control",
                                    mbus::TopicRule::Pub};
        ctx_ = std::make_unique<mbus::MbusContext>(0);
        topic_ = ctx_->register_topic<RobotControl::Control>(cfg);
    }
    void publish(const RobotControl::Control& msg) { topic_->write(msg); }

private:
    std::unique_ptr<mbus::MbusContext> ctx_;
    std::shared_ptr<mbus::MbusTopic<RobotControl::Control>> topic_;
};

class RobotControlSubscriber {
public:
    explicit RobotControlSubscriber(const std::string& topic_name) {
        guard_sim_topic(topic_name);
        const mbus::TopicConfig cfg{topic_name, "mbus::RobotControl_Control",
                                    mbus::TopicRule::Sub};
        ctx_ = std::make_unique<mbus::MbusContext>(0);
        topic_ = ctx_->register_topic<RobotControl::Control>(cfg);
    }
    bool poll(RobotControl::Control& out) {
        std::vector<mbus::DdsData<RobotControl::Control>> samples;
        auto rc = topic_->take(samples, 1U);
        if (rc != DDS_RETCODE_OK) {
            throw std::runtime_error(
                "mbus take(RobotControl::Control) failed (rc=" +
                std::to_string(rc) + ")");
        }
        if (samples.empty()) return false;
        out = samples.back().data;
        return true;
    }

private:
    std::unique_ptr<mbus::MbusContext> ctx_;
    std::shared_ptr<mbus::MbusTopic<RobotControl::Control>> topic_;
};

}  // namespace g0_pt

#endif  // G0_POLICY_TEST_MBUS_SIM_IO_HPP
