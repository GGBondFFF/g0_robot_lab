// Minimal mbus DDS wrapper for the virtual single-joint test.
//
// Mirrors motion-planner/src/dds_client.cpp but talks to the *virtual* topic
// only. Refuses any topic name that starts with "mc/" — that prefix belongs
// to real hardware.
#ifndef G0_SINGLE_JOINT_TEST_MBUS_SIM_CLIENT_HPP
#define G0_SINGLE_JOINT_TEST_MBUS_SIM_CLIENT_HPP

#include <mbus_wrapper.hpp>
#include <idl_motor_control.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace g0_sjt {

// Opt-in flag to allow mc/ (real-hardware) topics in the publisher /
// subscriber wrappers below. The single_joint_sender CLI sets this via
// --allow-real-hw after printing a loud warning.
inline bool& _allow_real_hw_flag() { static bool f = false; return f; }
inline void set_allow_real_hw_topics(bool allow) { _allow_real_hw_flag() = allow; }

class MbusSimPublisher {
public:
    explicit MbusSimPublisher(const std::string& topic_name) {
        if (topic_name.rfind("mc/", 0) == 0 && !_allow_real_hw_flag()) {
            throw std::runtime_error("refusing real-hardware topic prefix mc/ "
                                     "(pass --allow-real-hw to override): " + topic_name);
        }
        const mbus::TopicConfig cfg{topic_name, "mbus::MotorControl_Control",
                                    mbus::TopicRule::Pub};
        ctx_ = std::make_unique<mbus::MbusContext>(0);
        topic_ = ctx_->register_topic<MotorControl::Control>(cfg);
    }

    void publish(const MotorControl::Control& msg) {
        topic_->write(msg);
    }

private:
    std::unique_ptr<mbus::MbusContext> ctx_;
    std::shared_ptr<mbus::MbusTopic<MotorControl::Control>> topic_;
};

class MbusSimSubscriber {
public:
    explicit MbusSimSubscriber(const std::string& topic_name) {
        if (topic_name.rfind("mc/", 0) == 0 && !_allow_real_hw_flag()) {
            throw std::runtime_error("refusing real-hardware topic prefix mc/ "
                                     "(pass --allow-real-hw to override): " + topic_name);
        }
        const mbus::TopicConfig cfg{topic_name, "mbus::MotorControl_Control",
                                    mbus::TopicRule::Sub};
        ctx_ = std::make_unique<mbus::MbusContext>(0);
        topic_ = ctx_->register_topic<MotorControl::Control>(cfg);
    }

    // Non-blocking take: returns true and fills `out` if a fresh sample exists.
    bool poll(MotorControl::Control& out) {
        std::vector<mbus::DdsData<MotorControl::Control>> samples;
        auto rc = topic_->take(samples, 1U);
        if (rc != DDS_RETCODE_OK) {
            throw std::runtime_error("mbus take failed (rc=" + std::to_string(rc) + ")");
        }
        if (samples.empty()) return false;
        out = samples.back().data;
        return true;
    }

private:
    std::unique_ptr<mbus::MbusContext> ctx_;
    std::shared_ptr<mbus::MbusTopic<MotorControl::Control>> topic_;
};

}  // namespace g0_sjt

#endif  // G0_SINGLE_JOINT_TEST_MBUS_SIM_CLIENT_HPP
