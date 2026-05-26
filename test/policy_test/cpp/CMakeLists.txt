cmake_minimum_required(VERSION 3.16)
project(g0_policy_test CXX)

# Match motion-planner and single_joint_test (mbus C++ binding requires 17).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(mbus REQUIRED)

# mbus and CycloneDDS-CXX live in /usr/local/lib but the dynamic loader cache
# doesn't always pick that up. Use DT_RPATH (not DT_RUNPATH) so /usr/local/lib
# is searched for transitive deps like libddscxx.so.11.
function(g0_pt_apply_rpath tgt)
    target_link_options(${tgt} PRIVATE
        "-Wl,--disable-new-dtags"
        "-Wl,-rpath,/usr/local/lib")
endfunction()

# ─── policy_control_bridge (sim2real "brain" side) ────────────────────────
add_executable(policy_control_bridge src/policy_control_bridge.cpp)
target_include_directories(policy_control_bridge PRIVATE include)
target_link_libraries(policy_control_bridge PRIVATE
    mbus::mbus_cxx mbus::mbus_idl_cxx)
g0_pt_apply_rpath(policy_control_bridge)

# ─── plant_state_bridge (Isaac Lab side) ──────────────────────────────────
add_executable(plant_state_bridge src/plant_state_bridge.cpp)
target_include_directories(plant_state_bridge PRIVATE include)
target_link_libraries(plant_state_bridge PRIVATE
    mbus::mbus_cxx mbus::mbus_idl_cxx)
g0_pt_apply_rpath(plant_state_bridge)

# ─── fake_control_agent (TOML -> g0_sim/robot_control) ────────────────────
# tomlplusplus ships as a header-only library inside motion-planner's
# thirdparty install tree. We point straight at that include dir to avoid
# a duplicate vendor checkout. Override with -DG0_PT_TOMLPP_INCLUDE=... at
# configure time if you've installed it elsewhere.
set(G0_PT_TOMLPP_INCLUDE
    "/home/lz/ws/motion-planner/cpp/thirdparty/install/include"
    CACHE PATH "Directory containing toml++/toml.hpp")

if(NOT EXISTS "${G0_PT_TOMLPP_INCLUDE}/toml++/toml.hpp")
    message(FATAL_ERROR
        "tomlplusplus header not found at ${G0_PT_TOMLPP_INCLUDE}/toml++/toml.hpp. "
        "Pass -DG0_PT_TOMLPP_INCLUDE=<path> at configure time.")
endif()

add_executable(fake_control_agent src/fake_control_agent.cpp)
target_include_directories(fake_control_agent PRIVATE
    include "${G0_PT_TOMLPP_INCLUDE}")
target_link_libraries(fake_control_agent PRIVATE
    mbus::mbus_cxx mbus::mbus_idl_cxx)
g0_pt_apply_rpath(fake_control_agent)

# ─── sniff_motor_control (passive DDS subscriber + log writer) ────────────
add_executable(sniff_motor_control src/sniff_motor_control.cpp)
target_include_directories(sniff_motor_control PRIVATE include)
target_link_libraries(sniff_motor_control PRIVATE
    mbus::mbus_cxx mbus::mbus_idl_cxx)
g0_pt_apply_rpath(sniff_motor_control)
