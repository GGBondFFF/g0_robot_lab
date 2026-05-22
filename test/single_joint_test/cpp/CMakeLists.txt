cmake_minimum_required(VERSION 3.16)
project(g0_single_joint_test CXX)

# mbus C++ binding (and the CycloneDDS-CXX it depends on) requires C++17.
# motion-planner uses the same; keep aligned.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# mbus provides:
#   /usr/local/include/idl_motor_control.hpp   (IDL-generated POD)
#   mbus::mbus_idl_cxx                         (target for IDL link)
#   mbus::mbus_cxx                             (DDS publisher API, used in Phase 3+)
find_package(mbus REQUIRED)

# mbus and CycloneDDS-CXX install into /usr/local/lib but the dynamic loader
# cache on this host doesn't always include it. Use the old DT_RPATH tag
# (instead of DT_RUNPATH) so /usr/local/lib is searched for transitive deps
# like libddscxx.so.11 pulled in via mbus. RUNPATH applies only to direct
# deps, which breaks here.
function(g0_sjt_apply_rpath tgt)
    target_link_options(${tgt} PRIVATE
        "-Wl,--disable-new-dtags"
        "-Wl,-rpath,/usr/local/lib")
endfunction()

# ─── sender (publisher + dry-run) ──────────────────────────────────────────
add_executable(single_joint_sender src/single_joint_sender.cpp)
target_include_directories(single_joint_sender PRIVATE include)
target_link_libraries(single_joint_sender PRIVATE
    mbus::mbus_cxx mbus::mbus_idl_cxx)
g0_sjt_apply_rpath(single_joint_sender)

# ─── DDS->UDS bridge for the Python receiver ───────────────────────────────
add_executable(dds_to_uds_bridge src/dds_to_uds_bridge.cpp)
target_include_directories(dds_to_uds_bridge PRIVATE include)
target_link_libraries(dds_to_uds_bridge PRIVATE
    mbus::mbus_cxx mbus::mbus_idl_cxx)
g0_sjt_apply_rpath(dds_to_uds_bridge)
