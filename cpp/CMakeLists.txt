cmake_minimum_required(VERSION 3.16)
project(bridge_bot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BRIDGE_PYTHON "Build the pybind11 bridge_cpp module" OFF)
option(BRIDGE_TESTS "Build the C++ unit tests" ON)

find_package(Threads REQUIRED)

# 1. cardbot-framework (header-only C++ library + ismcsolver)
set(CARDBOT_TESTS OFF CACHE BOOL "" FORCE)
set(CARDBOT_PYTHON OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cardbot-framework/cpp cardbot)

# 2. Bo Haglund's Double Dummy Solver (third_party/dds, Apache-2.0)
set(DDS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/dds/library/src")
file(GLOB_RECURSE DDS_SOURCES "${DDS_SRC_DIR}/*.cpp")
add_library(dds STATIC ${DDS_SOURCES})
target_include_directories(dds PUBLIC "${DDS_SRC_DIR}")
target_compile_options(dds PRIVATE -O3 -fPIC -Wno-unknown-pragmas)
target_link_libraries(dds PUBLIC Threads::Threads)

# 3. bridge_lib: rules, sampler, DDS wrapper, encoder, search, agents
add_library(bridge_lib STATIC
    bridge/dds_solver.cpp
    bridge/encoder.cpp
    bridge/search.cpp
    bridge/agents.cpp
)
target_include_directories(bridge_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(bridge_lib PRIVATE -O3 -Wall -Wextra)
target_link_libraries(bridge_lib PUBLIC cardbot dds Threads::Threads)

# 4. bridge_duel CLI
add_executable(bridge_duel tools/duel.cpp)
target_compile_options(bridge_duel PRIVATE -O3 -Wall -Wextra)
target_link_libraries(bridge_duel PRIVATE bridge_lib)

# 5. C++ unit tests
if(BRIDGE_TESTS)
    enable_testing()
    foreach(t test_rules test_sampler test_dds test_search)
        add_executable(bridge_${t} tests/${t}.cpp)
        target_compile_options(bridge_${t} PRIVATE -O3 -Wall -Wextra)
        target_link_libraries(bridge_${t} PRIVATE bridge_lib)
        add_test(NAME ${t} COMMAND bridge_${t})
    endforeach()
endif()

# 6. Python module (bridge_cpp)
if(BRIDGE_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(bridge_cpp python/bridge_module.cpp)
    target_compile_options(bridge_cpp PRIVATE -O3 -Wall -Wextra)
    target_link_libraries(bridge_cpp PRIVATE bridge_lib)
endif()
