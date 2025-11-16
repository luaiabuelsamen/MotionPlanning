cmake_minimum_required(VERSION 3.10)
project(MotionPlanner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(src)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TINYXML2 REQUIRED tinyxml2)
find_package(Eigen3 REQUIRED)

# Find Protobuf and gRPC using pkg-config
find_package(Protobuf REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)
pkg_check_modules(GRPCPP REQUIRED grpc++)

link_directories(${TINYXML2_LIBRARY_DIRS})
link_directories(${GRPC_LIBRARY_DIRS})
link_directories(${GRPCPP_LIBRARY_DIRS})

# Generate protobuf and gRPC files
set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(GENERATED_PROTOBUF_PATH "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${GENERATED_PROTOBUF_PATH})

set(MOTION_PLANNER_PROTO "${PROTO_PATH}/motion_planner.proto")
set(MOTION_PLANNER_PB_CPP_FILE "${GENERATED_PROTOBUF_PATH}/motion_planner.pb.cc")
set(MOTION_PLANNER_PB_H_FILE "${GENERATED_PROTOBUF_PATH}/motion_planner.pb.h")
set(MOTION_PLANNER_GRPC_PB_CPP_FILE "${GENERATED_PROTOBUF_PATH}/motion_planner.grpc.pb.cc")
set(MOTION_PLANNER_GRPC_PB_H_FILE "${GENERATED_PROTOBUF_PATH}/motion_planner.grpc.pb.h")

add_custom_command(
    OUTPUT "${MOTION_PLANNER_PB_CPP_FILE}"
           "${MOTION_PLANNER_PB_H_FILE}"
           "${MOTION_PLANNER_GRPC_PB_CPP_FILE}"
           "${MOTION_PLANNER_GRPC_PB_H_FILE}"
    COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
    ARGS --grpc_out "${GENERATED_PROTOBUF_PATH}"
         --cpp_out "${GENERATED_PROTOBUF_PATH}"
         -I "${PROTO_PATH}"
         --plugin=protoc-gen-grpc=/opt/homebrew/bin/grpc_cpp_plugin
         "${MOTION_PLANNER_PROTO}"
    DEPENDS "${MOTION_PLANNER_PROTO}"
)

set(GENERATED_PROTOBUF_FILES
    ${MOTION_PLANNER_PB_CPP_FILE}
    ${MOTION_PLANNER_PB_H_FILE}
    ${MOTION_PLANNER_GRPC_PB_CPP_FILE}
    ${MOTION_PLANNER_GRPC_PB_H_FILE}
)

include_directories(${GENERATED_PROTOBUF_PATH})
include_directories(${TINYXML2_INCLUDE_DIRS})
include_directories(${GRPC_INCLUDE_DIRS})
include_directories(${PROTOBUF_INCLUDE_DIRS})

# gRPC server executable
add_executable(motion_planner_server 
    src/motion_planner_server.cpp
    ${GENERATED_PROTOBUF_FILES}
)
target_link_libraries(motion_planner_server
    ${TINYXML2_LIBRARIES}
    Eigen3::Eigen
    ${GRPC_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
)

# Test executable
enable_testing()
add_executable(test_motion_planner tests/test.cpp)
target_link_libraries(test_motion_planner ${TINYXML2_LIBRARIES} Eigen3::Eigen)
add_test(NAME test_motion_planner COMMAND test_motion_planner)

# Python gRPC test
add_test(NAME python_grpc_test
    COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/generate_python_proto.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
set_tests_properties(python_grpc_test PROPERTIES
    DEPENDS test_motion_planner
    ENVIRONMENT "PATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PATH}"
)