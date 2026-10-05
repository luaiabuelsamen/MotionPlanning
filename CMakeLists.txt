cmake_minimum_required(VERSION 3.10)
project(MotionPlanner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(src)

find_package(PkgConfig REQUIRED)
pkg_check_modules(TINYXML2 REQUIRED tinyxml2)
find_package(Eigen3 REQUIRED)

link_directories(${TINYXML2_LIBRARY_DIRS})
include_directories(${TINYXML2_INCLUDE_DIRS})

# Test executable (needs only Eigen and tinyxml2)
enable_testing()
add_executable(test_motion_planner tests/test.cpp)
target_link_libraries(test_motion_planner ${TINYXML2_LIBRARIES} Eigen3::Eigen)
target_compile_definitions(test_motion_planner PRIVATE
    MOTION_PLANNER_ASSETS="${CMAKE_CURRENT_SOURCE_DIR}/src/assets")
add_test(NAME test_motion_planner COMMAND test_motion_planner)

# gRPC server: optional, built when Protobuf, gRPC and the gRPC C++ plugin
# are found
find_package(Protobuf QUIET)
pkg_check_modules(GRPCPP QUIET grpc++)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
if(Protobuf_FOUND AND GRPCPP_FOUND AND GRPC_CPP_PLUGIN)
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
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --grpc_out "${GENERATED_PROTOBUF_PATH}"
             --cpp_out "${GENERATED_PROTOBUF_PATH}"
             -I "${PROTO_PATH}"
             --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
             "${MOTION_PLANNER_PROTO}"
        DEPENDS "${MOTION_PLANNER_PROTO}"
    )

    add_executable(motion_planner_server
        src/motion_planner_server.cpp
        ${MOTION_PLANNER_PB_CPP_FILE}
        ${MOTION_PLANNER_GRPC_PB_CPP_FILE}
    )
    target_include_directories(motion_planner_server PRIVATE
        ${GENERATED_PROTOBUF_PATH} ${GRPCPP_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS})
    target_link_libraries(motion_planner_server
        ${TINYXML2_LIBRARIES}
        Eigen3::Eigen
        ${GRPCPP_LIBRARIES}
        ${Protobuf_LIBRARIES}
    )

    # Python gRPC test
    add_test(NAME python_grpc_test
        COMMAND ${CMAKE_COMMAND} -E env bash ${CMAKE_CURRENT_SOURCE_DIR}/generate_python_proto.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(python_grpc_test PROPERTIES
        DEPENDS test_motion_planner
        ENVIRONMENT "PATH=${CMAKE_CURRENT_BINARY_DIR}:$ENV{PATH}"
    )
else()
    message(STATUS "Protobuf / gRPC not found: not building motion_planner_server")
endif()
