cmake_minimum_required(VERSION 3.10)
project(MotionPlanner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(src)

find_package(tinyxml2 REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(motion_planner src/main.cpp)
target_link_libraries(motion_planner tinyxml2::tinyxml2 Eigen3::Eigen)

enable_testing()
add_executable(test_motion_planner tests/test.cpp)
target_link_libraries(test_motion_planner tinyxml2::tinyxml2 Eigen3::Eigen)
add_test(NAME test_motion_planner COMMAND test_motion_planner)