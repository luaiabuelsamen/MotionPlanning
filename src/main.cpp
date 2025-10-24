#include "robot.hpp"
#include <iostream>
#include <iomanip>

int main() {
    Robot robot("/Users/labuelsamen/Desktop/motion_planner/src/assets/ur5e/ur5e.urdf");

    // Example joint angles
    std::vector<double> joint_angles = {M_PI/4, 0.0, 0.0, 0.0, 0.0, 0.0};

    auto [pos, quat] = robot.getEndEffectorPose(joint_angles);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "End-effector pose:\n";
    std::cout << "Position: [" << pos.x() << ", " << pos.y() << ", " << pos.z() << "]\n";
    std::cout << "Quaternion: [" << quat.w() << ", " << quat.x() << ", " 
              << quat.y() << ", " << quat.z() << "]\n";

    return 0;
}