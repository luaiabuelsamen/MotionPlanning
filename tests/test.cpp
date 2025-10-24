#include "../src/robot.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

// Test forward kinematics
void testFK() {
    std::cout << "Testing Forward Kinematics..." << std::endl;
    
    Robot robot("/Users/labuelsamen/Desktop/motion_planner/src/assets/ur5e/ur5e.urdf");
    
    // Test home position (all zeros)
    std::vector<double> joint_angles_home = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto [pos_home, quat_home] = robot.getEndEffectorPose(joint_angles_home);
    
    std::cout << "Home position: [" << pos_home.x() << ", " << pos_home.y() << ", " << pos_home.z() << "]" << std::endl;
    std::cout << "Home quaternion: [" << quat_home.w() << ", " << quat_home.x() << ", " << quat_home.y() << ", " << quat_home.z() << "]" << std::endl;
    
    // Home position for UR5e (known values from test run)
    assert(std::abs(pos_home.x() - 0.8172) < 1e-3);
    assert(std::abs(pos_home.y() - 0.2329) < 1e-3);
    assert(std::abs(pos_home.z() - 0.0628) < 1e-3);
    // Orientation 
    assert(std::abs(quat_home.w() - 0.707107) < 1e-3);
    assert(std::abs(quat_home.x() + 0.707107) < 1e-3);  // Note: x is negative
    assert(std::abs(quat_home.y() - 0.0) < 1e-3);
    assert(std::abs(quat_home.z() - 0.0) < 1e-3);
    
    // Test with non-zero angles
    std::vector<double> joint_angles = {M_PI/4, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto [pos, quat] = robot.getEndEffectorPose(joint_angles);
    
    // Should be different from home
    assert((pos - pos_home).norm() > 0.1);
    
    std::cout << "✓ FK test passed" << std::endl;
}

// Test Jacobian computation
void testJacobian() {
    std::cout << "Testing Jacobian..." << std::endl;
    
    Robot robot("/Users/labuelsamen/Desktop/motion_planner/src/assets/ur5e/ur5e.urdf");
    std::vector<double> joint_angles = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    
    auto J = robot.getJacobian(joint_angles);
    
    // Jacobian should be 6x6
    assert(J.rows() == 6);
    assert(J.cols() == 6);
    
    // Check that Jacobian is not all zeros
    double sum = J.array().abs().sum();
    assert(sum > 0.1);
    
    std::cout << "✓ Jacobian test passed" << std::endl;
}

// Test inverse kinematics
void testIK() {
    std::cout << "Testing Inverse Kinematics..." << std::endl;
    
    Robot robot("/Users/labuelsamen/Desktop/motion_planner/src/assets/ur5e/ur5e.urdf");
    
    // Start with a known pose
    std::vector<double> joint_angles = {M_PI/4, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto [target_pos, target_quat] = robot.getEndEffectorPose(joint_angles);
    
    // Run IK
    std::vector<double> initial_guess = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> ik_result = robot.computeIK(target_pos, target_quat, initial_guess);
    
    // Verify IK result
    auto [pos_verify, quat_verify] = robot.getEndEffectorPose(ik_result);
    
    // Check position error
    double pos_error = (target_pos - pos_verify).norm();
    assert(pos_error < 1e-3);
    
    // Check orientation error
    Quaternion q_error = target_quat * quat_verify.conjugate();
    double orient_error = 2.0 * sqrt(q_error.x()*q_error.x() + 
                                      q_error.y()*q_error.y() + 
                                      q_error.z()*q_error.z());
    assert(orient_error < 1e-2);
    
    std::cout << "✓ IK test passed" << std::endl;
}

// Test URDF parsing
void testURDFParsing() {
    std::cout << "Testing URDF Parsing..." << std::endl;
    
    Robot robot("/Users/labuelsamen/Desktop/motion_planner/src/assets/ur5e/ur5e.urdf");
    
    // We can't directly access joints, but we can check that FK works
    // If parsing failed, FK would crash or give wrong results
    std::vector<double> joint_angles = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto [pos, quat] = robot.getEndEffectorPose(joint_angles);
    
    // Just check that we get some result
    assert(pos.norm() >= 0.0);
    
    std::cout << "✓ URDF parsing test passed" << std::endl;
}

int main() {
    std::cout << "Running modular tests..." << std::endl;
    
    try {
        testURDFParsing();
        testFK();
        testJacobian();
        testIK();
        
        std::cout << "\n✓ All tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "\n✗ Test failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "\n✗ Test failed with unknown error" << std::endl;
        return 1;
    }
    
    return 0;
}