// robot.hpp
#pragma once
#include "transform.hpp"
#include "fk_solver.hpp"
#include "jacobian.hpp"
#include "ik_solver.hpp"
#include "urdf_parser.hpp"
#include "movej_optimizer.hpp"
#include "movel_optimizer.hpp"
#include "trajectory_utils.hpp"
#include <vector>
#include <iostream>
#include <memory>

class Robot {
private:
    std::vector<Joint> joints;
    FKSolver fk_solver;
    JacobianAnalytical jacobianSolver;
    IKSolver ik_solver;
    
    // Trajectory optimizers
    std::unique_ptr<MoveJOptimizer> movej_optimizer;
    std::unique_ptr<MoveLOptimizer> movel_optimizer;
    
    // Default configuration
    OptimizerConfig default_config;
    
public:

    Robot(const std::string& urdf_file) 
        : joints(URDFParser::parseJoints(urdf_file))
        , fk_solver(joints),
        jacobianSolver(fk_solver, joints) 
        , ik_solver(jacobianSolver, fk_solver, getNumDOFs(joints)) {
        
        size_t dof = getNumDOFs(joints);
        
        // Initialize optimizers with default configuration
        movej_optimizer = std::make_unique<MoveJOptimizer>(fk_solver, dof, default_config);
        movel_optimizer = std::make_unique<MoveLOptimizer>(fk_solver, ik_solver, 
                                                           jacobianSolver, dof, default_config);
        
        // Set joint limits from URDF
        default_config.joint_lower_limits.resize(dof);
        default_config.joint_upper_limits.resize(dof);
        
        size_t joint_idx = 0;
        for (const auto& joint : joints) {
            if (joint.type == "revolute") {
                default_config.joint_lower_limits[joint_idx] = joint.lower_limit;
                default_config.joint_upper_limits[joint_idx] = joint.upper_limit;
                joint_idx++;
            }
        }
    }

    Eigen::MatrixXd getJacobian(const std::vector<double>& joint_angles) {
        return jacobianSolver.compute(joint_angles);
    }

    std::pair<Vector3, Quaternion> getEndEffectorPose(const std::vector<double>& joint_angles) {
        Transform T = fk_solver.computeFK(joint_angles);
        return {T.getPosition(), T.getQuaternion()};
    }

    std::vector<double> computeIK(const Vector3& target_position, 
                                   const Quaternion& target_orientation,
                                   const std::vector<double>& initial_guess = {}) {
        return ik_solver.computeIK(target_position, target_orientation, initial_guess);
    }
    
    /**
     * MoveJ - Generate smooth joint space trajectory
     * @param start_config Starting joint configuration
     * @param goal_config Goal joint configuration
     * @param num_waypoints Number of waypoints in trajectory
     * @return Optimized trajectory in joint space
     */
    Trajectory moveJ(const std::vector<double>& start_config,
                     const std::vector<double>& goal_config,
                     size_t num_waypoints = 50) {
        return movej_optimizer->optimize(start_config, goal_config, num_waypoints);
    }
    
    /**
     * MoveL - Generate linear Cartesian trajectory
     * @param start_config Starting joint configuration
     * @param goal_config Goal joint configuration (defines goal pose)
     * @param num_waypoints Number of waypoints in trajectory
     * @return Optimized trajectory following linear Cartesian path
     */
    Trajectory moveL(const std::vector<double>& start_config,
                     const std::vector<double>& goal_config,
                     size_t num_waypoints = 50) {
        return movel_optimizer->optimize(start_config, goal_config, num_waypoints);
    }
    
    /**
     * MoveL with explicit Cartesian goal
     * @param start_config Starting joint configuration
     * @param goal_pos Goal position in Cartesian space
     * @param goal_quat Goal orientation as quaternion
     * @param num_waypoints Number of waypoints in trajectory
     * @return Optimized trajectory following linear Cartesian path
     */
    Trajectory moveL(const std::vector<double>& start_config,
                     const Vector3& goal_pos,
                     const Quaternion& goal_quat,
                     size_t num_waypoints = 50) {
        return movel_optimizer->optimizeWithPath(start_config, goal_pos, goal_quat, num_waypoints);
    }
    
    /**
     * Update optimizer configuration
     */
    void setOptimizerConfig(const OptimizerConfig& config) {
        default_config = config;
        size_t dof = getNumDOFs(joints);
        movej_optimizer = std::make_unique<MoveJOptimizer>(fk_solver, dof, default_config);
        movel_optimizer = std::make_unique<MoveLOptimizer>(fk_solver, ik_solver, 
                                                           jacobianSolver, dof, default_config);
    }
    
    /**
     * Get current optimizer configuration
     */
    const OptimizerConfig& getOptimizerConfig() const {
        return default_config;
    }
    
    /**
     * Get number of degrees of freedom
     */
    size_t getDOF() const {
        return getNumDOFs(joints);
    }

private:
    size_t getNumDOFs(const std::vector<Joint>& joints) const {
        size_t dof = 0;
        for (const auto& joint : joints) {
            if (joint.type == "revolute") dof++;
        }
        return dof;
    }
};