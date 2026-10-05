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

    /**
     * Load the kinematic chain from the URDF's root link to `tip_link` (by
     * default the leaf link with the most movable joints, e.g. tool0). Joint
     * angles are one value per revolute / continuous / prismatic joint on
     * that chain, root first. Throws std::runtime_error on an unreadable URDF.
     */
    Robot(const std::string& urdf_file, const std::string& tip_link = "")
        : joints(URDFParser::parseChain(urdf_file, tip_link))
        , fk_solver(joints),
        jacobianSolver(fk_solver, joints) 
        , ik_solver(jacobianSolver, fk_solver, getNumDOFs(joints)) {
        
        // Joint limits from the URDF (continuous joints: unlimited), set
        // before the optimizers copy the configuration.
        for (const auto& joint : joints) {
            if (joint.isMovable()) {
                default_config.joint_lower_limits.push_back(joint.lower_limit);
                default_config.joint_upper_limits.push_back(joint.upper_limit);
            }
        }
        setOptimizerConfig(default_config);
    }

    // The solvers hold references into this object.
    Robot(const Robot&) = delete;
    Robot& operator=(const Robot&) = delete;

    /**
     * Names of the movable joints, in joint-angle order
     */
    std::vector<std::string> getJointNames() const {
        std::vector<std::string> names;
        for (const auto& joint : joints) {
            if (joint.isMovable()) names.push_back(joint.name);
        }
        return names;
    }

    Eigen::MatrixXd getJacobian(const std::vector<double>& joint_angles) {
        return jacobianSolver.compute(joint_angles);
    }

    std::pair<Vector3, Quaternion> getEndEffectorPose(const std::vector<double>& joint_angles) {
        Transform T = fk_solver.computeFK(joint_angles);
        return {T.getPosition(), T.getQuaternion()};
    }

    /**
     * Inverse kinematics from initial_guess (default: all zeros). Pass
     * `converged` to learn whether the result actually reaches the target.
     */
    std::vector<double> computeIK(const Vector3& target_position, 
                                   const Quaternion& target_orientation,
                                   const std::vector<double>& initial_guess = {},
                                   bool* converged = nullptr) {
        return ik_solver.computeIK(target_position, target_orientation, initial_guess, converged);
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
     * MoveL - Generate linear Cartesian trajectory. Check traj.success:
     * it is false if IK failed along the line
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
     * MoveL with explicit Cartesian goal (check traj.success)
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
        ik_solver.setJointLimits(default_config.joint_lower_limits, default_config.joint_upper_limits);
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
            if (joint.isMovable()) dof++;
        }
        return dof;
    }
};