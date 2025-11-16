#pragma once
#include "trajectory_optimizer.hpp"
#include "ik_solver.hpp"
#include "jacobian.hpp"
#include <Eigen/Geometry>

/**
 * MoveL optimizer - optimizes trajectories for linear Cartesian motion
 * Creates a straight line path in Cartesian space and optimizes joint trajectory
 * Inspired by cuRobo but optimized for speed without GPU
 * 
 * Performance optimizations:
 * - IK-only approach (no iterative refinement by default)
 * - Minimum jerk smoothing in joint space after IK
 * - Optional fast refinement with reduced iterations
 */
class MoveLOptimizer : public TrajectoryOptimizer {
private:
    IKSolver& ik_solver;
    JacobianAnalytical& jacobian_solver;
    bool use_fast_mode;  // Skip expensive optimization, rely on IK + smoothing
    
public:
    MoveLOptimizer(FKSolver& fk, IKSolver& ik, JacobianAnalytical& jac,
                   size_t degrees_of_freedom,
                   const OptimizerConfig& cfg = OptimizerConfig(),
                   bool fast_mode = true)
        : TrajectoryOptimizer(fk, degrees_of_freedom, cfg),
          ik_solver(ik), jacobian_solver(jac), use_fast_mode(fast_mode) {}
    
    /**
     * Generate Cartesian linear path waypoints
     * Returns vector of (position, orientation) pairs
     */
    std::vector<std::pair<Vector3, Quaternion>> 
    generateCartesianPath(const Vector3& start_pos, const Quaternion& start_quat,
                         const Vector3& goal_pos, const Quaternion& goal_quat,
                         size_t num_waypoints) {
        std::vector<std::pair<Vector3, Quaternion>> path;
        path.reserve(num_waypoints);
        
        for (size_t i = 0; i < num_waypoints; ++i) {
            double alpha = static_cast<double>(i) / (num_waypoints - 1);
            
            // Linear interpolation for position
            Vector3 pos = (1.0 - alpha) * start_pos + alpha * goal_pos;
            
            // Spherical linear interpolation (slerp) for orientation
            Quaternion quat = start_quat.slerp(alpha, goal_quat);
            
            path.emplace_back(pos, quat);
        }
        
        return path;
    }
    
    /**
     * Generate seed trajectory using IK for each Cartesian waypoint
     * This is the main bottleneck - IK called for every waypoint
     */
    Trajectory generateIKSeed(const std::vector<double>& start_config,
                             const std::vector<std::pair<Vector3, Quaternion>>& cartesian_path) {
        size_t num_waypoints = cartesian_path.size();
        Trajectory traj(num_waypoints, dof, config.dt);
        
        // Set start configuration
        traj.points[0].position = start_config;
        
        // Solve IK for each waypoint, using previous solution as seed (warm start)
        std::vector<double> current_config = start_config;
        
        for (size_t i = 1; i < num_waypoints; ++i) {
            const auto& [pos, quat] = cartesian_path[i];
            
            // Fast IK with reduced iterations (cuRobo uses similar approach)
            current_config = ik_solver.computeIK(pos, quat, current_config);
            
            traj.points[i].position = current_config;
        }
        
        computeDerivatives(traj);
        return traj;
    }
    
    /**
     * Fast joint space smoothing after IK
     * Much faster than iterative Cartesian optimization
     * Similar to cuRobo's approach of IK + smoothing
     */
    Trajectory smoothJointTrajectory(const Trajectory& traj) {
        Trajectory smoothed = traj;
        
        // Simple weighted averaging to smooth (like a low-pass filter)
        // Keep endpoints fixed
        for (size_t iter = 0; iter < 3; ++iter) {  // Just 3 passes
            Trajectory temp = smoothed;
            for (size_t i = 1; i < traj.size() - 1; ++i) {
                for (size_t j = 0; j < dof; ++j) {
                    // Weighted average with neighbors
                    smoothed.points[i].position[j] = 
                        0.25 * temp.points[i-1].position[j] +
                        0.50 * temp.points[i].position[j] +
                        0.25 * temp.points[i+1].position[j];
                }
            }
        }
        
        computeDerivatives(smoothed);
        applyJointLimits(smoothed);
        return smoothed;
    }
    
    /**
     * Compute Cartesian path error (for verification only in fast mode)
     */
    double computeCartesianError(const Trajectory& traj,
                                 const std::vector<std::pair<Vector3, Quaternion>>& cartesian_path) {
        double error = 0.0;
        
        for (size_t i = 0; i < traj.points.size(); ++i) {
            // Compute FK for current joint configuration
            Transform current_transform = fk_solver.computeFK(traj.points[i].position);
            Vector3 current_pos = current_transform.getPosition();
            Quaternion current_quat = current_transform.getQuaternion();
            
            const auto& [target_pos, target_quat] = cartesian_path[i];
            
            // Position error
            Vector3 pos_error = current_pos - target_pos;
            error += pos_error.squaredNorm();
            
            // Orientation error
            Quaternion q_error = target_quat * current_quat.conjugate();
            Eigen::Vector3d orient_error(q_error.x(), q_error.y(), q_error.z());
            error += 10.0 * orient_error.squaredNorm();
        }
        
        return config.goal_weight * error;
    }
    
    /**
     * Fast Cartesian trajectory optimization (default)
     * Uses IK + joint space smoothing - much faster than iterative optimization
     * Similar to cuRobo's approach without GPU
     */
    Trajectory optimizeFast(const std::vector<double>& start_config,
                           const std::vector<std::pair<Vector3, Quaternion>>& cartesian_path) {
        // Generate IK-based trajectory (main bottleneck but unavoidable)
        Trajectory traj = generateIKSeed(start_config, cartesian_path);
        
        // Apply fast joint space smoothing instead of expensive optimization
        traj = smoothJointTrajectory(traj);
        
        return traj;
    }
    
    /**
     * Slow but more accurate Cartesian optimization (for comparison)
     * Uses Jacobian-based refinement - expensive!
     * Set use_fast_mode=false in constructor to enable
     */
    Trajectory optimizeAccurate(Trajectory& traj,
                                const std::vector<std::pair<Vector3, Quaternion>>& cartesian_path) {
        const double step_size = 0.01;  // Increased step size
        const size_t n = traj.points.size();
        const size_t max_iter = 5;  // Reduced iterations (was config.max_iterations)
        
        for (size_t iter = 0; iter < max_iter; ++iter) {
            // Update using Jacobian-based optimization (expensive!)
            for (size_t i = 1; i < n - 1; ++i) {
                // Compute FK for current configuration
                Transform current_transform = fk_solver.computeFK(traj.points[i].position);
                Vector3 current_pos = current_transform.getPosition();
                Quaternion current_quat = current_transform.getQuaternion();
                
                const auto& [target_pos, target_quat] = cartesian_path[i];
                
                // Compute Cartesian error
                Vector3 pos_error = target_pos - current_pos;
                
                // Orientation error
                Quaternion q_error = target_quat * current_quat.conjugate();
                Eigen::Vector3d orient_error;
                orient_error << q_error.x(), q_error.y(), q_error.z();
                orient_error *= 2.0;
                
                // Combine into 6D error vector
                Eigen::VectorXd error(6);
                error << pos_error.x(), pos_error.y(), pos_error.z(),
                         orient_error(0), orient_error(1), orient_error(2);
                
                // Compute Jacobian
                Eigen::MatrixXd J = jacobian_solver.compute(traj.points[i].position);
                
                // Compute joint space correction using damped least squares
                double damping = 0.01;
                Eigen::MatrixXd JJt = J * J.transpose();
                Eigen::MatrixXd damped = JJt + damping * damping * Eigen::MatrixXd::Identity(6, 6);
                Eigen::VectorXd delta_q = J.transpose() * damped.inverse() * error;
                
                // Update joint configuration
                for (size_t j = 0; j < dof; ++j) {
                    traj.points[i].position[j] += step_size * delta_q(j);
                }
            }
            
            // Apply joint limits
            applyJointLimits(traj);
        }
        
        // Final derivative computation
        computeDerivatives(traj);
        
        return traj;
    }
    
    /**
     * Main optimization interface for MoveL
     * Uses fast mode by default (IK + smoothing)
     */
    Trajectory optimize(const std::vector<double>& start_config,
                       const std::vector<double>& goal_config,
                       size_t num_waypoints) override {
        // Compute start and goal Cartesian poses
        Transform start_transform = fk_solver.computeFK(start_config);
        Vector3 start_pos = start_transform.getPosition();
        Quaternion start_quat = start_transform.getQuaternion();
        
        Transform goal_transform = fk_solver.computeFK(goal_config);
        Vector3 goal_pos = goal_transform.getPosition();
        Quaternion goal_quat = goal_transform.getQuaternion();
        
        // Generate Cartesian linear path
        auto cartesian_path = generateCartesianPath(start_pos, start_quat,
                                                    goal_pos, goal_quat,
                                                    num_waypoints);
        
        if (use_fast_mode) {
            // Fast mode: IK + smoothing (recommended)
            return optimizeFast(start_config, cartesian_path);
        } else {
            // Accurate mode: IK + Jacobian refinement (slower)
            Trajectory traj = generateIKSeed(start_config, cartesian_path);
            return optimizeAccurate(traj, cartesian_path);
        }
    }
    
    /**
     * Optimize with custom Cartesian waypoints
     */
    Trajectory optimizeWithPath(const std::vector<double>& start_config,
                               const Vector3& goal_pos,
                               const Quaternion& goal_quat,
                               size_t num_waypoints) {
        // Compute start Cartesian pose
        Transform start_transform = fk_solver.computeFK(start_config);
        Vector3 start_pos = start_transform.getPosition();
        Quaternion start_quat = start_transform.getQuaternion();
        
        // Generate Cartesian linear path
        auto cartesian_path = generateCartesianPath(start_pos, start_quat,
                                                    goal_pos, goal_quat,
                                                    num_waypoints);
        
        if (use_fast_mode) {
            return optimizeFast(start_config, cartesian_path);
        } else {
            Trajectory traj = generateIKSeed(start_config, cartesian_path);
            return optimizeAccurate(traj, cartesian_path);
        }
    }
};