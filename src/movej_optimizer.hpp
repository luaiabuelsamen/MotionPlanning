#pragma once
#include "trajectory_optimizer.hpp"
#include <cmath>

/**
 * MoveJ optimizer - optimizes trajectories in joint space
 * Minimizes acceleration and jerk while smoothly moving from start to goal
 * Inspired by cuRobo's joint space trajectory optimization
 */
class MoveJOptimizer : public TrajectoryOptimizer {
public:
    MoveJOptimizer(FKSolver& fk, size_t degrees_of_freedom,
                   const OptimizerConfig& cfg = OptimizerConfig())
        : TrajectoryOptimizer(fk, degrees_of_freedom, cfg) {}
    
    /**
     * Generate linear interpolation seed trajectory
     */
    Trajectory generateLinearSeed(const std::vector<double>& start_config,
                                  const std::vector<double>& goal_config,
                                  size_t num_waypoints) {
        Trajectory traj(num_waypoints, dof, config.dt);
        
        for (size_t i = 0; i < num_waypoints; ++i) {
            double alpha = static_cast<double>(i) / (num_waypoints - 1);
            
            for (size_t j = 0; j < dof; ++j) {
                traj.points[i].position[j] = 
                    (1.0 - alpha) * start_config[j] + alpha * goal_config[j];
            }
        }
        
        computeDerivatives(traj);
        return traj;
    }
    
    /**
     * Compute goal cost - how far the final configuration is from target
     */
    double computeGoalCost(const Trajectory& traj, 
                          const std::vector<double>& goal_config) {
        double cost = 0.0;
        const auto& final_point = traj.points.back();
        
        for (size_t j = 0; j < dof; ++j) {
            double diff = final_point.position[j] - goal_config[j];
            cost += diff * diff;
        }
        
        return config.goal_weight * cost;
    }
    
    /**
     * Compute total cost for trajectory
     */
    double computeTotalCost(const Trajectory& traj,
                           const std::vector<double>& goal_config) {
        double smoothness = computeSmoothnessCost(traj);
        double goal_cost = computeGoalCost(traj, goal_config);
        return smoothness + goal_cost;
    }
    
    /**
     * Optimize trajectory using gradient descent
     * This is a simplified version of cuRobo's optimization
     */
    Trajectory optimizeGradient(Trajectory& traj,
                               const std::vector<double>& start_config,
                               const std::vector<double>& goal_config) {
        const double step_size = 0.01;
        const size_t n = traj.points.size();
        
        double prev_cost = std::numeric_limits<double>::max();
        
        for (size_t iter = 0; iter < config.max_iterations; ++iter) {
            // Recompute derivatives
            computeDerivatives(traj);
            
            // Compute current cost
            double current_cost = computeTotalCost(traj, goal_config);
            
            // Check convergence
            if (std::abs(current_cost - prev_cost) < config.convergence_threshold) {
                break;
            }
            prev_cost = current_cost;
            
            // Compute gradients using finite differences
            Eigen::MatrixXd gradient(n, dof);
            gradient.setZero();
            
            const double epsilon = 1e-6;
            
            // Only optimize interior points (keep start and goal fixed)
            for (size_t i = 1; i < n - 1; ++i) {
                for (size_t j = 0; j < dof; ++j) {
                    // Perturb position
                    double original = traj.points[i].position[j];
                    
                    traj.points[i].position[j] = original + epsilon;
                    computeDerivatives(traj);
                    double cost_plus = computeTotalCost(traj, goal_config);
                    
                    traj.points[i].position[j] = original - epsilon;
                    computeDerivatives(traj);
                    double cost_minus = computeTotalCost(traj, goal_config);
                    
                    // Restore original
                    traj.points[i].position[j] = original;
                    
                    // Compute gradient
                    gradient(i, j) = (cost_plus - cost_minus) / (2.0 * epsilon);
                }
            }
            
            // Update trajectory using gradient descent
            for (size_t i = 1; i < n - 1; ++i) {
                for (size_t j = 0; j < dof; ++j) {
                    traj.points[i].position[j] -= step_size * gradient(i, j);
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
     * Optimize trajectory using analytical approach
     * Smoother and faster than gradient descent for joint space
     */
    Trajectory optimizeAnalytical(const std::vector<double>& start_config,
                                 const std::vector<double>& goal_config,
                                 size_t num_waypoints) {
        Trajectory traj(num_waypoints, dof, config.dt);
        
        // Use minimum jerk trajectory (quintic polynomial)
        // This is a closed-form solution for smooth motion
        double total_time = (num_waypoints - 1) * config.dt;
        
        for (size_t i = 0; i < num_waypoints; ++i) {
            double t = i * config.dt;
            double s = computeMinimumJerkScaling(t, total_time);
            
            for (size_t j = 0; j < dof; ++j) {
                traj.points[i].position[j] = 
                    start_config[j] + s * (goal_config[j] - start_config[j]);
            }
        }
        
        computeDerivatives(traj);
        applyJointLimits(traj);
        
        return traj;
    }
    
    /**
     * Minimum jerk scaling function (quintic polynomial)
     * Returns value between 0 and 1 with smooth derivatives
     */
    double computeMinimumJerkScaling(double t, double T) {
        if (t <= 0) return 0.0;
        if (t >= T) return 1.0;
        
        double tau = t / T;
        // Quintic: 10τ³ - 15τ⁴ + 6τ⁵
        return 10.0 * std::pow(tau, 3) - 15.0 * std::pow(tau, 4) + 6.0 * std::pow(tau, 5);
    }
    
    /**
     * Main optimization interface
     * Uses analytical method by default (much faster and smoother)
     * Set use_gradient=true for iterative optimization
     */
    Trajectory optimize(const std::vector<double>& start_config,
                       const std::vector<double>& goal_config,
                       size_t num_waypoints) override {
        return optimizeAnalytical(start_config, goal_config, num_waypoints);
    }
    
    /**
     * Optimize with gradient descent option
     */
    Trajectory optimize(const std::vector<double>& start_config,
                       const std::vector<double>& goal_config,
                       size_t num_waypoints,
                       bool use_gradient) {
        if (use_gradient) {
            Trajectory seed = generateLinearSeed(start_config, goal_config, num_waypoints);
            return optimizeGradient(seed, start_config, goal_config);
        } else {
            return optimizeAnalytical(start_config, goal_config, num_waypoints);
        }
    }
};
