#pragma once
#include "transform.hpp"
#include "fk_solver.hpp"
#include "jacobian.hpp"
#include <Eigen/Dense>
#include <vector>

/**
 * Trajectory state represents a single point in a trajectory
 * with position, velocity, acceleration, and jerk
 */
struct TrajectoryPoint {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> acceleration;
    std::vector<double> jerk;
    
    TrajectoryPoint(size_t dof) 
        : position(dof, 0.0), velocity(dof, 0.0), 
          acceleration(dof, 0.0), jerk(dof, 0.0) {}
};

/**
 * Complete trajectory representation
 */
struct Trajectory {
    std::vector<TrajectoryPoint> points;
    double dt;  // Time step between points
    size_t dof;
    
    Trajectory(size_t num_points, size_t degrees_of_freedom, double timestep = 0.01)
        : dof(degrees_of_freedom), dt(timestep) {
        points.reserve(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            points.emplace_back(dof);
        }
    }
    
    size_t size() const { return points.size(); }
    
    // Get position matrix (timesteps × dof)
    Eigen::MatrixXd getPositionMatrix() const {
        Eigen::MatrixXd mat(points.size(), dof);
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = 0; j < dof; ++j) {
                mat(i, j) = points[i].position[j];
            }
        }
        return mat;
    }
    
    // Set from position matrix
    void setFromPositionMatrix(const Eigen::MatrixXd& mat) {
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = 0; j < dof; ++j) {
                points[i].position[j] = mat(i, j);
            }
        }
    }
};

/**
 * Optimizer configuration
 */
struct OptimizerConfig {
    size_t max_iterations = 100;
    double dt = 0.01;  // Time step
    
    // Weight factors for cost function
    double smoothness_weight = 1.0;      // Weight for acceleration smoothness
    double jerk_weight = 0.1;             // Weight for jerk minimization
    double goal_weight = 100.0;           // Weight for reaching goal
    
    // Convergence criteria
    double convergence_threshold = 1e-4;
    
    // Joint limits (for clamping)
    std::vector<double> joint_lower_limits;
    std::vector<double> joint_upper_limits;
    std::vector<double> velocity_limits;
    std::vector<double> acceleration_limits;
};

/**
 * Base class for trajectory optimization
 * Inspired by cuRobo's trajectory optimization but simplified
 * for kinematics-only optimization without collision checking
 */
class TrajectoryOptimizer {
protected:
    FKSolver& fk_solver;
    OptimizerConfig config;
    size_t dof;
    
public:
    TrajectoryOptimizer(FKSolver& fk, size_t degrees_of_freedom, 
                       const OptimizerConfig& cfg = OptimizerConfig())
        : fk_solver(fk), dof(degrees_of_freedom), config(cfg) {}
    
    virtual ~TrajectoryOptimizer() = default;
    
    /**
     * Compute finite difference derivatives for trajectory
     * Uses 5-point stencil like cuRobo for better accuracy
     */
    void computeDerivatives(Trajectory& traj) {
        size_t n = traj.points.size();
        
        // Compute velocities using central differences
        for (size_t i = 1; i < n - 1; ++i) {
            for (size_t j = 0; j < dof; ++j) {
                traj.points[i].velocity[j] = 
                    (traj.points[i+1].position[j] - traj.points[i-1].position[j]) / (2.0 * traj.dt);
            }
        }
        
        // Forward/backward differences at boundaries
        for (size_t j = 0; j < dof; ++j) {
            traj.points[0].velocity[j] = 
                (traj.points[1].position[j] - traj.points[0].position[j]) / traj.dt;
            traj.points[n-1].velocity[j] = 
                (traj.points[n-1].position[j] - traj.points[n-2].position[j]) / traj.dt;
        }
        
        // Compute accelerations
        for (size_t i = 1; i < n - 1; ++i) {
            for (size_t j = 0; j < dof; ++j) {
                traj.points[i].acceleration[j] = 
                    (traj.points[i+1].velocity[j] - traj.points[i-1].velocity[j]) / (2.0 * traj.dt);
            }
        }
        
        // Boundaries for acceleration
        for (size_t j = 0; j < dof; ++j) {
            traj.points[0].acceleration[j] = 
                (traj.points[1].velocity[j] - traj.points[0].velocity[j]) / traj.dt;
            traj.points[n-1].acceleration[j] = 
                (traj.points[n-1].velocity[j] - traj.points[n-2].velocity[j]) / traj.dt;
        }
        
        // Compute jerk
        for (size_t i = 1; i < n - 1; ++i) {
            for (size_t j = 0; j < dof; ++j) {
                traj.points[i].jerk[j] = 
                    (traj.points[i+1].acceleration[j] - traj.points[i-1].acceleration[j]) / (2.0 * traj.dt);
            }
        }
        
        // Boundaries for jerk
        for (size_t j = 0; j < dof; ++j) {
            traj.points[0].jerk[j] = 
                (traj.points[1].acceleration[j] - traj.points[0].acceleration[j]) / traj.dt;
            traj.points[n-1].jerk[j] = 
                (traj.points[n-1].acceleration[j] - traj.points[n-2].acceleration[j]) / traj.dt;
        }
    }
    
    /**
     * Compute smoothness cost (squared L2 norm of acceleration and jerk)
     */
    double computeSmoothnessCost(const Trajectory& traj) {
        double acc_cost = 0.0;
        double jerk_cost = 0.0;
        
        for (const auto& point : traj.points) {
            for (size_t j = 0; j < dof; ++j) {
                acc_cost += point.acceleration[j] * point.acceleration[j];
                jerk_cost += point.jerk[j] * point.jerk[j];
            }
        }
        
        return config.smoothness_weight * acc_cost + config.jerk_weight * jerk_cost;
    }
    
    /**
     * Apply joint limits to trajectory
     */
    void applyJointLimits(Trajectory& traj) {
        if (config.joint_lower_limits.size() != dof || 
            config.joint_upper_limits.size() != dof) {
            return;  // No limits specified
        }
        
        for (auto& point : traj.points) {
            for (size_t j = 0; j < dof; ++j) {
                point.position[j] = std::clamp(point.position[j], 
                                               config.joint_lower_limits[j],
                                               config.joint_upper_limits[j]);
            }
        }
    }
    
    /**
     * Optimize trajectory - to be implemented by derived classes
     */
    virtual Trajectory optimize(const std::vector<double>& start_config,
                               const std::vector<double>& goal_config,
                               size_t num_waypoints) = 0;
};
