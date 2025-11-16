#pragma once
#include "trajectory_optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace TrajectoryUtils {

/**
 * Linear interpolation between two trajectories
 */
inline Trajectory linearInterpolate(const Trajectory& traj, size_t new_num_points) {
    if (new_num_points <= traj.size()) {
        // Simple decimation if fewer points needed
        Trajectory result(new_num_points, traj.dof, traj.dt);
        double stride = static_cast<double>(traj.size() - 1) / (new_num_points - 1);
        
        for (size_t i = 0; i < new_num_points; ++i) {
            size_t idx = static_cast<size_t>(i * stride);
            result.points[i] = traj.points[idx];
        }
        return result;
    }
    
    // Upsample trajectory
    Trajectory result(new_num_points, traj.dof, traj.dt);
    
    for (size_t i = 0; i < new_num_points; ++i) {
        // Find position in original trajectory
        double pos = static_cast<double>(i) * (traj.size() - 1) / (new_num_points - 1);
        size_t idx0 = static_cast<size_t>(std::floor(pos));
        size_t idx1 = std::min(idx0 + 1, traj.size() - 1);
        double alpha = pos - idx0;
        
        // Interpolate position
        for (size_t j = 0; j < traj.dof; ++j) {
            result.points[i].position[j] = 
                (1.0 - alpha) * traj.points[idx0].position[j] + 
                alpha * traj.points[idx1].position[j];
        }
    }
    
    return result;
}

/**
 * Cubic spline interpolation for smoother trajectories
 */
inline Trajectory cubicSplineInterpolate(const Trajectory& traj, size_t new_num_points) {
    Trajectory result(new_num_points, traj.dof, traj.dt);
    
    // For each DOF, perform cubic interpolation
    for (size_t j = 0; j < traj.dof; ++j) {
        // Extract positions for this DOF
        std::vector<double> y(traj.size());
        for (size_t i = 0; i < traj.size(); ++i) {
            y[i] = traj.points[i].position[j];
        }
        
        // Compute second derivatives for cubic spline (natural boundary conditions)
        std::vector<double> y2(traj.size(), 0.0);
        std::vector<double> u(traj.size() - 1);
        
        // Forward pass
        for (size_t i = 1; i < traj.size() - 1; ++i) {
            double sig = 0.5;
            double p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;
            u[i] = (y[i + 1] - 2.0 * y[i] + y[i - 1]) / traj.dt;
            u[i] = (6.0 * u[i] / (2.0 * traj.dt) - sig * u[i - 1]) / p;
        }
        
        // Backward pass
        for (int i = traj.size() - 2; i >= 0; --i) {
            y2[i] = y2[i] * y2[i + 1] + u[i];
        }
        
        // Interpolate at new points
        for (size_t i = 0; i < new_num_points; ++i) {
            double pos = static_cast<double>(i) * (traj.size() - 1) / (new_num_points - 1);
            size_t idx0 = static_cast<size_t>(std::floor(pos));
            size_t idx1 = std::min(idx0 + 1, traj.size() - 1);
            
            double h = traj.dt;
            double a = idx1 - pos;
            double b = pos - idx0;
            
            result.points[i].position[j] = 
                a * y[idx0] + b * y[idx1] +
                ((a * a * a - a) * y2[idx0] + (b * b * b - b) * y2[idx1]) * (h * h) / 6.0;
        }
    }
    
    return result;
}

/**
 * Adjust trajectory timing to respect velocity and acceleration limits
 * Similar to cuRobo's time-optimal trajectory scaling
 */
inline Trajectory scaleTrajectoryTime(const Trajectory& traj,
                                     const std::vector<double>& max_vel,
                                     const std::vector<double>& max_acc) {
    if (max_vel.size() != traj.dof || max_acc.size() != traj.dof) {
        return traj;  // Can't scale without limits
    }
    
    // Find maximum velocity scaling factor needed
    double max_vel_scale = 0.0;
    for (size_t i = 0; i < traj.points.size(); ++i) {
        for (size_t j = 0; j < traj.dof; ++j) {
            double vel_ratio = std::abs(traj.points[i].velocity[j]) / max_vel[j];
            max_vel_scale = std::max(max_vel_scale, vel_ratio);
        }
    }
    
    // Find maximum acceleration scaling factor needed
    double max_acc_scale = 0.0;
    for (size_t i = 0; i < traj.points.size(); ++i) {
        for (size_t j = 0; j < traj.dof; ++j) {
            double acc_ratio = std::abs(traj.points[i].acceleration[j]) / max_acc[j];
            max_acc_scale = std::max(max_acc_scale, acc_ratio);
        }
    }
    
    // Time scaling factor (need to scale time, which scales velocity down and acc down²)
    double time_scale = std::max(max_vel_scale, std::sqrt(max_acc_scale));
    
    if (time_scale <= 1.0) {
        return traj;  // Already within limits
    }
    
    // Create new trajectory with scaled time
    Trajectory scaled = traj;
    scaled.dt *= time_scale;
    
    // Scale velocities and accelerations
    for (auto& point : scaled.points) {
        for (size_t j = 0; j < traj.dof; ++j) {
            point.velocity[j] /= time_scale;
            point.acceleration[j] /= (time_scale * time_scale);
            point.jerk[j] /= (time_scale * time_scale * time_scale);
        }
    }
    
    return scaled;
}

/**
 * Get total trajectory duration
 */
inline double getTrajectoryduration(const Trajectory& traj) {
    return traj.dt * (traj.size() - 1);
}

/**
 * Extract positions at specific time
 */
inline std::vector<double> interpolateAtTime(const Trajectory& traj, double time) {
    if (time <= 0.0) {
        return traj.points[0].position;
    }
    
    double total_time = getTrajectoryduration(traj);
    if (time >= total_time) {
        return traj.points.back().position;
    }
    
    // Find surrounding waypoints
    double pos = time / traj.dt;
    size_t idx0 = static_cast<size_t>(std::floor(pos));
    size_t idx1 = std::min(idx0 + 1, traj.size() - 1);
    double alpha = pos - idx0;
    
    // Linear interpolation
    std::vector<double> result(traj.dof);
    for (size_t j = 0; j < traj.dof; ++j) {
        result[j] = (1.0 - alpha) * traj.points[idx0].position[j] + 
                    alpha * traj.points[idx1].position[j];
    }
    
    return result;
}

/**
 * Smooth trajectory using moving average filter
 */
inline Trajectory smoothTrajectory(const Trajectory& traj, size_t window_size = 5) {
    if (window_size < 3 || window_size >= traj.size()) {
        return traj;
    }
    
    Trajectory smoothed = traj;
    size_t half_window = window_size / 2;
    
    // Smooth interior points (keep start and end fixed)
    for (size_t i = half_window; i < traj.size() - half_window; ++i) {
        for (size_t j = 0; j < traj.dof; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < window_size; ++k) {
                sum += traj.points[i - half_window + k].position[j];
            }
            smoothed.points[i].position[j] = sum / window_size;
        }
    }
    
    return smoothed;
}

/**
 * Check if trajectory respects joint limits
 */
inline bool checkJointLimits(const Trajectory& traj,
                            const std::vector<double>& lower_limits,
                            const std::vector<double>& upper_limits) {
    if (lower_limits.size() != traj.dof || upper_limits.size() != traj.dof) {
        return true;  // No limits to check
    }
    
    for (const auto& point : traj.points) {
        for (size_t j = 0; j < traj.dof; ++j) {
            if (point.position[j] < lower_limits[j] || 
                point.position[j] > upper_limits[j]) {
                return false;
            }
        }
    }
    
    return true;
}

/**
 * Check if trajectory respects velocity limits
 */
inline bool checkVelocityLimits(const Trajectory& traj,
                               const std::vector<double>& max_vel) {
    if (max_vel.size() != traj.dof) {
        return true;  // No limits to check
    }
    
    for (const auto& point : traj.points) {
        for (size_t j = 0; j < traj.dof; ++j) {
            if (std::abs(point.velocity[j]) > max_vel[j]) {
                return false;
            }
        }
    }
    
    return true;
}

/**
 * Check if trajectory respects acceleration limits
 */
inline bool checkAccelerationLimits(const Trajectory& traj,
                                   const std::vector<double>& max_acc) {
    if (max_acc.size() != traj.dof) {
        return true;  // No limits to check
    }
    
    for (const auto& point : traj.points) {
        for (size_t j = 0; j < traj.dof; ++j) {
            if (std::abs(point.acceleration[j]) > max_acc[j]) {
                return false;
            }
        }
    }
    
    return true;
}

/**
 * Print trajectory statistics
 */
inline void printTrajectoryStats(const Trajectory& traj) {
    std::cout << "Trajectory Statistics:\n";
    std::cout << "  Duration: " << getTrajectoryduration(traj) << " s\n";
    std::cout << "  Waypoints: " << traj.size() << "\n";
    std::cout << "  DOF: " << traj.dof << "\n";
    
    // Compute max velocity and acceleration per joint
    std::vector<double> max_vel(traj.dof, 0.0);
    std::vector<double> max_acc(traj.dof, 0.0);
    
    for (const auto& point : traj.points) {
        for (size_t j = 0; j < traj.dof; ++j) {
            max_vel[j] = std::max(max_vel[j], std::abs(point.velocity[j]));
            max_acc[j] = std::max(max_acc[j], std::abs(point.acceleration[j]));
        }
    }
    
    std::cout << "  Max velocities: [";
    for (size_t j = 0; j < traj.dof; ++j) {
        std::cout << max_vel[j];
        if (j < traj.dof - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    
    std::cout << "  Max accelerations: [";
    for (size_t j = 0; j < traj.dof; ++j) {
        std::cout << max_acc[j];
        if (j < traj.dof - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

} // namespace TrajectoryUtils
