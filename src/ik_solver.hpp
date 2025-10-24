#pragma once
#include "jacobian.hpp"
#include "transform.hpp"
#include <Eigen/Dense>

struct SolverOptions {
    double position_tolerance = 1e-4;
    double orientation_tolerance = 1e-3;  // Radians
    size_t max_iterations = 100;
    double damping = 0.01;  // For damped least squares
    double step_size = 1.0;  // Can reduce if oscillating
};

class IKSolver {
private:
    JacobianAnalytical& jacobian_solver;
    FKSolver& fk_solver;
    const size_t n_joints;
    const size_t task_dim;  //3 for position-only, 6 for full pose
    SolverOptions options;

public:
    IKSolver(JacobianAnalytical& jac_solver, FKSolver& fk, size_t num_joints,
             size_t task_dimension = 6,  // Default to full 6D pose
             SolverOptions opts = SolverOptions())
        : jacobian_solver(jac_solver)
        , fk_solver(fk)
        , n_joints(num_joints)
        , task_dim(task_dimension)
        , options(opts) {}

    std::vector<double> computeIK(const Vector3& target_position, 
                                   const Quaternion& target_orientation,
                                   const std::vector<double>& initial_guess = {}) {
        
        std::vector<double> joint_angles = initial_guess.empty() ? 
                                           std::vector<double>(n_joints, 0.0) : 
                                           initial_guess;
        
        for (size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
            Transform current_transform = fk_solver.computeFK(joint_angles);
            Vector3 current_pos = current_transform.getPosition();
            Quaternion current_quat = current_transform.getQuaternion();
            
            // Position error
            Vector3 pos_error = target_position - current_pos;
            
            // Orientation error
            Quaternion q_error = target_orientation * current_quat.conjugate();
            Eigen::Vector3d orientation_error;
            orientation_error << q_error.x(), q_error.y(), q_error.z();
            orientation_error *= 2.0;
            
            // Combine into task_dim error vector
            Eigen::VectorXd error(task_dim);
            if (task_dim == 3) {
                // Position-only
                error << pos_error.x(), pos_error.y(), pos_error.z();
            } else if (task_dim == 6) {
                // Full pose
                error << pos_error.x(), pos_error.y(), pos_error.z(),
                         orientation_error(0), orientation_error(1), orientation_error(2);
            }
            
            // Check convergence
            double pos_error_norm = pos_error.norm();
            double orient_error_norm = orientation_error.norm();
            
            if (pos_error_norm < options.position_tolerance && 
                (task_dim == 3 || orient_error_norm < options.orientation_tolerance)) {
                return joint_angles;
            }
            
            // Compute Jacobian (task_dim × n_joints)
            Eigen::MatrixXd J = jacobian_solver.compute(joint_angles);
            
            // Extract relevant rows based on task_dim
            if (task_dim == 3) {
                J = J.topRows(3);  // Only position rows
            }
            
            // Damped Least Squares
            Eigen::MatrixXd JJt = J * J.transpose();
            Eigen::MatrixXd damped = JJt + 
                options.damping * options.damping * Eigen::MatrixXd::Identity(task_dim, task_dim);
            
            Eigen::VectorXd delta_q = J.transpose() * damped.inverse() * error;
            
            for (size_t i = 0; i < n_joints; ++i) {
                joint_angles[i] += options.step_size * delta_q(i);
            }
        }
        
        return joint_angles;
    }
};