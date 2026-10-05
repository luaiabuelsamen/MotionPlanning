#pragma once
#include "jacobian.hpp"
#include "transform.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

struct SolverOptions {
    double position_tolerance = 1e-4;
    double orientation_tolerance = 1e-3;  // Radians
    size_t max_iterations = 100;
    double damping = 0.01;  // For damped least squares
    double step_size = 1.0;  // Can reduce if oscillating
    double max_step = 0.2;   // Largest change of any joint per iteration (rad or m)
};

// Rotation taking `current` to `target`, as a rotation vector (axis * angle,
// angle in [0, pi]) in the base frame.
inline Eigen::Vector3d orientationError(const Quaternion& target, const Quaternion& current) {
    Quaternion q = target * current.conjugate();
    if (q.w() < 0) q.coeffs() = -q.coeffs();  // q and -q are the same rotation: take the short way
    Eigen::Vector3d v(q.x(), q.y(), q.z());
    double s = v.norm();
    if (s < 1e-12) return 2.0 * v;
    return (2.0 * std::atan2(s, q.w()) / s) * v;
}

class IKSolver {
private:
    JacobianAnalytical& jacobian_solver;
    FKSolver& fk_solver;
    const size_t n_joints;
    const size_t task_dim;  //3 for position-only, 6 for full pose
    SolverOptions options;
    std::vector<double> lower_limits, upper_limits;

public:
    IKSolver(JacobianAnalytical& jac_solver, FKSolver& fk, size_t num_joints,
             size_t task_dimension = 6,  // Default to full 6D pose
             SolverOptions opts = SolverOptions())
        : jacobian_solver(jac_solver)
        , fk_solver(fk)
        , n_joints(num_joints)
        , task_dim(task_dimension)
        , options(opts) {}

    // Solutions are kept within these limits (empty: unlimited).
    void setJointLimits(const std::vector<double>& lower, const std::vector<double>& upper) {
        lower_limits = lower;
        upper_limits = upper;
    }

    const SolverOptions& getOptions() const { return options; }
    void setOptions(const SolverOptions& opts) { options = opts; }

    // Damped least squares from initial_guess. If `converged` is given, it is
    // set to whether the tolerances were met; otherwise the returned angles
    // are the last iterate, which may be far from the target.
    std::vector<double> computeIK(const Vector3& target_position,
                                   const Quaternion& target_orientation,
                                   const std::vector<double>& initial_guess = {},
                                   bool* converged = nullptr) {

        std::vector<double> joint_angles = initial_guess.empty() ?
                                           std::vector<double>(n_joints, 0.0) :
                                           initial_guess;
        clampToLimits(joint_angles);
        Quaternion target_quat = target_orientation.normalized();

        for (size_t iteration = 0; iteration <= options.max_iterations; ++iteration) {
            Transform current_transform = fk_solver.computeFK(joint_angles);
            Vector3 pos_error = target_position - current_transform.getPosition();
            Eigen::Vector3d orientation_error = orientationError(target_quat, current_transform.getQuaternion());

            if (pos_error.norm() < options.position_tolerance &&
                (task_dim == 3 || orientation_error.norm() < options.orientation_tolerance)) {
                if (converged) *converged = true;
                return joint_angles;
            }
            if (iteration == options.max_iterations) break;

            Eigen::VectorXd error(task_dim);
            if (task_dim == 3) {
                error << pos_error;
            } else {
                error << pos_error, orientation_error;
            }

            Eigen::MatrixXd J = jacobian_solver.compute(joint_angles);
            if (task_dim == 3) {
                J = J.topRows(3).eval();
            }

            // Damped least squares, with the step limited so that near
            // singularities the solver cannot leap to another IK branch.
            Eigen::MatrixXd damped = J * J.transpose() +
                options.damping * options.damping * Eigen::MatrixXd::Identity(task_dim, task_dim);
            Eigen::VectorXd delta_q = options.step_size * (J.transpose() * damped.ldlt().solve(error));
            double largest = delta_q.cwiseAbs().maxCoeff();
            if (largest > options.max_step) delta_q *= options.max_step / largest;

            for (size_t i = 0; i < n_joints; ++i) {
                joint_angles[i] += delta_q(i);
            }
            clampToLimits(joint_angles);
        }

        if (converged) *converged = false;
        return joint_angles;
    }

private:
    void clampToLimits(std::vector<double>& q) const {
        if (lower_limits.size() != q.size() || upper_limits.size() != q.size()) return;
        for (size_t i = 0; i < q.size(); ++i) q[i] = std::clamp(q[i], lower_limits[i], upper_limits[i]);
    }
};
