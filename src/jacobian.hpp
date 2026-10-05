#pragma once
#include "fk_solver.hpp"
#include "transform.hpp"
#include <vector>

// Geometric Jacobian of the tip frame: rows 0-2 linear velocity, rows 3-5
// angular velocity, both in the base frame; one column per movable joint.
class JacobianAnalytical {
private:
    FKSolver& fk_solver;
    std::vector<Joint> joints;

public:
    JacobianAnalytical(FKSolver& solver, const std::vector<Joint>& j)
        : fk_solver(solver), joints(j) {}

    Eigen::MatrixXd compute(const std::vector<double>& joint_angles) {
        size_t n_joints = joint_angles.size();
        Eigen::MatrixXd J(6, n_joints);
        J.setZero();

        fk_solver.computeFK(joint_angles);

        Vector3 p_end = fk_solver.transforms.back().getPosition();
        size_t col = 0;

        for (size_t i = 0; i < joints.size() && col < n_joints; i++) {
            if (!joints[i].isMovable()) continue;

            Transform joint_frame = Transform(fk_solver.transforms[i] * FKSolver::originTransform(joints[i]));
            Vector3 axis = joint_frame.rotation() * joints[i].axis;

            if (joints[i].isRevolute()) {
                J.col(col).head<3>() = axis.cross(p_end - joint_frame.getPosition());
                J.col(col).tail<3>() = axis;
            } else {
                J.col(col).head<3>() = axis;
            }
            col++;
        }

        return J;
    }
};
