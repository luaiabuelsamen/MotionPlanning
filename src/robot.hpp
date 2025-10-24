// fk_solver.hpp
#pragma once
#include "transform.hpp"
#include "fk_solver.hpp"
#include "jacobian.hpp"
#include "ik_solver.hpp"
#include "urdf_parser.hpp"
#include <vector>
#include <iostream>

class Robot {
private:
    std::vector<Joint> joints;
    FKSolver fk_solver;
    JacobianAnalytical jacobianSolver;
    IKSolver ik_solver;
    
public:

    Robot(const std::string& urdf_file) 
        : joints(URDFParser::parseJoints(urdf_file))
        , fk_solver(joints),
        jacobianSolver(fk_solver, joints) 
        , ik_solver(jacobianSolver, fk_solver, getNumDOFs(joints)) {
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

private:
    size_t getNumDOFs(const std::vector<Joint>& joints) {
        size_t dof = 0;
        for (const auto& joint : joints) {
            if (joint.type == "revolute") dof++;
        }
        return dof;
    }
};