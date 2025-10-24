#include "fk_solver.hpp"
#include "transform.hpp"
#include <vector>
#pragma once

class JacobianAnalytical {
private:
    FKSolver& fk_solver;
    std::vector<Joint> joints;
    
public:
    JacobianAnalytical(FKSolver& solver, const std::vector<Joint>& j) 
        : fk_solver(solver), joints(j) {}
    
    // Return Eigen matrix instead of std::vector
    Eigen::MatrixXd compute(const std::vector<double>& joint_angles) {
        size_t n_joints = joint_angles.size();
        Eigen::MatrixXd J(6, n_joints);  // 6 rows, n_joints columns
        J.setZero();
        
        // Ensure FK is computed
        fk_solver.computeFK(joint_angles);
        
        Vector3 p_end = fk_solver.transforms.back().getPosition();
        size_t revolute_idx = 0;
        
        for (size_t i = 0; i < joints.size() && revolute_idx < n_joints; i++) {
            if (joints[i].type != "revolute") continue;
            
            Transform origin_transform = Transform::fromRPY(
                joints[i].origin_rpy.x(), joints[i].origin_rpy.y(), joints[i].origin_rpy.z(),
                joints[i].origin_xyz.x(), joints[i].origin_xyz.y(), joints[i].origin_xyz.z()
            );
            
            Transform joint_frame = Transform(fk_solver.transforms[i] * origin_transform);
            Vector3 p_joint = joint_frame.getPosition();
            Vector3 z_joint = joint_frame.rotation() * joints[i].axis;
            Vector3 diff = p_end - p_joint;
            Vector3 j_linear = z_joint.cross(diff);

            // Much cleaner with Eigen!
            J.col(revolute_idx).head<3>() = j_linear;      // Linear part (rows 0-2)
            J.col(revolute_idx).tail<3>() = z_joint;       // Angular part (rows 3-5)
            
            revolute_idx++;
        }
        
        return J;
    }
};