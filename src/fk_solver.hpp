// fk_solver.hpp
#pragma once
#include "transform.hpp"
#include "urdf_parser.hpp"
#include <vector>
// fk_solver.hpp
class FKSolver {
private:
    const std::vector<Joint>& joints;
    
public:
    std::vector<Transform> transforms;

    FKSolver(const std::vector<Joint>& joints) 
        : joints(joints), transforms(joints.size() + 1) {  // +1 for base!
        transforms[0] = Transform();  // Base/identity
    }
    
    // fk_solver.hpp - CORRECTED
    Transform computeFK(const std::vector<double>& joint_angles) {
        transforms[0] = Transform();  // Base is identity
        
        size_t angle_idx = 0;
        for (size_t i = 0; i < joints.size(); i++) {
            const Joint& joint = joints[i];
            
            // ALWAYS apply the joint's origin transform (translation + fixed rotation)
            Transform origin_transform = Transform::fromRPY(
                joint.origin_rpy.x(), joint.origin_rpy.y(), joint.origin_rpy.z(),
                joint.origin_xyz.x(), joint.origin_xyz.y(), joint.origin_xyz.z()
            );
            
            // Start with transform up to this joint's origin
            Transform accumulated = Transform(transforms[i] * origin_transform);
            
            // For revolute joints, add variable rotation
            if (joint.type == "revolute" && angle_idx < joint_angles.size()) {
                double angle = joint_angles[angle_idx++];
                Transform rotation;
                
                if (joint.axis.z() != 0) {
                    rotation = Transform::fromRPY(0, 0, angle * joint.axis.z(), 0, 0, 0);
                } else if (joint.axis.y() != 0) {
                    rotation = Transform::fromRPY(0, angle * joint.axis.y(), 0, 0, 0, 0);
                } else if (joint.axis.x() != 0) {
                    rotation = Transform::fromRPY(angle * joint.axis.x(), 0, 0, 0, 0, 0);
                }
                
                accumulated = Transform(accumulated * rotation);
            }
            
            transforms[i + 1] = accumulated;
        }
        
        return transforms.back();
    }
};