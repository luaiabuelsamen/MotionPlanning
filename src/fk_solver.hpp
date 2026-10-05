// fk_solver.hpp
#pragma once
#include "transform.hpp"
#include "urdf_parser.hpp"
#include <vector>

// Forward kinematics along a joint chain (root to tip, see
// URDFParser::parseChain). transforms[i] is the frame of the link before
// joint i; transforms.back() is the tip link.
class FKSolver {
private:
    const std::vector<Joint>& joints;

public:
    std::vector<Transform> transforms;

    FKSolver(const std::vector<Joint>& joints)
        : joints(joints), transforms(joints.size() + 1) {  // +1 for base!
        transforms[0] = Transform();  // Base/identity
    }

    // The joint's motion for joint position q (angle or offset).
    static Transform jointMotion(const Joint& joint, double q) {
        Transform motion;
        if (joint.isRevolute()) {
            motion.linear() = Eigen::AngleAxisd(q, joint.axis).toRotationMatrix();
        } else if (joint.isPrismatic()) {
            motion.translation() = q * joint.axis;
        }
        return motion;
    }

    static Transform originTransform(const Joint& joint) {
        return Transform::fromRPY(
            joint.origin_rpy.x(), joint.origin_rpy.y(), joint.origin_rpy.z(),
            joint.origin_xyz.x(), joint.origin_xyz.y(), joint.origin_xyz.z());
    }

    // joint_angles holds one value per movable joint, in chain order.
    Transform computeFK(const std::vector<double>& joint_angles) {
        transforms[0] = Transform();  // Base is identity

        size_t angle_idx = 0;
        for (size_t i = 0; i < joints.size(); i++) {
            const Joint& joint = joints[i];
            Transform accumulated = Transform(transforms[i] * originTransform(joint));
            if (joint.isMovable() && angle_idx < joint_angles.size()) {
                accumulated = Transform(accumulated * jointMotion(joint, joint_angles[angle_idx++]));
            }
            transforms[i + 1] = accumulated;
        }

        return transforms.back();
    }
};
