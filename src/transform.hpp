// transform.hpp
#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>

using Vector3 = Eigen::Vector3d;
using Quaternion = Eigen::Quaterniond;

class Transform : public Eigen::Isometry3d {
public:
    Transform() : Eigen::Isometry3d(Eigen::Isometry3d::Identity()) {}
    Transform(const Eigen::Isometry3d& other) : Eigen::Isometry3d(other) {}
    
    // Create transform from RPY + translation
    static Transform fromRPY(double roll, double pitch, double yaw, 
                             double x, double y, double z) {
        Transform t;
        t.linear() = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                      Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
        t.translation() = Eigen::Vector3d(x, y, z);
        return t;
    }
    
    // Extract position
    Vector3 getPosition() const {
        return translation();
    }
    
    // Get quaternion
    Quaternion getQuaternion() const {
        return Quaternion(rotation());
    }
};