// urdf_parser.hpp
#pragma once
#include "transform.hpp"
#include <string>
#include <vector>
#include <tinyxml2.h>  // You'll need to install this: apt install libtinyxml2-dev

struct Joint {
    std::string name;
    std::string type;
    Vector3 origin_xyz;
    Vector3 origin_rpy;
    Vector3 axis;
    double lower_limit;
    double upper_limit;
};

class URDFParser {
public:
    static std::vector<Joint> parseJoints(const std::string& urdf_file) {
        std::vector<Joint> joints;
        tinyxml2::XMLDocument doc;
        
        if (doc.LoadFile(urdf_file.c_str()) != tinyxml2::XML_SUCCESS) {
            return joints;
        }
        
        auto* robot = doc.FirstChildElement("robot");
        if (!robot) return joints;
        
        for (auto* joint_elem = robot->FirstChildElement("joint");
             joint_elem != nullptr;
             joint_elem = joint_elem->NextSiblingElement("joint")) {
            
            Joint joint;
            joint.name = joint_elem->Attribute("name");
            joint.type = joint_elem->Attribute("type");
            
            // Parse origin
            if (auto* origin = joint_elem->FirstChildElement("origin")) {
                if (origin->Attribute("xyz")) {
                    sscanf(origin->Attribute("xyz"), "%lf %lf %lf",
                           &joint.origin_xyz[0], &joint.origin_xyz[1], &joint.origin_xyz[2]);
                }
                if (origin->Attribute("rpy")) {
                    sscanf(origin->Attribute("rpy"), "%lf %lf %lf",
                           &joint.origin_rpy[0], &joint.origin_rpy[1], &joint.origin_rpy[2]);
                }
            }
            
            // Parse axis
            if (auto* axis = joint_elem->FirstChildElement("axis")) {
                if (axis->Attribute("xyz")) {
                    sscanf(axis->Attribute("xyz"), "%lf %lf %lf",
                           &joint.axis[0], &joint.axis[1], &joint.axis[2]);
                }
            }
            
            // Parse limits
            if (auto* limit = joint_elem->FirstChildElement("limit")) {
                joint.lower_limit = limit->DoubleAttribute("lower");
                joint.upper_limit = limit->DoubleAttribute("upper");
            }
            
            joints.push_back(joint);
        }
        
        return joints;
    }
};