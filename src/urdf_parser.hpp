// urdf_parser.hpp
#pragma once
#include "transform.hpp"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <tinyxml2.h>  // apt install libtinyxml2-dev / brew install tinyxml2

struct Joint {
    std::string name;
    std::string type;
    std::string parent;  // parent link
    std::string child;   // child link
    Vector3 origin_xyz = Vector3::Zero();
    Vector3 origin_rpy = Vector3::Zero();
    Vector3 axis = Vector3::UnitX();  // URDF default
    double lower_limit = -std::numeric_limits<double>::infinity();
    double upper_limit = std::numeric_limits<double>::infinity();

    bool isRevolute() const { return type == "revolute" || type == "continuous"; }
    bool isPrismatic() const { return type == "prismatic"; }
    bool isMovable() const { return isRevolute() || isPrismatic(); }
};

class URDFParser {
public:
    // All joints, in file order. Throws std::runtime_error if the file cannot
    // be read.
    static std::vector<Joint> parseJoints(const std::string& urdf_file) {
        std::vector<Joint> joints;
        tinyxml2::XMLDocument doc;

        if (doc.LoadFile(urdf_file.c_str()) != tinyxml2::XML_SUCCESS) {
            throw std::runtime_error("Failed to load URDF file " + urdf_file + ": " + doc.ErrorStr());
        }

        auto* robot = doc.FirstChildElement("robot");
        if (!robot) {
            throw std::runtime_error("No <robot> element in URDF " + urdf_file);
        }

        for (auto* joint_elem = robot->FirstChildElement("joint");
             joint_elem != nullptr;
             joint_elem = joint_elem->NextSiblingElement("joint")) {

            Joint joint;
            joint.name = attribute(joint_elem, "name");
            joint.type = attribute(joint_elem, "type");
            if (auto* parent = joint_elem->FirstChildElement("parent")) joint.parent = attribute(parent, "link");
            if (auto* child = joint_elem->FirstChildElement("child")) joint.child = attribute(child, "link");

            if (auto* origin = joint_elem->FirstChildElement("origin")) {
                readVector(origin, "xyz", joint.origin_xyz);
                readVector(origin, "rpy", joint.origin_rpy);
            }

            if (auto* axis = joint_elem->FirstChildElement("axis")) {
                readVector(axis, "xyz", joint.axis);
            }
            if (joint.axis.norm() > 0) joint.axis.normalize();

            // Continuous joints have no position limits, even if a <limit>
            // tag gives effort / velocity.
            if (auto* limit = joint_elem->FirstChildElement("limit")) {
                if (joint.type == "revolute" || joint.type == "prismatic") {
                    joint.lower_limit = limit->DoubleAttribute("lower", joint.lower_limit);
                    joint.upper_limit = limit->DoubleAttribute("upper", joint.upper_limit);
                }
            }

            joints.push_back(joint);
        }

        return joints;
    }

    // The joints from the root link to `tip_link`, in kinematic order. With
    // an empty tip, picks the leaf link whose chain has the most movable
    // joints (ties: the longest chain, then file order) - the end effector of
    // a serial arm. Throws if the joints do not form a tree.
    static std::vector<Joint> parseChain(const std::string& urdf_file, const std::string& tip_link = "") {
        std::vector<Joint> all = parseJoints(urdf_file);
        std::map<std::string, size_t> joint_of_child;
        for (size_t i = 0; i < all.size(); ++i) {
            if (all[i].parent.empty() || all[i].child.empty()) {
                throw std::runtime_error("Joint '" + all[i].name + "' has no parent or child link");
            }
            if (!joint_of_child.emplace(all[i].child, i).second) {
                throw std::runtime_error("Link '" + all[i].child + "' has more than one parent joint");
            }
        }

        // Joints from the root to `link`, root first.
        auto chainTo = [&](const std::string& link) {
            std::vector<Joint> chain;
            std::string current = link;
            while (joint_of_child.count(current)) {
                const Joint& j = all[joint_of_child.at(current)];
                chain.push_back(j);
                current = j.parent;
                if (chain.size() > all.size()) throw std::runtime_error("URDF joints form a cycle");
            }
            std::reverse(chain.begin(), chain.end());
            return chain;
        };

        if (!tip_link.empty()) {
            if (!joint_of_child.count(tip_link)) {
                throw std::runtime_error("Tip link '" + tip_link + "' is not the child of any joint");
            }
            return chainTo(tip_link);
        }

        std::vector<Joint> best;
        size_t best_movable = 0;
        for (const Joint& j : all) {
            bool leaf = std::none_of(all.begin(), all.end(), [&](const Joint& k) { return k.parent == j.child; });
            if (!leaf) continue;
            std::vector<Joint> chain = chainTo(j.child);
            size_t movable = std::count_if(chain.begin(), chain.end(), [](const Joint& k) { return k.isMovable(); });
            if (best.empty() || movable > best_movable || (movable == best_movable && chain.size() > best.size())) {
                best = chain;
                best_movable = movable;
            }
        }
        return best;
    }

private:
    static std::string attribute(const tinyxml2::XMLElement* elem, const char* name) {
        const char* value = elem->Attribute(name);
        if (!value) {
            throw std::runtime_error(std::string("URDF <") + elem->Name() + "> is missing attribute '" + name + "'");
        }
        return value;
    }

    static void readVector(const tinyxml2::XMLElement* elem, const char* name, Vector3& out) {
        if (const char* value = elem->Attribute(name)) {
            double x, y, z;
            if (std::sscanf(value, "%lf %lf %lf", &x, &y, &z) == 3) out = Vector3(x, y, z);
        }
    }
};
