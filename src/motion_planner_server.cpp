#include <grpcpp/grpcpp.h>
#include <grpcpp/alarm.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>
#include "motion_planner.grpc.pb.h"
#include "../src/robot.hpp"
#include <memory>
#include <iostream>
#include <csignal>
#include <atomic>
#include <filesystem>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerAsyncResponseWriter;
using grpc::CompletionQueue;
using grpc::ServerCompletionQueue;
using grpc::Status;
using motionplanner::MotionPlanner;
using motionplanner::InitRobotRequest;
using motionplanner::InitRobotResponse;
using motionplanner::MoveLRequest;
using motionplanner::MoveJRequest;
using motionplanner::TrajectoryResponse;
using motionplanner::FKRequest;
using motionplanner::FKResponse;
using motionplanner::IKRequest;
using motionplanner::IKResponse;

namespace fs = std::filesystem;

std::atomic<bool> g_terminate(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
        g_terminate = true;
    }
}

class MotionPlannerServiceImpl final {
public:
    MotionPlannerServiceImpl() : robot_(nullptr) {}
    
    ~MotionPlannerServiceImpl() {
        // Server and completion queue are already shutdown in HandleRpcs()
        // Just clean up here
    }

    void Run(const std::string& server_address) {
        ServerBuilder builder;
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
        builder.RegisterService(&service_);
        cq_ = builder.AddCompletionQueue();
        server_ = builder.BuildAndStart();
        std::cout << "Motion Planner gRPC Server listening on " << server_address << std::endl;
        std::cout << "Waiting for InitRobot call to load robot..." << std::endl;
        HandleRpcs();
    }

private:
    class CallDataBase {
    public:
        virtual void Proceed(bool ok) = 0;
        virtual ~CallDataBase() = default;
    };

    // InitRobot Call Handler
    class InitRobotCallData : public CallDataBase {
    public:
        InitRobotCallData(MotionPlanner::AsyncService* service, ServerCompletionQueue* cq, 
                         std::unique_ptr<Robot>& robot)
            : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE), robot_(robot) {
            Proceed(true);
        }
        
        void Proceed(bool ok) override {
            if (status_ == CREATE) {
                if (!ok  ) {
                    delete this;
                    return;
                }
                status_ = PROCESS;
                service_->RequestInitRobot(&ctx_, &request_, &responder_, cq_, cq_, this);
            } else if (status_ == PROCESS) {
                if (!ok  ) {
                    delete this;
                    return;
                }
                // Only spawn new request handler if not shutting down
                if (!g_terminate) {
                    new InitRobotCallData(service_, cq_, robot_);
                }
                
                std::string robot_type = request_.robot_type();
                std::cout << "[InitRobot] Loading robot: " << robot_type << std::endl;
                
                // Construct URDF path: src/assets/<robot_type>/<robot_type>.urdf
                fs::path urdf_path = fs::path("src") / "assets" / robot_type / (robot_type + ".urdf");
                
                if (!fs::exists(urdf_path)) {
                    response_.set_success(false);
                    response_.set_message("URDF not found: " + urdf_path.string());
                    response_.set_dof(0);
                    std::cerr << "[InitRobot] Failed: " << response_.message() << std::endl;
                } else {
                    try {
                        robot_ = std::make_unique<Robot>(urdf_path.string());
                        int dof = robot_->getDOF();
                        
                        if (dof > 0) {
                            response_.set_success(true);
                            response_.set_message("Robot '" + robot_type + "' loaded successfully");
                            response_.set_dof(dof);
                            std::cout << "[InitRobot] Success: " << dof << " DOF robot loaded" << std::endl;
                        } else {
                            robot_ = nullptr;
                            response_.set_success(false);
                            response_.set_message("Failed to parse robot from URDF");
                            response_.set_dof(0);
                            std::cerr << "[InitRobot] Failed: Invalid robot" << std::endl;
                        }
                    } catch (const std::exception& e) {
                        robot_ = nullptr;
                        response_.set_success(false);
                        response_.set_message("Exception loading robot: " + std::string(e.what()));
                        response_.set_dof(0);
                        std::cerr << "[InitRobot] Exception: " << e.what() << std::endl;
                    }
                }
                
                status_ = FINISH;
                responder_.Finish(response_, Status::OK, this);
            } else {
                delete this;
            }
        }
        
    private:
        MotionPlanner::AsyncService* service_;
        ServerCompletionQueue* cq_;
        ServerContext ctx_;
        InitRobotRequest request_;
        InitRobotResponse response_;
        ServerAsyncResponseWriter<InitRobotResponse> responder_;
        enum CallStatus { CREATE, PROCESS, FINISH } status_;
        std::unique_ptr<Robot>& robot_;
    };

    // MoveL Call Handler
    class MoveLCallData : public CallDataBase {
    public:
        MoveLCallData(MotionPlanner::AsyncService* service, ServerCompletionQueue* cq,
                     std::unique_ptr<Robot>& robot)
            : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE), robot_(robot) {
            Proceed(true);
        }
        
        void Proceed(bool ok) override {
            if (status_ == CREATE) {
                if (!ok  ) {
                    delete this;
                    return;
                }
                status_ = PROCESS;
                service_->RequestMoveL(&ctx_, &request_, &responder_, cq_, cq_, this);
            } else if (status_ == PROCESS) {
                if (!ok  ) {
                    delete this;
                    return;
                }
                // Only spawn new request handler if not shutting down
                if (!g_terminate) {
                    new MoveLCallData(service_, cq_, robot_);
                }
                
                if (!robot_) {
                    response_.set_success(false);
                    response_.set_message("Robot not initialized. Call InitRobot first.");
                    std::cerr << "[MoveL] Error: Robot not initialized" << std::endl;
                } else {
                    auto& pose_msg = request_.target_pose();
                    auto& start_msg = request_.start_joints();
                    bool fast_mode = request_.fast_mode();
                    
                    std::cout << "[MoveL] Request - Target: (" << pose_msg.x() << ", " << pose_msg.y() 
                              << ", " << pose_msg.z() << "), Fast mode: " << fast_mode << std::endl;
                    
                    // Convert proto to Eigen types
                    Eigen::Vector3d position(pose_msg.x(), pose_msg.y(), pose_msg.z());
                    Eigen::Quaterniond orientation(pose_msg.qw(), pose_msg.qx(), pose_msg.qy(), pose_msg.qz());
                    
                    Eigen::VectorXd start_joints(start_msg.values_size());
                    for (int i = 0; i < start_msg.values_size(); ++i) {
                        start_joints(i) = start_msg.values(i);
                    }
                    
                    std::vector<double> start_joints_vec(start_joints.data(), start_joints.data() + start_joints.size());
                    
                    try {
                        auto result = robot_->moveL(start_joints_vec, position, orientation, 50);
                        
                        response_.set_success(true);
                        response_.set_message("MoveL trajectory generated");
                        response_.set_duration(result.points.size() * result.dt);
                        response_.set_num_waypoints(result.points.size());
                        
                        auto traj_matrix = result.getPositionMatrix();
                        for (int i = 0; i < traj_matrix.rows(); ++i) {
                            auto* waypoint = response_.add_trajectory();
                            for (int j = 0; j < traj_matrix.cols(); ++j) {
                                waypoint->add_values(traj_matrix(i, j));
                            }
                        }
                        std::cout << "[MoveL] Success: " << result.points.size() 
                                  << " waypoints, duration: " << result.points.size() * result.dt << "s" << std::endl;
                    } catch (const std::exception& e) {
                        response_.set_success(false);
                        response_.set_message("Exception in MoveL: " + std::string(e.what()));
                        std::cerr << "[MoveL] Exception: " << e.what() << std::endl;
                    }
                }
                
                status_ = FINISH;
                responder_.Finish(response_, Status::OK, this);
            } else {
                delete this;
            }
        }
        
    private:
        MotionPlanner::AsyncService* service_;
        ServerCompletionQueue* cq_;
        ServerContext ctx_;
        MoveLRequest request_;
        TrajectoryResponse response_;
        ServerAsyncResponseWriter<TrajectoryResponse> responder_;
        enum CallStatus { CREATE, PROCESS, FINISH } status_;
        std::unique_ptr<Robot>& robot_;
    };

    // MoveJ Call Handler
    class MoveJCallData : public CallDataBase {
    public:
        MoveJCallData(MotionPlanner::AsyncService* service, ServerCompletionQueue* cq,
                     std::unique_ptr<Robot>& robot)
            : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE), robot_(robot) {
            Proceed(true);
        }
        
        void Proceed(bool ok) override {
            if (status_ == CREATE) {
                if (!ok  ) {
                    delete this;
                    return;
                }
                status_ = PROCESS;
                service_->RequestMoveJ(&ctx_, &request_, &responder_, cq_, cq_, this);
            } else if (status_ == PROCESS) {
                if (!ok  ) {
                    delete this;
                    return;
                }
                // Only spawn new request handler if not shutting down
                if (!g_terminate) {
                    new MoveJCallData(service_, cq_, robot_);
                }
                
                if (!robot_) {
                    response_.set_success(false);
                    response_.set_message("Robot not initialized. Call InitRobot first.");
                    std::cerr << "[MoveJ] Error: Robot not initialized" << std::endl;
                } else {
                    auto& target_msg = request_.target_joints();
                    auto& start_msg = request_.start_joints();
                    
                    std::cout << "[MoveJ] Request - " << target_msg.values_size() << " target joints" << std::endl;
                    
                    Eigen::VectorXd target_joints(target_msg.values_size());
                    for (int i = 0; i < target_msg.values_size(); ++i) {
                        target_joints(i) = target_msg.values(i);
                    }
                    
                    Eigen::VectorXd start_joints(start_msg.values_size());
                    for (int i = 0; i < start_msg.values_size(); ++i) {
                        start_joints(i) = start_msg.values(i);
                    }
                    
                    std::vector<double> start_joints_vec(start_joints.data(), start_joints.data() + start_joints.size());
                    std::vector<double> target_joints_vec(target_joints.data(), target_joints.data() + target_joints.size());
                    
                    try {
                        auto result = robot_->moveJ(start_joints_vec, target_joints_vec, 50);
                        
                        response_.set_success(true);
                        response_.set_message("MoveJ trajectory generated");
                        response_.set_duration(result.points.size() * result.dt);
                        response_.set_num_waypoints(result.points.size());
                        
                        auto traj_matrix = result.getPositionMatrix();
                        for (int i = 0; i < traj_matrix.rows(); ++i) {
                            auto* waypoint = response_.add_trajectory();
                            for (int j = 0; j < traj_matrix.cols(); ++j) {
                                waypoint->add_values(traj_matrix(i, j));
                            }
                        }
                        std::cout << "[MoveJ] Success: " << result.points.size() 
                                  << " waypoints, duration: " << result.points.size() * result.dt << "s" << std::endl;
                    } catch (const std::exception& e) {
                        response_.set_success(false);
                        response_.set_message("Exception in MoveJ: " + std::string(e.what()));
                        std::cerr << "[MoveJ] Exception: " << e.what() << std::endl;
                    }
                }
                
                status_ = FINISH;
                responder_.Finish(response_, Status::OK, this);
            } else {
                delete this;
            }
        }
        
    private:
        MotionPlanner::AsyncService* service_;
        ServerCompletionQueue* cq_;
        ServerContext ctx_;
        MoveJRequest request_;
        TrajectoryResponse response_;
        ServerAsyncResponseWriter<TrajectoryResponse> responder_;
        enum CallStatus { CREATE, PROCESS, FINISH } status_;
        std::unique_ptr<Robot>& robot_;
    };

    // FK Call Handler
    class FKCallData : public CallDataBase {
    public:
        FKCallData(MotionPlanner::AsyncService* service, ServerCompletionQueue* cq,
                  std::unique_ptr<Robot>& robot)
            : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE), robot_(robot) {
            Proceed(true);
        }
        
        void Proceed(bool ok) override {
            if (status_ == CREATE) {
                if (!ok  ) {
                    delete this;
                    return;
                }
                status_ = PROCESS;
                service_->RequestComputeFK(&ctx_, &request_, &responder_, cq_, cq_, this);
            } else if (status_ == PROCESS) {
                if (!ok  ) {
                    delete this;
                    return;
                }
                // Only spawn new request handler if not shutting down
                if (!g_terminate) {
                    new FKCallData(service_, cq_, robot_);
                }
                
                if (!robot_) {
                    response_.set_success(false);
                    response_.set_message("Robot not initialized. Call InitRobot first.");
                } else {
                    auto& joint_msg = request_.joint_values();
                    
                    Eigen::VectorXd joints(joint_msg.values_size());
                    for (int i = 0; i < joint_msg.values_size(); ++i) {
                        joints(i) = joint_msg.values(i);
                    }
                    
                    std::vector<double> joints_vec(joints.data(), joints.data() + joints.size());
                    
                    try {
                        auto [position, orientation] = robot_->getEndEffectorPose(joints_vec);
                        
                        response_.set_success(true);
                        response_.set_message("FK computed successfully");
                        
                        auto* pose = response_.mutable_end_effector_pose();
                        pose->set_x(position.x());
                        pose->set_y(position.y());
                        pose->set_z(position.z());
                        pose->set_qx(orientation.x());
                        pose->set_qy(orientation.y());
                        pose->set_qz(orientation.z());
                        pose->set_qw(orientation.w());
                        
                        std::cout << "[FK] Success: Position (" << position.transpose() << ")" << std::endl;
                    } catch (const std::exception& e) {
                        response_.set_success(false);
                        response_.set_message("Exception in FK: " + std::string(e.what()));
                        std::cerr << "[FK] Exception: " << e.what() << std::endl;
                    }
                }
                
                status_ = FINISH;
                responder_.Finish(response_, Status::OK, this);
            } else {
                delete this;
            }
        }
        
    private:
        MotionPlanner::AsyncService* service_;
        ServerCompletionQueue* cq_;
        ServerContext ctx_;
        FKRequest request_;
        FKResponse response_;
        ServerAsyncResponseWriter<FKResponse> responder_;
        enum CallStatus { CREATE, PROCESS, FINISH } status_;
        std::unique_ptr<Robot>& robot_;
    };

    // IK Call Handler
    class IKCallData : public CallDataBase {
    public:
        IKCallData(MotionPlanner::AsyncService* service, ServerCompletionQueue* cq,
                  std::unique_ptr<Robot>& robot)
            : service_(service), cq_(cq), responder_(&ctx_), status_(CREATE), robot_(robot) {
            Proceed(true);
        }
        
        void Proceed(bool ok) override {
            if (status_ == CREATE) {
                if (!ok  ) {
                    delete this;
                    return;
                }
                status_ = PROCESS;
                service_->RequestComputeIK(&ctx_, &request_, &responder_, cq_, cq_, this);
            } else if (status_ == PROCESS) {
                if (!ok  ) {
                    delete this;
                    return;
                }
                // Only spawn new request handler if not shutting down
                if (!g_terminate) {
                    new IKCallData(service_, cq_, robot_);
                }
                
                if (!robot_) {
                    response_.set_success(false);
                    response_.set_message("Robot not initialized. Call InitRobot first.");
                } else {
                    auto& pose_msg = request_.target_pose();
                    auto& seed_msg = request_.seed();
                    double tolerance = request_.tolerance() > 0 ? request_.tolerance() : 1e-4;
                    int max_iterations = request_.max_iterations() > 0 ? request_.max_iterations() : 100;
                    
                    Eigen::Vector3d position(pose_msg.x(), pose_msg.y(), pose_msg.z());
                    Eigen::Quaterniond orientation(pose_msg.qw(), pose_msg.qx(), pose_msg.qy(), pose_msg.qz());
                    
                    Eigen::VectorXd seed(seed_msg.values_size());
                    for (int i = 0; i < seed_msg.values_size(); ++i) {
                        seed(i) = seed_msg.values(i);
                    }
                    
                    std::vector<double> seed_vec(seed.data(), seed.data() + seed.size());
                    
                    try {
                        auto solution = robot_->computeIK(position, orientation, seed_vec);
                        
                        response_.set_success(true);
                        response_.set_message("IK solution found");
                        
                        auto* joint_solution = response_.mutable_joint_solution();
                        for (double val : solution) {
                            joint_solution->add_values(val);
                        }
                        std::cout << "[IK] Success: Found solution" << std::endl;
                    } catch (const std::exception& e) {
                        response_.set_success(false);
                        response_.set_message("Exception in IK: " + std::string(e.what()));
                        std::cerr << "[IK] Exception: " << e.what() << std::endl;
                    }
                }
                
                status_ = FINISH;
                responder_.Finish(response_, Status::OK, this);
            } else {
                delete this;
            }
        }
        
    private:
        MotionPlanner::AsyncService* service_;
        ServerCompletionQueue* cq_;
        ServerContext ctx_;
        IKRequest request_;
        IKResponse response_;
        ServerAsyncResponseWriter<IKResponse> responder_;
        enum CallStatus { CREATE, PROCESS, FINISH } status_;
        std::unique_ptr<Robot>& robot_;
    };

    void HandleRpcs() {
        // Spawn instances for each RPC type
        new InitRobotCallData(&service_, cq_.get(), robot_);
        new MoveLCallData(&service_, cq_.get(), robot_);
        new MoveJCallData(&service_, cq_.get(), robot_);
        new FKCallData(&service_, cq_.get(), robot_);
        new IKCallData(&service_, cq_.get(), robot_);
        
        void* tag;
        bool ok;
        while (true) {
            // Set a timeout to check g_terminate periodically
            auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(100);
            auto status = cq_->AsyncNext(&tag, &ok, deadline);
            
            if (status == grpc::CompletionQueue::GOT_EVENT) {
                static_cast<CallDataBase*>(tag)->Proceed(ok);
            } else if (status == grpc::CompletionQueue::TIMEOUT) {
                if (g_terminate) {
                    std::cout << "Shutting down server..." << std::endl;
                    // First shutdown the server to stop accepting new requests
                    server_->Shutdown();
                    std::cout << "Shutting down completion queue..." << std::endl;
                    // Then shutdown the completion queue
                    cq_->Shutdown();
                    break;
                }
                continue;
            } else if (status == grpc::CompletionQueue::SHUTDOWN) {
                std::cout << "Completion queue shut down" << std::endl;
                break;
            }
        }
        
        // Drain all remaining events from the completion queue
        // After Shutdown(), all pending operations will complete with ok=false
        int drained = 0;
        while (cq_->Next(&tag, &ok)) {
            // Process remaining events by calling Proceed(false) which will clean them up
            static_cast<CallDataBase*>(tag)->Proceed(ok);
            drained++;
        }
        std::cout << "Drained " << drained << " pending operations" << std::endl;
    }

    std::unique_ptr<ServerCompletionQueue> cq_;
    MotionPlanner::AsyncService service_;
    std::unique_ptr<Server> server_;
    std::unique_ptr<Robot> robot_;
};

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    std::string server_address = "0.0.0.0:50051";
    if (argc > 1) {
        server_address = argv[1];
    }
    
    MotionPlannerServiceImpl service;
    service.Run(server_address);
    
    return 0;
}
