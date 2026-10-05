#include "../src/robot.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

#ifndef MOTION_PLANNER_ASSETS
#define MOTION_PLANNER_ASSETS "../src/assets"
#endif

// Unlike assert, also checked in release builds.
#define REQUIRE(cond)                                                                       \
    do {                                                                                    \
        if (!(cond)) {                                                                      \
            std::ostringstream msg;                                                         \
            msg << __FILE__ << ":" << __LINE__ << ": REQUIRE failed: " << #cond;            \
            throw std::runtime_error(msg.str());                                            \
        }                                                                                   \
    } while (0)

const std::string& getUrdfPath() {
    static const std::string path = std::string(MOTION_PLANNER_ASSETS) + "/ur5e/ur5e.urdf";
    return path;
}

const std::vector<double> kHome = {0.0, -1.57, 1.57, -1.57, -1.57, 0.0};

std::vector<double> randomConfig(std::mt19937& rng, const std::vector<double>& center, double spread) {
    std::uniform_real_distribution<double> u(-spread, spread);
    std::vector<double> q = center;
    for (double& v : q) v += u(rng);
    return q;
}

double rotationDistance(const Quaternion& a, const Quaternion& b) {
    return orientationError(a, b).norm();
}

// A small URDF exercising what the UR5e does not: a sibling fixed joint
// listed after the arm, a joint without <origin>, a joint without <axis>
// (default x), a non-coordinate axis, a continuous and a prismatic joint.
std::string writeTestUrdf() {
    std::string path = "/tmp/motion_planner_test.urdf";
    std::ofstream f(path);
    f << R"(<robot name="t">
  <link name="base"/><link name="a"/><link name="b"/><link name="c"/><link name="tip"/><link name="side"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0.3" rpy="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/></joint>
  <joint name="j2" type="continuous"><parent link="a"/><child link="b"/>
    <origin xyz="0.2 0 0" rpy="0.3 0 0"/><axis xyz="0 1 1"/></joint>
  <joint name="j3" type="prismatic"><parent link="b"/><child link="c"/>
    <limit lower="0" upper="0.5" effort="10" velocity="1"/></joint>
  <joint name="tool" type="fixed"><parent link="c"/><child link="tip"/>
    <origin xyz="0 0.1 0" rpy="0 1.5707963 0"/></joint>
  <joint name="sibling" type="fixed"><parent link="base"/><child link="side"/>
    <origin xyz="1 2 3" rpy="0 0 3.14159265"/></joint>
</robot>
)";
    return path;
}

// The same chain as writeTestUrdf, composed by hand.
Transform testUrdfFK(const std::vector<double>& q) {
    auto Rz = [](double a) { return Eigen::AngleAxisd(a, Vector3::UnitZ()).toRotationMatrix(); };
    Transform T;
    T = T * Transform::fromRPY(0, 0, 0.5, 0, 0, 0.3);
    T.rotate(Rz(q[0]));
    T = T * Transform::fromRPY(0.3, 0, 0, 0.2, 0, 0);
    T.rotate(Eigen::AngleAxisd(q[1], Vector3(0, 1, 1).normalized()));
    T.translate(Vector3(q[2], 0, 0));  // prismatic, default axis x, no origin
    T = T * Transform::fromRPY(0, 1.5707963, 0, 0, 0.1, 0);
    return T;
}

// Test forward kinematics
void testFK() {
    std::cout << "Testing Forward Kinematics..." << std::endl;

    Robot robot(getUrdfPath());
    REQUIRE(robot.getDOF() == 6);

    // Home position (all zeros)
    std::vector<double> joint_angles_home = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto [pos_home, quat_home] = robot.getEndEffectorPose(joint_angles_home);

    std::cout << "  tool0 at zero: [" << pos_home.x() << ", " << pos_home.y() << ", " << pos_home.z() << "], "
              << "quaternion [" << quat_home.w() << ", " << quat_home.x() << ", " << quat_home.y() << ", "
              << quat_home.z() << "]" << std::endl;

    REQUIRE(std::abs(pos_home.x() - 0.8172) < 1e-3);
    REQUIRE(std::abs(pos_home.y() - 0.2329) < 1e-3);
    REQUIRE(std::abs(pos_home.z() - 0.0628) < 1e-3);
    // tool0 z (the tool axis) points along base_link +y at zero, x along -x.
    Eigen::Matrix3d R = quat_home.toRotationMatrix();
    REQUIRE((R.col(2) - Vector3(0, 1, 0)).norm() < 1e-6);
    REQUIRE((R.col(0) - Vector3(-1, 0, 0)).norm() < 1e-6);

    std::vector<double> joint_angles = {M_PI/4, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto [pos, quat] = robot.getEndEffectorPose(joint_angles);
    REQUIRE((pos - pos_home).norm() > 0.1);

    std::cout << "✓ FK test passed" << std::endl;
}

// FK follows the kinematic tree (not file order), handles missing origins /
// axes, general axes, continuous and prismatic joints.
void testURDFTree() {
    std::cout << "Testing URDF tree handling..." << std::endl;

    Robot robot(writeTestUrdf());
    REQUIRE(robot.getDOF() == 3);
    REQUIRE((robot.getJointNames() == std::vector<std::string>{"j1", "j2", "j3"}));

    const auto& cfg = robot.getOptimizerConfig();
    REQUIRE(cfg.joint_lower_limits.size() == 3);
    REQUIRE(cfg.joint_lower_limits[0] == -1 && cfg.joint_upper_limits[0] == 1);
    REQUIRE(std::isinf(cfg.joint_lower_limits[1]) && std::isinf(cfg.joint_upper_limits[1]));  // continuous
    REQUIRE(cfg.joint_lower_limits[2] == 0 && cfg.joint_upper_limits[2] == 0.5);

    std::mt19937 rng(1);
    double worst_pos = 0, worst_rot = 0;
    for (int i = 0; i < 100; ++i) {
        std::vector<double> q = randomConfig(rng, {0, 0, 0.25}, 0.9);
        auto [p, r] = robot.getEndEffectorPose(q);
        Transform expected = testUrdfFK(q);
        worst_pos = std::max(worst_pos, (p - expected.getPosition()).norm());
        worst_rot = std::max(worst_rot, rotationDistance(r, expected.getQuaternion()));
    }
    std::cout << "  vs hand-composed chain: " << worst_pos << " m, " << worst_rot << " rad" << std::endl;
    REQUIRE(worst_pos < 1e-12 && worst_rot < 1e-7);

    // An explicit tip link.
    Robot side(writeTestUrdf(), "side");
    REQUIRE(side.getDOF() == 0);
    REQUIRE((side.getEndEffectorPose({}).first - Vector3(1, 2, 3)).norm() < 1e-12);

    bool threw = false;
    try {
        Robot missing("/nonexistent.urdf");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    REQUIRE(threw);

    std::cout << "✓ URDF tree test passed" << std::endl;
}

// Jacobian against finite differences of FK (position and orientation).
void testJacobian() {
    std::cout << "Testing Jacobian..." << std::endl;

    for (const std::string& urdf : {getUrdfPath(), writeTestUrdf()}) {
        Robot robot(urdf);
        std::mt19937 rng(2);
        double worst = 0;
        for (int trial = 0; trial < 20; ++trial) {
            std::vector<double> q = randomConfig(rng, std::vector<double>(robot.getDOF(), 0.2), 1.0);
            Eigen::MatrixXd J = robot.getJacobian(q);
            REQUIRE(J.rows() == 6 && J.cols() == (int)robot.getDOF());
            const double h = 1e-6;
            for (size_t j = 0; j < robot.getDOF(); ++j) {
                std::vector<double> qp = q, qm = q;
                qp[j] += h;
                qm[j] -= h;
                auto [pp, rp] = robot.getEndEffectorPose(qp);
                auto [pm, rm] = robot.getEndEffectorPose(qm);
                Eigen::VectorXd col(6);
                col << (pp - pm) / (2 * h), orientationError(rp, rm) / (2 * h);
                worst = std::max(worst, (col - J.col(j)).norm());
            }
        }
        std::cout << "  " << robot.getDOF() << "-dof robot: max column error " << worst << std::endl;
        REQUIRE(worst < 1e-6);
    }

    std::cout << "✓ Jacobian test passed" << std::endl;
}

// Test inverse kinematics
void testIK() {
    std::cout << "Testing Inverse Kinematics..." << std::endl;

    Robot robot(getUrdfPath());

    // The original test: from all zeros.
    std::vector<double> joint_angles = {M_PI/4, 0.0, 0.0, 0.0, 0.0, 0.0};
    auto [target_pos, target_quat] = robot.getEndEffectorPose(joint_angles);
    bool converged = false;
    std::vector<double> ik_result = robot.computeIK(target_pos, target_quat, {0, 0, 0, 0, 0, 0}, &converged);
    auto [pos_verify, quat_verify] = robot.getEndEffectorPose(ik_result);
    REQUIRE(converged);
    REQUIRE((target_pos - pos_verify).norm() < 1e-3);
    REQUIRE(rotationDistance(target_quat, quat_verify) < 1e-2);

    // Random reachable targets from nearby seeds, including targets whose
    // orientation error starts beyond 180 degrees in quaternion terms. The
    // convergence flag must be truthful.
    std::mt19937 rng(3);
    int solved = 0, trials = 300;
    for (int i = 0; i < trials; ++i) {
        std::vector<double> q_goal = randomConfig(rng, kHome, 1.5);
        std::vector<double> seed = randomConfig(rng, q_goal, 0.4);
        auto [p, r] = robot.getEndEffectorPose(q_goal);
        bool ok = false;
        std::vector<double> q = robot.computeIK(p, r, seed, &ok);
        auto [p2, r2] = robot.getEndEffectorPose(q);
        bool reached = (p - p2).norm() < 1e-4 && rotationDistance(r, r2) < 1e-3;
        REQUIRE(ok == reached);
        solved += ok;
    }
    std::cout << "  random targets solved from a nearby seed: " << solved << " / " << trials << std::endl;
    REQUIRE(solved >= trials * 98 / 100);

    // Solutions stay within the configured joint limits.
    OptimizerConfig cfg = robot.getOptimizerConfig();
    cfg.joint_lower_limits[0] = 0.5;
    robot.setOptimizerConfig(cfg);
    std::vector<double> q = robot.computeIK(target_pos, target_quat, {0, 0, 0, 0, 0, 0});
    REQUIRE(q[0] >= 0.5);

    std::cout << "✓ IK test passed" << std::endl;
}

// URDF joint limits reach the optimizers (they used to be read after the
// optimizers copied the configuration, and were ignored).
void testJointLimits() {
    std::cout << "Testing joint limits..." << std::endl;

    Robot robot(getUrdfPath());
    const auto& cfg = robot.getOptimizerConfig();
    REQUIRE(cfg.joint_lower_limits.size() == 6 && cfg.joint_upper_limits.size() == 6);
    REQUIRE(std::abs(cfg.joint_upper_limits[0] - 2 * M_PI) < 1e-6);

    std::vector<double> goal = kHome;
    goal[0] = 7.0;  // beyond 2 pi
    Trajectory traj = robot.moveJ(kHome, goal, 30);
    REQUIRE(!traj.success);
    REQUIRE(traj.points.back().position[0] <= 2 * M_PI + 1e-9);
    std::mt19937 rng(4);
    REQUIRE(robot.moveJ(kHome, randomConfig(rng, kHome, 0.5), 30).success);

    std::cout << "✓ Joint limits test passed" << std::endl;
}

// Test MoveJ trajectory optimization
void testMoveJ() {
    std::cout << "Testing MoveJ Trajectory Optimization..." << std::endl;

    Robot robot(getUrdfPath());

    std::vector<double> start_config = {0.0, -M_PI/4, M_PI/2, -M_PI/4, -M_PI/2, 0.0};
    std::vector<double> goal_config = {M_PI/4, -M_PI/6, M_PI/3, -M_PI/3, -M_PI/2, M_PI/6};

    Trajectory traj = robot.moveJ(start_config, goal_config, 50);

    REQUIRE(traj.size() == 50);
    REQUIRE(traj.dof == 6);
    REQUIRE(traj.success);

    for (size_t j = 0; j < 6; ++j) {
        REQUIRE(std::abs(traj.points[0].position[j] - start_config[j]) < 1e-6);
        REQUIRE(std::abs(traj.points.back().position[j] - goal_config[j]) < 1e-6);
    }

    // Velocities start and end near zero
    double start_vel_norm = 0.0;
    double end_vel_norm = 0.0;
    for (size_t j = 0; j < 6; ++j) {
        start_vel_norm += traj.points[0].velocity[j] * traj.points[0].velocity[j];
        end_vel_norm += traj.points.back().velocity[j] * traj.points.back().velocity[j];
    }
    REQUIRE(sqrt(start_vel_norm) < 0.1);
    REQUIRE(sqrt(end_vel_norm) < 0.1);

    std::cout << "  Duration: " << TrajectoryUtils::getTrajectoryduration(traj) << " seconds" << std::endl;
    std::cout << "✓ MoveJ test passed" << std::endl;
}

// Largest change of any joint between consecutive waypoints.
double largestStep(const Trajectory& t) {
    double big = 0;
    for (size_t i = 1; i < t.size(); ++i)
        for (size_t j = 0; j < t.dof; ++j)
            big = std::max(big, std::abs(t.points[i].position[j] - t.points[i - 1].position[j]));
    return big;
}

// Largest distance of the tool from the straight start-goal segment.
double pathDeviation(Robot& robot, const Trajectory& t) {
    Vector3 a = robot.getEndEffectorPose(t.points.front().position).first;
    Vector3 b = robot.getEndEffectorPose(t.points.back().position).first;
    double worst = 0;
    for (const auto& point : t.points) {
        Vector3 p = robot.getEndEffectorPose(point.position).first;
        double s = (b - a).squaredNorm() > 0 ? std::clamp((p - a).dot(b - a) / (b - a).squaredNorm(), 0.0, 1.0) : 0.0;
        worst = std::max(worst, (p - (a + s * (b - a))).norm());
    }
    return worst;
}

// Test MoveL trajectory optimization
void testMoveL() {
    std::cout << "Testing MoveL Trajectory Optimization..." << std::endl;

    Robot robot(getUrdfPath());

    std::vector<double> start_config = {0.0, -M_PI/4, M_PI/2, -M_PI/4, -M_PI/2, 0.0};
    std::vector<double> goal_config = {M_PI/6, -M_PI/3, M_PI/3, -M_PI/4, -M_PI/2, M_PI/6};

    Trajectory traj = robot.moveL(start_config, goal_config, 50);

    REQUIRE(traj.size() == 50);
    REQUIRE(traj.dof == 6);
    REQUIRE(traj.success);
    for (size_t j = 0; j < 6; ++j) {
        REQUIRE(std::abs(traj.points[0].position[j] - start_config[j]) < 1e-6);
    }

    auto goal = robot.getEndEffectorPose(goal_config);
    auto end = robot.getEndEffectorPose(traj.points.back().position);
    double deviation = pathDeviation(robot, traj);
    std::cout << "  goal error " << (end.first - goal.first).norm() << " m, path off the straight line by at most "
              << deviation << " m" << std::endl;
    REQUIRE((end.first - goal.first).norm() < 1e-3);
    REQUIRE(deviation < 0.01);  // smoothing after IK rounds the line slightly

    std::cout << "  Duration: " << TrajectoryUtils::getTrajectoryduration(traj) << " seconds" << std::endl;
    std::cout << "✓ MoveL test passed" << std::endl;
}

// Many random MoveL goals: a trajectory reported successful must reach its
// goal without joint jumps (IK used to diverge silently on ~1% of these).
void testMoveLRandom() {
    std::cout << "Testing MoveL on random goals..." << std::endl;

    Robot robot(getUrdfPath());
    std::mt19937 rng(7);
    int failed = 0, trials = 500;
    double worst_step = 0, worst_goal = 0;
    for (int i = 0; i < trials; ++i) {
        auto [pos, quat] = robot.getEndEffectorPose(randomConfig(rng, kHome, 0.6));
        Trajectory traj = robot.moveL(kHome, pos, quat, 60);
        if (!traj.success) {
            failed++;
            continue;
        }
        auto end = robot.getEndEffectorPose(traj.points.back().position);
        worst_step = std::max(worst_step, largestStep(traj));
        worst_goal = std::max(worst_goal, (end.first - pos).norm());
    }
    std::cout << "  " << trials - failed << " / " << trials << " succeeded; on those: largest joint step "
              << worst_step << " rad, worst goal error " << worst_goal << " m" << std::endl;
    REQUIRE(worst_step < 0.3);
    REQUIRE(worst_goal < 1e-3);
    REQUIRE(failed <= trials / 100);

    std::cout << "✓ MoveL random test passed" << std::endl;
}

// Test MoveL with explicit Cartesian target
void testMoveLCartesian() {
    std::cout << "Testing MoveL with Cartesian Target..." << std::endl;

    Robot robot(getUrdfPath());

    std::vector<double> start_config = {0.0, -M_PI/4, M_PI/2, -M_PI/4, -M_PI/2, 0.0};

    // Move 10cm in X direction
    auto [current_pos, current_quat] = robot.getEndEffectorPose(start_config);
    Vector3 target_pos = current_pos + Vector3(0.1, 0.0, 0.0);

    Trajectory traj = robot.moveL(start_config, target_pos, current_quat, 50);

    REQUIRE(traj.size() == 50);
    REQUIRE(traj.success);

    auto [final_pos, final_quat] = robot.getEndEffectorPose(traj.points.back().position);
    double pos_error = (final_pos - target_pos).norm();

    std::cout << "  Position error: " << pos_error << " m" << std::endl;
    REQUIRE(pos_error < 1e-3);
    REQUIRE(rotationDistance(final_quat, current_quat) < 1e-2);

    std::cout << "✓ MoveL Cartesian test passed" << std::endl;
}

// Test trajectory interpolation utilities
void testTrajectoryUtils() {
    std::cout << "Testing Trajectory Utilities..." << std::endl;

    Robot robot(getUrdfPath());

    std::vector<double> start_config = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> goal_config = {M_PI/4, 0.0, 0.0, 0.0, 0.0, 0.0};

    Trajectory traj = robot.moveJ(start_config, goal_config, 20);
    double duration = TrajectoryUtils::getTrajectoryduration(traj);

    // Resampling keeps the duration and the endpoints.
    for (size_t n : {10, 50}) {
        for (bool spline : {false, true}) {
            Trajectory r = spline ? TrajectoryUtils::cubicSplineInterpolate(traj, n)
                                  : TrajectoryUtils::linearInterpolate(traj, n);
            REQUIRE(r.size() == n);
            REQUIRE(std::abs(TrajectoryUtils::getTrajectoryduration(r) - duration) < 1e-12);
            REQUIRE(std::abs(r.points.front().position[0] - 0.0) < 1e-12);
            REQUIRE(std::abs(r.points.back().position[0] - M_PI / 4) < 1e-12);
            // Positions agree with the original at the same time.
            double t = 0.37 * duration;
            double a = TrajectoryUtils::interpolateAtTime(r, t)[0];
            double b = TrajectoryUtils::interpolateAtTime(traj, t)[0];
            REQUIRE(std::abs(a - b) < 0.02);
        }
    }

    // Time scaling
    std::vector<double> max_vel(6, 1.0);
    std::vector<double> max_acc(6, 5.0);
    Trajectory scaled = TrajectoryUtils::scaleTrajectoryTime(traj, max_vel, max_acc);
    REQUIRE(TrajectoryUtils::checkVelocityLimits(scaled, max_vel));
    REQUIRE(TrajectoryUtils::checkAccelerationLimits(scaled, max_acc));

    double mid_time = duration / 2.0;
    std::vector<double> mid_config = TrajectoryUtils::interpolateAtTime(traj, mid_time);
    REQUIRE(mid_config.size() == 6);

    std::cout << "✓ Trajectory utilities test passed" << std::endl;
}

int main() {
    std::cout << "Running motion planner tests..." << std::endl;

    try {
        testFK();
        testURDFTree();
        testJacobian();
        testIK();
        testJointLimits();
        testMoveJ();
        testMoveL();
        testMoveLRandom();
        testMoveLCartesian();
        testTrajectoryUtils();

        std::cout << "\n✓ All tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "\n✗ Test failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cout << "\n✗ Test failed with unknown error" << std::endl;
        return 1;
    }

    return 0;
}
