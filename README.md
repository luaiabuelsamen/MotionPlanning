# Motion Planner

A C++ project for motion planning with forward/inverse kinematics and trajectory optimization.

## Dependencies

The planner itself is header-only and needs Eigen and tinyxml2; the gRPC
server additionally needs Protobuf and gRPC (it is skipped if they are not
found).

```bash
brew install pkg-config protobuf grpc tinyxml2 eigen         # macOS
sudo apt install pkg-config libtinyxml2-dev libeigen3-dev    # Linux (planner and tests)
```

## Building

1. Clone or download the repository
2. Navigate to the project root directory
3. Run the build command:

```bash
make all
```

This will:
- Create a `build/` directory
- Configure the project with CMake
- Compile all executables

## Running

After building, the executables will be in the `build/` directory:

- **Main executable**: `./build/motion_planner`
- **gRPC server**: `./build/motion_planner_server`
- **Tests**: `./build/test_motion_planner`

### Running Tests

```bash
cd build
./test_motion_planner
```

Or from the root:

```bash
make test
```

## API

The motion planner provides:
- Forward kinematics (FK) along the URDF tree from the root to the tip link
  (by default the leaf with the most movable joints, e.g. `tool0`);
  revolute, continuous and prismatic joints
- Inverse kinematics (IK): damped least squares with step limiting and joint
  limits; `computeIK(..., &converged)` reports whether it reached the target
- Joint-space trajectory optimization (MoveJ)
- Cartesian-space trajectory optimization (MoveL)

`Trajectory::success` is false when a plan cannot be trusted (goal outside
the joint limits, IK failure along a MoveL line).
- gRPC server interface for remote control

## Project Structure

- `src/`: Source code
- `proto/`: Protocol buffer definitions
- `tests/`: Unit tests
- `build/`: Build artifacts (generated)# MotionPlanning
