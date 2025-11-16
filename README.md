# Motion Planner

A C++ project for motion planning with forward/inverse kinematics and trajectory optimization.

## Dependencies

This project requires the following libraries (install via Homebrew on macOS):

```bash
brew install pkg-config protobuf grpc tinyxml2 eigen
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
- Forward kinematics (FK)
- Inverse kinematics (IK)
- Joint-space trajectory optimization (MoveJ)
- Cartesian-space trajectory optimization (MoveL)
- gRPC server interface for remote control

## Project Structure

- `src/`: Source code
- `proto/`: Protocol buffer definitions
- `tests/`: Unit tests
- `build/`: Build artifacts (generated)# MotionPlanning
