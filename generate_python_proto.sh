#!/bin/bash
# Generate Python gRPC files from proto and run tests

set -e  # Exit on any error

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# Generate Python files
echo "Generating Python gRPC code from proto/motion_planner.proto..."
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./proto \
    --grpc_python_out=./proto \
    ./proto/motion_planner.proto

echo "✓ Generated proto/motion_planner_pb2.py"
echo "✓ Generated proto/motion_planner_pb2_grpc.py"

# Start the C++ server in background
echo "Starting motion planner server..."
./build/motion_planner_server &
SERVER_PID=$!

# Wait a moment for server to start
sleep 2

# Run the Python test client
echo "Running Python test client..."
PYTHONPATH=$(pwd):$(pwd)/proto python tests/test_client.py ur5e localhost:50051

# Clean up
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true

echo "✓ Python tests completed successfully"
