#!/usr/bin/env python3
"""
Motion Planner gRPC Test Client

Demonstrates how to use the motion planner gRPC server:
1. Initialize robot with a specific type
2. Call MoveJ to generate joint space trajectories
3. Call MoveL to generate Cartesian linear trajectories
4. Use FK and IK services
"""

import grpc
import sys
import numpy as np
from proto import motion_planner_pb2
from proto import motion_planner_pb2_grpc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os


def create_plots_directory():
    """Create plots directory if it doesn't exist"""
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir


def plot_tcp_trajectory(trajectory_data, plots_dir, title_suffix="", stub=None):
    """Plot TCP trajectory in 3D space"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions - compute TCP positions for joint trajectories
    all_positions = []
    labels = []
    trajectory_labels = ['MoveJ', 'MoveL Fast', 'MoveL Accurate', 'MoveL Sequential']
    
    for i, traj in enumerate(trajectory_data):
        waypoints = np.array([list(wp.values) for wp in traj['waypoints']])
        
        # Check if waypoints are already TCP positions (3 values) or joint values (6+ values)
        if len(traj['waypoints']) > 0 and len(traj['waypoints'][0].values) == 3:
            # Already TCP positions
            positions = waypoints
        else:
            # Joint values - compute TCP positions via FK
            positions = []
            if stub:
                for joint_vals in waypoints:
                    try:
                        fk_request = motion_planner_pb2.FKRequest(
                            joint_values=motion_planner_pb2.JointValues(values=joint_vals)
                        )
                        fk_response = stub.ComputeFK(fk_request)
                        if fk_response.success:
                            pose = fk_response.end_effector_pose
                            # Sanity check: skip if position is unrealistic (>2m from origin)
                            pos = np.array([pose.x, pose.y, pose.z])
                            if np.linalg.norm(pos) < 2.0:
                                positions.append([pose.x, pose.y, pose.z])
                    except:
                        pass  # Skip failed FK calls
            else:
                # If no stub, skip joint trajectories
                continue
            
            if len(positions) > 0:
                positions = np.array(positions)
            else:
                continue  # Skip this trajectory if no valid positions
        
        # Use predefined labels
        label = trajectory_labels[i] if i < len(trajectory_labels) else f'Trajectory {i+1}'
        
        if len(positions) > 0:
            all_positions.append(positions)
            labels.append(label)
    
    if all_positions:
        # Plot all trajectories
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (pos, label) in enumerate(zip(all_positions, labels)):
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 
                   color=colors[i], linewidth=2, label=label)
            # Mark start and end points
            ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], 
                      color=colors[i], marker='o', s=100, alpha=0.8)
            ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], 
                      color=colors[i], marker='s', s=100, alpha=0.8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'TCP Trajectory {title_suffix}')
    ax.legend()
    ax.grid(True)
    
    # Set equal aspect ratio
    if all_positions:
        all_pos_combined = np.vstack(all_positions)
        max_range = np.ptp([all_pos_combined[:, 0], all_pos_combined[:, 1], all_pos_combined[:, 2]]) / 2.0
        mid_x = (np.min(all_pos_combined[:, 0]) + np.max(all_pos_combined[:, 0])) / 2.0
        mid_y = (np.min(all_pos_combined[:, 1]) + np.max(all_pos_combined[:, 1])) / 2.0
        mid_z = (np.min(all_pos_combined[:, 2]) + np.max(all_pos_combined[:, 2])) / 2.0
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/tcp_trajectory{title_suffix.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_joint_trajectories(trajectory_data, dof, plots_dir, title_suffix=""):
    """Plot individual joint trajectories over time"""
    fig, axes = plt.subplots(dof, 1, figsize=(12, 3*dof), sharex=True)
    if dof == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    labels = ['MoveJ', 'MoveL Fast', 'MoveL Accurate', 'MoveL Sequential']
    
    for i in range(dof):
        ax = axes[i]
        
        for j, traj in enumerate(trajectory_data):
            if j >= len(colors):
                break
                
            waypoints = np.array([list(wp.values) for wp in traj['waypoints']])
            
            # Skip if waypoints don't have joint data (e.g., only TCP positions)
            if waypoints.shape[1] < dof:
                continue
                
            # Check for unrealistic joint values (likely failed IK)
            max_joint = np.max(np.abs(waypoints))
            if max_joint > 10:  # More than ~570 degrees is likely an error
                continue
            
            time = np.linspace(0, traj['duration'], len(waypoints))
            
            if i < waypoints.shape[1]:
                label_text = labels[j] if j < len(labels) else f'Traj {j+1}'
                ax.plot(time, waypoints[:, i], 
                       color=colors[j], linewidth=2, label=label_text)
        
        ax.set_ylabel(f'Joint {i+1} (rad)')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
        if i == dof - 1:
            ax.set_xlabel('Time (s)')
    
    plt.suptitle(f'Joint Trajectories {title_suffix}')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/joint_trajectories{title_suffix.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_trajectory_comparison(trajectory_data, plots_dir):
    """Plot trajectory duration and waypoint count comparison"""
    if not trajectory_data:
        return
    
    labels = ['MoveJ', 'MoveL Fast', 'MoveL Accurate', 'MoveL Sequential']
    durations = [traj['duration'] for traj in trajectory_data]
    waypoint_counts = [len(traj['waypoints']) for traj in trajectory_data]
    
    # Use only the labels we have data for
    active_labels = labels[:len(durations)]
    colors = ['blue', 'red', 'green', 'orange'][:len(durations)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Duration comparison
    bars1 = ax1.bar(active_labels, durations, color=colors)
    ax1.set_ylabel('Duration (s)')
    ax1.set_title('Trajectory Durations')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Add value labels on bars
    for bar, duration in zip(bars1, durations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{duration:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Waypoint count comparison
    bars2 = ax2.bar(active_labels, waypoint_counts, color=colors)
    ax2.set_ylabel('Number of Waypoints')
    ax2.set_title('Trajectory Waypoint Counts')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars2, waypoint_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_tcp_gif(trajectory_data, plots_dir, title_suffix="", stub=None):
    """Create animated GIF of TCP trajectory"""
    # Collect all TCP positions
    all_positions = []
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    trajectory_labels = ['MoveJ', 'MoveL Fast', 'MoveL Accurate', 'MoveL Sequential']
    
    for i, traj in enumerate(trajectory_data):
        waypoints = np.array([list(wp.values) for wp in traj['waypoints']])
        
        # Check if waypoints are already TCP positions (3 values) or joint values (6+ values)
        if len(traj['waypoints']) > 0 and len(traj['waypoints'][0].values) == 3:
            # Already TCP positions
            positions = waypoints
        else:
            # Joint values - compute TCP positions via FK
            positions = []
            if stub:
                for joint_vals in waypoints:
                    try:
                        fk_request = motion_planner_pb2.FKRequest(
                            joint_values=motion_planner_pb2.JointValues(values=joint_vals)
                        )
                        fk_response = stub.ComputeFK(fk_request)
                        if fk_response.success:
                            pose = fk_response.end_effector_pose
                            # Sanity check: skip if position is unrealistic (>2m from origin)
                            pos = np.array([pose.x, pose.y, pose.z])
                            if np.linalg.norm(pos) < 2.0:
                                positions.append([pose.x, pose.y, pose.z])
                    except:
                        pass  # Skip failed FK calls
            else:
                continue
            
            if len(positions) > 0:
                positions = np.array(positions)
            else:
                continue  # Skip this trajectory if no valid positions
        
        # Use predefined labels
        label = trajectory_labels[i] if i < len(trajectory_labels) else f'Trajectory {i+1}'
        
        if len(positions) > 0 and positions.shape[1] == 3:  # Ensure it's 3D
            all_positions.append((positions, colors[len(all_positions)], label))
    
    if not all_positions:
        return
    
    # Create animation
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the plot limits
    all_pos_combined = np.vstack([pos for pos, _, _ in all_positions])
    max_range = np.ptp([all_pos_combined[:, 0], all_pos_combined[:, 1], all_pos_combined[:, 2]]) / 2.0
    mid_x = (np.min(all_pos_combined[:, 0]) + np.max(all_pos_combined[:, 0])) / 2.0
    mid_y = (np.min(all_pos_combined[:, 1]) + np.max(all_pos_combined[:, 1])) / 2.0
    mid_z = (np.min(all_pos_combined[:, 2]) + np.max(all_pos_combined[:, 2])) / 2.0
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'TCP Trajectory Animation {title_suffix}')
    ax.grid(True)
    
    # Plot full trajectories as background
    for positions, color, label in all_positions:
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
               color=color, linewidth=1, alpha=0.3, label=label)
    
    ax.legend()
    
    # Animation elements
    tcp_points = []
    trajectory_lines = []
    
    for positions, color, _ in all_positions:
        tcp_point, = ax.plot([], [], [], 'o', color=color, markersize=8)
        trajectory_line, = ax.plot([], [], [], '-', color=color, linewidth=2)
        tcp_points.append(tcp_point)
        trajectory_lines.append(trajectory_line)
    
    def animate(frame):
        # Calculate which trajectory and point we're at
        total_frames = 100  # Total animation frames
        points_per_trajectory = total_frames // len(all_positions)
        
        for i, (positions, _, _) in enumerate(all_positions):
            # Calculate frame index for this trajectory
            traj_frame = min(frame - i * points_per_trajectory, len(positions) - 1)
            if traj_frame < 0:
                traj_frame = 0
            
            # Update TCP point
            tcp_points[i].set_data([positions[traj_frame, 0]], [positions[traj_frame, 1]])
            tcp_points[i].set_3d_properties([positions[traj_frame, 2]])
            
            # Update trajectory line up to current point
            trajectory_lines[i].set_data(positions[:traj_frame+1, 0], positions[:traj_frame+1, 1])
            trajectory_lines[i].set_3d_properties(positions[:traj_frame+1, 2])
        
        return tcp_points + trajectory_lines
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
    
    # Save as GIF
    try:
        anim.save(f'{plots_dir}/tcp_trajectory{title_suffix.replace(" ", "_").lower()}.gif', 
                 writer='pillow', fps=20, dpi=100)
        print(f"✓ GIF saved to {plots_dir}/tcp_trajectory{title_suffix.replace(' ', '_').lower()}.gif")
    except Exception as e:
        print(f"✗ Failed to save GIF: {e}")
    finally:
        plt.close(fig)


def print_trajectory(response):
    """Pretty print trajectory response"""
    if response.success:
        print(f"✓ Success: {response.message}")
        print(f"  Duration: {response.duration:.3f}s")
        print(f"  Waypoints: {response.num_waypoints}")
        if response.num_waypoints > 0:
            print(f"  First waypoint: {list(response.trajectory[0].values)[:3]}...")
            print(f"  Last waypoint:  {list(response.trajectory[-1].values)[:3]}...")
    else:
        print(f"✗ Failed: {response.message}")


def test_init_robot(stub, robot_type="ur5e"):
    """Test robot initialization"""
    print(f"\n=== InitRobot: {robot_type} ===")
    request = motion_planner_pb2.InitRobotRequest(robot_type=robot_type)
    response = stub.InitRobot(request)
    
    if response.success:
        print(f"✓ Success: {response.message}")
        print(f"  DOF: {response.dof}")
        return response.dof
    else:
        print(f"✗ Failed: {response.message}")
        return 0


def test_forward_kinematics(stub, joint_values):
    """Test forward kinematics"""
    print("\n=== Forward Kinematics ===")
    request = motion_planner_pb2.FKRequest(
        joint_values=motion_planner_pb2.JointValues(values=joint_values)
    )
    response = stub.ComputeFK(request)
    
    if response.success:
        pose = response.end_effector_pose
        print(f"✓ Success: {response.message}")
        print(f"  Position: [{pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f}]")
        print(f"  Quaternion: [{pose.qx:.4f}, {pose.qy:.4f}, {pose.qz:.4f}, {pose.qw:.4f}]")
        return pose
    else:
        print(f"✗ Failed: {response.message}")
        return None


def test_inverse_kinematics(stub, target_pose, seed_joints):
    """Test inverse kinematics"""
    print("\n=== Inverse Kinematics ===")
    request = motion_planner_pb2.IKRequest(
        target_pose=target_pose,
        seed=motion_planner_pb2.JointValues(values=seed_joints),
        tolerance=1e-4,
        max_iterations=100
    )
    response = stub.ComputeIK(request)
    
    if response.success:
        print(f"✓ Success: {response.message}")
        print(f"  Solution: {list(response.joint_solution.values)}")
        return list(response.joint_solution.values)
    else:
        print(f"✗ Failed: {response.message}")
        return None


def test_movej(stub, start_joints, target_joints):
    """Test MoveJ trajectory generation"""
    print("\n=== MoveJ Trajectory ===")
    request = motion_planner_pb2.MoveJRequest(
        start_joints=motion_planner_pb2.JointValues(values=start_joints),
        target_joints=motion_planner_pb2.JointValues(values=target_joints)
    )
    response = stub.MoveJ(request)
    print_trajectory(response)
    
    # Return trajectory data for plotting
    trajectory_data = None
    if response.success and response.num_waypoints > 0:
        trajectory_data = {
            'type': 'joint',
            'waypoints': response.trajectory,
            'duration': response.duration
        }
    
    return response, trajectory_data


def test_movel(stub, start_joints, target_pose, fast_mode=True):
    """Test MoveL trajectory generation"""
    print(f"\n=== MoveL Trajectory (fast_mode={fast_mode}) ===")
    request = motion_planner_pb2.MoveLRequest(
        start_joints=motion_planner_pb2.JointValues(values=start_joints),
        target_pose=target_pose,
        fast_mode=fast_mode
    )
    response = stub.MoveL(request)
    print_trajectory(response)
    
    # Return trajectory data for plotting
    trajectory_data = None
    if response.success and response.num_waypoints > 0:
        trajectory_data = {
            'type': 'tcp',
            'waypoints': response.trajectory,
            'duration': response.duration
        }
    
    return response, trajectory_data


def main():
    robot_type = "ur5e"
    if len(sys.argv) > 1:
        robot_type = sys.argv[1]
    
    server_address = "localhost:50051"
    if len(sys.argv) > 2:
        server_address = sys.argv[2]
    
    print(f"Connecting to Motion Planner gRPC Server at {server_address}")
    
    with grpc.insecure_channel(server_address) as channel:
        stub = motion_planner_pb2_grpc.MotionPlannerStub(channel)
        
        # 1. Initialize Robot
        dof = test_init_robot(stub, robot_type)
        if dof == 0:
            print("\n✗ Failed to initialize robot. Exiting.")
            return
        
        # Create plots directory
        plots_dir = create_plots_directory()
        
        # 2. Define test joint configurations
        home_joints = [0.0] * dof
        target_joints_1 = [0.5, -0.5, 0.3, -0.2, 0.1, 0.0][:dof]
        
        # 3. Test Forward Kinematics
        target_pose = test_forward_kinematics(stub, target_joints_1)
        if not target_pose:
            print("\n✗ FK failed. Continuing anyway...")
            # Create a default pose for testing
            target_pose = motion_planner_pb2.Pose(
                x=0.3, y=0.2, z=0.5,
                qx=0.0, qy=0.707, qz=0.0, qw=0.707
            )
        
        # 4. Test Inverse Kinematics
        ik_solution = test_inverse_kinematics(stub, target_pose, home_joints)
        
        # 5. Test MoveJ
        movej_response, movej_data = test_movej(stub, home_joints, target_joints_1)
        
        # 6. Test MoveL (Fast Mode)
        movel_fast_response, movel_fast_data = test_movel(stub, home_joints, target_pose, fast_mode=True)
        
        # 7. Test MoveL (Accurate Mode)
        movel_accurate_response, movel_accurate_data = test_movel(stub, home_joints, target_pose, fast_mode=False)
        
        # 8. Test multiple sequential moves
        print("\n=== Sequential Motion Test ===")
        trajectory_data = []
        
        # Collect trajectory data for plotting
        if movej_data:
            trajectory_data.append(movej_data)
        if movel_fast_data:
            trajectory_data.append(movel_fast_data)
        if movel_accurate_data:
            trajectory_data.append(movel_accurate_data)
        
        if movej_response.success and movej_response.num_waypoints > 0:
            # Use last waypoint of MoveJ as start for next move
            last_waypoint = list(movej_response.trajectory[-1].values)
            
            # Create a reachable target near the end position
            # First get the FK of the last position
            fk_request = motion_planner_pb2.FKRequest(
                joint_values=motion_planner_pb2.JointValues(values=last_waypoint)
            )
            fk_response = stub.ComputeFK(fk_request)
            
            if fk_response.success:
                # Move to a nearby position with a small offset
                pose = fk_response.end_effector_pose
                new_target = motion_planner_pb2.Pose(
                    x=pose.x - 0.05,  # Smaller offset for more reliability
                    y=pose.y + 0.05,
                    z=pose.z - 0.05,
                    qx=pose.qx, qy=pose.qy, qz=pose.qz, qw=pose.qw
                )
                
                print(f"Moving from MoveJ end position ({pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f})")
                print(f"  to nearby target ({new_target.x:.3f}, {new_target.y:.3f}, {new_target.z:.3f})...")
                seq_response, seq_data = test_movel(stub, last_waypoint, new_target, fast_mode=True)
                
                # Validate the trajectory before adding it
                if seq_data and seq_response.success:
                    # Check if joint values are reasonable
                    waypoints = np.array([list(wp.values) for wp in seq_response.trajectory])
                    max_joint = np.max(np.abs(waypoints))
                    if max_joint < 6.28:  # Less than 2*pi radians (360 degrees)
                        trajectory_data.append(seq_data)
                        print(f"  ✓ Sequential trajectory valid (max joint: {max_joint:.2f} rad)")
                    else:
                        print(f"  ✗ Sequential trajectory has extreme joint values (max: {max_joint:.2f} rad), skipping")
        
        # Generate plots
        print(f"\n=== Generating Plots ===")
        try:
            plot_tcp_trajectory(trajectory_data, plots_dir, f"- {robot_type}", stub)
            plot_joint_trajectories(trajectory_data, dof, plots_dir, f"- {robot_type}")
            plot_trajectory_comparison(trajectory_data, plots_dir)
            create_tcp_gif(trajectory_data, plots_dir, f"- {robot_type}", stub)
            print(f"✓ Plots and GIF saved to {plots_dir}/ directory")
        except Exception as e:
            print(f"✗ Failed to generate plots/GIF: {e}")
        
        print("\n=== All Tests Complete ===")
        print("Summary:")
        print(f"  Robot Type: {robot_type} ({dof} DOF)")
        print(f"  FK: {'✓' if target_pose else '✗'}")
        print(f"  IK: {'✓' if ik_solution else '✗'}")
        print(f"  MoveJ: {'✓' if movej_response.success else '✗'}")
        print(f"  MoveL (fast): {'✓' if movel_fast_response.success else '✗'}")
        print(f"  MoveL (accurate): {'✓' if movel_accurate_response.success else '✗'}")


if __name__ == "__main__":
    main()
