import numpy as np
import rerun as rr
from jaca.visualizer import Visualizer


def main():
    visualizer = Visualizer()
    visualizer.init("combined_example")

    # Camera
    camera_pose = np.array([0, 0, 5, 0, 0, -0.7071, 0.7071])  # x, y, z, qx, qy, qz, qw
    rgb_image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

    # Point Cloud
    num_points = 1000
    points = (
        np.random.rand(num_points, 3) * 10 - 5
    )  # Random points in a 10x10x10 cube centered at origin
    colors = np.random.randint(0, 255, size=(num_points, 3), dtype=np.uint8)

    # Robot Model
    urdf_path = "/home/haoyang/project/haoyang/Jaca/urdf/hillbot_alpha/alpha1.urdf"
    joint_positions = {
        "joint1": 0.5,
        "joint2": -0.3,
        "joint3": 0.7,
    }

    # Trajectory
    num_waypoints = 50
    t = np.linspace(0, 2 * np.pi, num_waypoints)
    waypoints = np.column_stack([5 * np.cos(t), 5 * np.sin(t), np.zeros_like(t)])

    # Visualize all components
    visualizer.visualize_all(
        camera_pose, rgb_image, points, colors, urdf_path, joint_positions, waypoints
    )

    # Log the recording
    visualizer.log_file_system()


if __name__ == "__main__":
    main()
