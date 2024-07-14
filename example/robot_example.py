from jaca.robot_model import URDFVisualizer

import rerun as rr
import numpy as np
import time


def test_urdf_visualization():
    # Initialize Rerun
    # rr.init("URDF Test Visualization", recording_id="test_recording")

    # Create a URDFVisualizer instance and visualize the URDF file
    urdf_filepath = (
        "/home/haoyang/project/haoyang/Jaca/urdf/xarm_urdf/xarm7_gripper.urdf"
    )
    visualizer = URDFVisualizer(
        "test_prefix", urdf_filepath, entity_path_prefix="test_prefix"
    )
    # visualizer.visualize()

    visualizer.set_root_pose(np.array([0, 0, 0]), np.array([0, 0, 0]))
    visualizer.visualize()

    time.sleep(2.0)
    # Update root pose and re-visualize
    visualizer.set_root_pose(np.array([1, 1, 2.0]), np.array([0.5, 0.5, 0.5]))
    visualizer.visualize()

    time.sleep(2.0)
    # Update root pose and re-visualize
    visualizer.set_root_pose(np.array([1, 1, 12.0]), np.array([0.5, 0.5, 0.5]))
    visualizer.visualize()

    # Start the Rerun Viewer process (if it's not automatically started)
    rr.spawn()


if __name__ == "__main__":
    test_urdf_visualization()
