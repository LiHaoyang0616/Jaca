from jaca.robot_model import URDFVisualizer

import rerun as rr


def test_urdf_visualization():
    # Initialize Rerun
    rr.init("URDF Test Visualization", recording_id="test_recording")

    # Create a URDFVisualizer instance and visualize the URDF file
    urdf_filepath = (
        "/home/haoyang/project/haoyang/Jaca/urdf/xarm_urdf/xarm7_gripper.urdf"
    )
    visualizer = URDFVisualizer(urdf_filepath, entity_path_prefix="test_prefix")
    visualizer.visualize()

    # Start the Rerun Viewer process (if it's not automatically started)
    rr.spawn()


if __name__ == "__main__":
    test_urdf_visualization()
