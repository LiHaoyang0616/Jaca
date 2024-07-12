import numpy as np
import rerun as rr
from jaca.visualizer import Visualizer


def main():
    rr.init("camera_example", spawn=True)
    visualizer = Visualizer()

    # Create a sample camera pose (position and orientation)
    pose = np.array([1, 2, 3, 0, 0, 0, 1])  # x, y, z, qx, qy, qz, qw

    # Create a sample RGB image
    rgb_image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)

    # Visualize the camera
    visualizer.visualize_camera(pose, rgb_image)

    # Log the recording
    rr.log_file_system()


if __name__ == "__main__":
    main()
