import numpy as np
from jaca.camera import CameraVisualizer
import rerun as rr
import PIL.Image as Image
import time


def create_test_image(width, height):
    """Create a simple test RGB image."""
    r = np.linspace(0, 1, width)
    g = np.linspace(0, 1, height)
    b = r[:, np.newaxis] * g
    return np.dstack((r[:, np.newaxis] * g, b, np.flip(b))) * 255


def create_test_depth(width, height):
    """Create a simple test depth image."""
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)
    return np.sqrt(xx**2 + yy**2) * 10  # Scale to 0-10 meters


def main():
    # Create a CameraVisualizer instance
    viz = CameraVisualizer("test_camera")

    image_path = (
        "/home/haoyang/project/haoyang/MobileCap/data/20240701-125304/images/0003.png"
    )
    rgb = Image.open(image_path)
    rgb_image_1 = np.array(rgb)
    depth_image_1 = np.ones((rgb_image_1.shape[0], rgb_image_1.shape[1]))

    # Load test images
    width, height = 1920, 1080
    rgb_image_2 = np.random.rand(width, height, 3)
    depth_image_2 = np.ones((width, height)) * 1.5

    # Add multiple cameras
    intrinsics_1 = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
    intrinsics_2 = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]])

    extrinsics_1 = np.eye(4)
    extrinsics_1[0, 3] = 0.1

    extrinsics_2 = np.eye(4)
    extrinsics_2[:3, 3] = [0.1, 0.2, 0.3]

    viz.add_camera("camera_1", intrinsics_1, extrinsics_1)
    viz.add_camera("camera_2", intrinsics_2, extrinsics_2)

    for i in range(3):
        # Log the camera data for each camera
        extrinsics_1[0, 3] += 0.1 * i
        viz.update_camera("camera_1", extrinsics=extrinsics_1)
        viz.log_camera(
            "camera_1",
            image=rgb_image_1,
            depth=depth_image_1,
            observation_time=time.time(),
        )

        time.sleep(0.5)
        extrinsics_2[2, 3] += 0.1 * i
        viz.update_camera("camera_2", extrinsics=extrinsics_2)
        viz.log_camera(
            "camera_2",
            image=rgb_image_2,
            depth=depth_image_2,
            observation_time=time.time(),
        )

        time.sleep(1.0)

    # Generate a random Gaussian point cloud of 100 points
    points = np.random.randn(100, 3)
    colors = (np.random.rand(100, 3) * 255).astype(np.uint8)

    viz.log_point_cloud(points, colors=colors)
    print("log point cloud")


if __name__ == "__main__":
    main()
