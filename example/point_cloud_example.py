import numpy as np
import rerun as rr
from jaca.point_cloud import PointCloudVisualizer
import time


def main():
    # Initialize the visualizer
    viz = PointCloudVisualizer("point_cloud_example")

    # Generate a random Gaussian point cloud of 100 points
    points = np.random.randn(100, 3)
    viz.log_point_cloud(points, classification="random", radius=0.5)
    time.sleep(0.1)

    # ============================================
    # example of transforming the point cloud
    # ============================================

    # # Log the point cloud shifted significantly
    # points += 1
    # viz.log_point_cloud(points)  # This should have a new UUID
    # time.sleep(0.1)

    # # Log a transformation frame
    # tf = np.eye(4)
    # tf[0, 3] = 1
    # viz.log_tf(tf, scale=0.5)
    # time.sleep(0.1)

    # # Log a random trajectory
    # trajectory = np.random.randn(10, 2)
    # viz.log_trajectory(trajectory)
    # time.sleep(0.1)

    # # Log a color camera image
    # image = np.random.rand(100, 100, 3)
    # tf = np.eye(4)
    # viz.log_camera(image, tf)
    # time.sleep(0.1)

    # # Log a depth camera image
    # depth_image = np.ones(shape=(100, 100))
    # viz.log_camera(depth_image, tf)
    # time.sleep(0.1)


if __name__ == "__main__":
    main()
