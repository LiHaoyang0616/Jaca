import numpy as np
import rerun as rr
from jaca.visualizer import Visualizer


def main():
    rr.init("point_cloud_example")
    visualizer = Visualizer()

    # Create a sample point cloud
    num_points = 1000
    points = np.random.rand(num_points, 3) * 10  # Random points in a 10x10x10 cube
    colors = np.random.randint(0, 255, size=(num_points, 3), dtype=np.uint8)

    # Visualize the point cloud
    visualizer.visualize_point_cloud(points, colors)

    # Show the visualization
    visualizer.show()


if __name__ == "__main__":
    main()
