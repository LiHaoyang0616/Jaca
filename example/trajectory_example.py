import numpy as np
import rerun as rr
from jaca.visualizer import Visualizer


def main():
    rr.init("trajectory_example")
    visualizer = Visualizer()

    # Create a sample trajectory
    num_waypoints = 50
    t = np.linspace(0, 2 * np.pi, num_waypoints)
    waypoints = np.column_stack([5 * np.cos(t), 5 * np.sin(t), t])

    # Visualize the trajectory
    visualizer.visualize_trajectory(waypoints)

    # Show the visualization
    visualizer.show()


if __name__ == "__main__":
    main()
