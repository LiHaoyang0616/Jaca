import rerun as rr
import numpy as np


class Trajectory:
    def __init__(self, name):
        self.name = name

    def visualize(self, waypoints):
        rr.log(f"{self.name}/trajectory", rr.LineStrips3D(waypoints))

        # Visualize waypoints as small spheres
        for i, waypoint in enumerate(waypoints):
            rr.log(
                f"{self.name}/waypoint_{i}",
                rr.Points3D(positions=[waypoint], radii=[0.1], colors=[[255, 0, 0]]),
            )  # Red color for waypoints
