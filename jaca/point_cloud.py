import rerun as rr
import numpy as np


class PointCloud:
    def __init__(self, name):
        self.name = name

    def visualize(self, points, colors=None):
        if colors is None:
            colors = np.ones_like(points) * 255

        rr.log(f"{self.name}/points", rr.Points3D(points, colors=colors))
