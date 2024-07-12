import rerun as rr
import numpy as np


class Camera:
    def __init__(self, name):
        self.name = name

    def visualize(self, pose, rgb_image):
        # Visualize camera pose
        rr.log(f"{self.name}", rr.Transform3D(translation=pose[:3], rotation=pose[3:]))

        # Visualize RGB image
        rr.log(f"{self.name}/image", rr.Image(rgb_image))

        # Visualize camera frustum
        rr.log(
            f"{self.name}/frustum",
            rr.ViewCoordinates(xyz=rr.ViewCoordinates.RIGHT_HAND_Y_DOWN),
        )

        # Add pinhole camera
        rr.log(
            f"{self.name}/pinhole",
            rr.Pinhole(
                resolution=[rgb_image.shape[1], rgb_image.shape[0]],
                focal_length=[500.0, 500.0],
                principal_point=[rgb_image.shape[1] / 2, rgb_image.shape[0] / 2],
            ),
        )

        # Add a transform to represent the camera's position and orientation
        rr.log(
            f"{self.name}/transform",
            rr.Transform3D(translation=pose[:3], rotation=pose[3:]),
        )

        # Add a view coordinates system to represent the camera's view

        # In camera.py
        rr.log(
            f"{self.name}/view",
            rr.ViewCoordinates(xyz=rr.ViewCoordinates.RIGHT_HAND_Y_DOWN),
        )
