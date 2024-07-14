import numpy as np
import rerun as rr
from rerun.datatypes import Quaternion, TranslationRotationScale3D
import time
import uuid
from transforms3d.quaternions import mat2quat
from typing import Optional, Tuple, Dict


class CameraVisualizer:
    def __init__(self, name: Optional[str] = None, remote_url: Optional[str] = None):
        self.name = name or str(uuid.uuid4())
        rr.init(self.name)
        if remote_url is not None:
            rr.connect(remote_url)
        else:
            rr.spawn()
        self.set_time_seconds(time.time())
        rr.log(
            f"{self.name}", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True
        )  # Set an up-axis
        rr.log(
            f"{self.name}/world_axes",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )  # Show the axes

        self.camera_data: Dict = {}
        self.cameras: Dict[str, Dict] = {}

    def add_camera(
        self,
        camera_id: str,
        intrinsics: Optional[np.ndarray] = None,
        extrinsics: Optional[np.ndarray] = None,
    ):
        """
        Add a new camera to the visualizer.

        Args:
            camera_id (str): Unique identifier for the camera.
            intrinsics (np.ndarray, optional): 3x3 camera intrinsic matrix. If None, will be estimated from the image size when logging images.
            extrinsics (np.ndarray, optional): 4x4 camera extrinsic matrix (transformation matrix). If None, will default to identity matrix.
        """
        if intrinsics is not None and intrinsics.shape != (3, 3):
            raise ValueError(
                f"Intrinsics for camera '{camera_id}' must be a 3x3 matrix."
            )

        self.cameras[camera_id] = {
            "intrinsics": intrinsics,
            "extrinsics": extrinsics if extrinsics is not None else np.eye(4),
        }

    def update_camera(
        self,
        camera_id: str,
        intrinsics: Optional[np.ndarray] = None,
        extrinsics: Optional[np.ndarray] = None,
    ):
        """
        Update the intrinsics and/or extrinsics of an existing camera.

        Args:
            camera_id (str): Unique identifier for the camera.
            intrinsics (np.ndarray, optional): New 3x3 camera intrinsic matrix. If None, keeps the existing matrix.
            extrinsics (np.ndarray, optional): New 4x4 camera extrinsic matrix. If None, keeps the existing matrix.
        """
        if camera_id not in self.cameras:
            raise ValueError(
                f"Camera '{camera_id}' not found. Please add the camera first."
            )

        if intrinsics is not None:
            if intrinsics.shape != (3, 3):
                raise ValueError(
                    f"Intrinsics for camera '{camera_id}' must be a 3x3 matrix."
                )
            self.cameras[camera_id]["intrinsics"] = intrinsics

        if extrinsics is not None:
            if extrinsics.shape != (4, 4):
                raise ValueError(
                    f"Extrinsics for camera '{camera_id}' must be a 4x4 matrix."
                )
            self.cameras[camera_id]["extrinsics"] = extrinsics

    def log_camera(
        self,
        camera_id: str,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        observation_time: Optional[float] = None,
    ) -> None:
        """
        Log camera data including RGB image, depth image (if provided), camera pose, and properties.

        Args:
            camera_id (str): Unique identifier for the camera.
            image (np.ndarray): RGB image of shape [h, w, 3]
            depth (np.ndarray, optional): Depth image of shape [h, w]. If provided, will be logged separately.
            observation_time (float, optional): Time of observation. If None, current time is used.
        """
        if camera_id not in self.cameras:
            raise ValueError(
                f"Camera '{camera_id}' not found. Please add the camera first."
            )

        if observation_time is None:
            observation_time = time.time()
        else:
            observation_time = observation_time

        self.set_time_seconds(observation_time)

        self.camera_data = self.cameras[camera_id]

        # Log the content
        self._log_rgb_image(camera_id, image)
        if depth is not None:
            self._log_depth_image(camera_id, depth)
        # Visualize the camera frame
        self._log_camera_properties(
            camera_id, image.shape, self.camera_data["intrinsics"]
        )
        # Transform the camera frame
        self._log_camera_pose(camera_id, self.camera_data["extrinsics"])

    def _log_rgb_image(self, camera_id: str, image: np.ndarray) -> None:
        """Log RGB image"""
        rr.log(f"{self.name}/{camera_id}/camera/rgb", rr.Image(image))

    def _log_depth_image(self, camera_id: str, depth: np.ndarray) -> None:
        """Log depth image"""
        rr.log(f"{self.name}/{camera_id}/camera/depth", rr.DepthImage(depth, meter=1.0))

    def _log_camera_properties(
        self,
        camera_id: str,
        image_shape: Tuple[int, int, int],
        intrinsics: Optional[np.ndarray],
    ) -> None:
        """Log camera intrinsics and resolution"""
        if intrinsics is None:
            intrinsics = self._estimate_intrinsics(image_shape)

        rr.log(
            f"{self.name}/{camera_id}/camera",
            rr.Pinhole(
                image_from_camera=intrinsics,
                resolution=(image_shape[1], image_shape[0]),
            ),
        )

    def _log_camera_pose(self, camera_id: str, transform: np.ndarray) -> None:
        """Log camera pose"""
        rotation = Quaternion(xyzw=np.roll(mat2quat(transform[:3, :3]), -1))
        translation = transform[:3, 3]

        camera_transform = TranslationRotationScale3D(
            translation=translation,
            rotation=rotation,
        )
        rr.log(f"{self.name}/{camera_id}/camera", rr.Transform3D(camera_transform))

    @staticmethod
    def _estimate_intrinsics(image_shape: Tuple[int, int, int]) -> np.ndarray:
        """Estimate camera intrinsics from image shape"""
        return np.array(
            [
                [image_shape[1] / 2, 0, image_shape[1] / 2],
                [0, image_shape[0] / 2, image_shape[0] / 2],
                [0, 0, 1],
            ]
        )

    def log_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        observation_time: Optional[float] = None,
    ) -> None:
        """
        Log a point cloud.

        Args:
            points (np.ndarray): Point cloud data of shape [n, 3]
            colors (np.ndarray, optional): Color data of shape [n, 3] or [3,]
            observation_time (float, optional): Time of observation. If None, current time is used.
        """
        observation_time = observation_time or time.time()
        self.set_time_seconds(observation_time)
        rr.log(f"{self.name}/points", rr.Points3D(points, colors=colors))

    def log_annotations(
        self,
        annotations: dict,
        observation_time: Optional[float] = None,
    ) -> None:
        """
        Log annotations.

        Args:
            annotations (dict): Annotations data
            observation_time (float, optional): Time of observation. If None, current time is used.
        """
        observation_time = observation_time or time.time()
        rr.set_time_seconds("real_clock", observation_time)
        for key, value in annotations.items():
            rr.log(f"{self.name}/annotations/{key}", value)

    def set_time_sequence(self, time_line: str, time_sequence: np.ndarray) -> None:
        """Set the time sequence for the visualizer."""
        rr.set_time_sequence(time_line, time_sequence)

    def set_time_seconds(self, observation_time: float) -> None:
        """Set the time in seconds for the visualizer."""
        rr.set_time_seconds("real_clock", observation_time)


# Test script to demonstrate usage
if __name__ == "__main__":
    # Initialize the visualizer
    viz = CameraVisualizer()

    # Add multiple cameras
    intrinsics_1 = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
    extrinsics_1 = np.eye(4)
    extrinsics_1[0, 3] = 1

    intrinsics_2 = np.array([[800, 0, 400], [0, 800, 400], [0, 0, 1]])
    extrinsics_2 = np.eye(4)
    extrinsics_2[1, 3] = 2

    viz.add_camera("camera_1", intrinsics_1, extrinsics_1)
    viz.add_camera("camera_2", intrinsics_2, extrinsics_2)

    # Load test images
    rgb_image_1 = np.random.rand(100, 100, 3)
    depth_image_1 = np.ones((100, 100))

    rgb_image_2 = np.random.rand(100, 100, 3)
    depth_image_2 = np.ones((100, 100)) * 2

    # Log the camera data for each camera
    viz.log("camera_1", rgb_image_1, depth=depth_image_1)
    viz.log("camera_2", rgb_image_2, depth=depth_image_2)

    # Generate a random Gaussian point cloud of 100 points
    points = np.random.randn(100, 3)
    colors = (np.random.rand(100, 3) * 255).astype(np.uint8)
    viz.log_point_cloud(points, colors=colors)

    # Log some annotations
    annotations = {
        "object": rr.TextLog("This is a test object."),
    }
    viz.log_annotations(annotations)
