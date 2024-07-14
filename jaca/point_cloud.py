import rerun as rr
from rerun.datatypes import Quaternion, TranslationRotationScale3D
import time
import numpy as np
import uuid
import open3d as o3d
from transforms3d.quaternions import mat2quat


class PointCloudVisualizer:
    def __init__(self, name: str = None, remote_url: str = None):
        """
        Initialize the rerun client

        Args:
            name (str, optional): Name of the rerun instance. Defaults to None.
            remote_url (str, optional): The ip:port to connect to.
        """
        if name is None:
            name = str(uuid.uuid4())
        rr.init(name)
        if remote_url is not None:
            rr.connect(remote_url)
        else:
            rr.spawn()
        self.name = name

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
        self.uuid_to_oobb = {}  # {uuid, oobb} for each point cloud
        self.uuid_to_color = {}  # {uuid, color} for each point cloud

    def _pcd_to_oobb(self, pcd: np.ndarray):
        """
        Find the oriented bounding box of a point cloud using Open3D
        """
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
        oobb = o3d_pcd.get_oriented_bounding_box()
        return oobb

    def _get_random_rgb(self) -> np.ndarray:
        return (np.random.rand(3) * 255).astype(np.uint8)

    def log_point_cloud(
        self,
        pcd: np.ndarray,
        obb: np.ndarray = None,
        log_bbox: bool = True,
        colors: np.ndarray = None,
        id: str = None,
        classification: str = None,
        radius: float = 0.01,
        observation_time: float = None,
    ):
        """
        Log point cloud with optional color, classification, and time

        Args:
            pcd (np.ndarray): [n, 3]
            obb (np.ndarray, optional): [8, 3]. Oriented bounding box of the pcd.
            colors (np.ndarray, optional): [3, ] or [n, 3]. Defaults to None for persistent random colors.
            id (str, optional): For persistence. If None, will try to find the uuid of the closest point cloud.
            classification (str, optional): Defaults to None if you don't want any labels.
            observation_time (float, optional): Time since epoch in seconds. If None current time used.
            radius (float, optional): Defaults to 0.01 meters.
        """
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        if obb is None:
            oobb = self._pcd_to_oobb(pcd)
        else:
            oobb = obb
        extent = oobb.extent
        R = oobb.R
        center = oobb.center

        if id is None:
            id = str(uuid.uuid4())
        if colors is None:
            colors = self.uuid_to_color.get(id, self._get_random_rgb())
        self.uuid_to_oobb[id] = oobb
        self.uuid_to_color[id] = colors
        rr.log(f"/pcd/pts/{id}", rr.Points3D(pcd, colors=colors, radii=radius))
        # draw bounding box too
        if log_bbox:
            q = mat2quat(R)
            q = np.roll(q, -1)  # turn wxyz to xyzw
            q = Quaternion(xyzw=q)
            rr.log(
                f"/pcd/obb/{id}",
                rr.Boxes3D(
                    half_sizes=extent / 2,
                    centers=center,
                    rotations=q,
                    colors=colors,
                    labels=classification,
                ),
            )

    def remove_point_cloud(
        self,
        id: str,
        observation_time: float = None,
    ):
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        pcd = np.zeros((1, 3))
        colors = self.uuid_to_color.get(id, self._get_random_rgb())
        self.uuid_to_oobb[id] = None
        self.uuid_to_color[id] = colors
        rr.log(f"/pcd/pts/{id}", rr.Points3D(pcd, colors=colors, radii=0.001))
        rr.log(
            f"/pcd/obb/{id}",
            rr.Boxes3D(
                half_sizes=0.001,
                centers=pcd[0],
                colors=colors,
                labels="removed",
            ),
        )

    def log_tf(
        self,
        tf: np.ndarray,
        scale: float = 0.3,
        id: str = None,
        observation_time: float = None,
    ):
        """
        Log transformation frame

        Args:
            tf (np.ndarray): [4, 4]
            scale (float, optional): Defaults to 0.3 meters.
            observation_time (float, optional): Time since epoch in seconds. If None current time used.
        """
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        if id is None:
            id = str(uuid.uuid4())
        transform = TranslationRotationScale3D(
            translation=tf[:3, 3],
            rotation=Quaternion(xyzw=np.roll(mat2quat(tf[:3, :3]), -1)),
            scale=scale * 2,  # this is the halfsize scale
        )
        rr.log(f"/tf/{id}", rr.Transform3D(transform))

    def set_time_sequence(self, time_line: str, time_sequence: np.ndarray) -> None:
        """Set the time sequence for the visualizer."""
        rr.set_time_sequence(time_line, time_sequence)

    def set_time_seconds(self, observation_time: float) -> None:
        """Set the time in seconds for the visualizer."""
        rr.set_time_seconds("real_clock", observation_time)

    def log_trajectory(
        self,
        trajectory: np.ndarray,
        id: str = None,
        colors: np.ndarray = None,
        observation_time: float = None,
    ):
        """
        Log trajectory

        Args:
            trajectory (np.ndarray): [n, 2] or [n, 3]
            observation_time (float, optional): Time since epoch in seconds. If None current time used.
        """
        if observation_time is None:
            observation_time = time.time()
        rr.set_time_seconds("real_clock", observation_time)
        if id is None:
            id = str(uuid.uuid4())
        if trajectory.shape[1] == 2:
            trajectory = np.hstack((trajectory, np.zeros((trajectory.shape[0], 1))))
        colors = colors or self._get_random_rgb()
        rr.log(f"/trajectory/{id}", rr.Clear(recursive=True))
        for i in range(trajectory.shape[0] - 1):
            rr.log(
                f"/trajectory/{id}/{i}",
                rr.Arrows3D(
                    origins=trajectory[i],
                    vectors=trajectory[i + 1] - trajectory[i],
                    colors=colors,
                ),
            )


# Test script to demonstrate usage
if __name__ == "__main__":
    # Initialize the visualizer
    viz = PointCloudVisualizer()

    # Generate a random Gaussian point cloud of 100 points
    points = np.random.randn(100, 3)
    viz.log_point_cloud(points, classification="random")
    time.sleep(0.1)

    # Log the same point cloud shifted slightly
    points += 0.1
    viz.log_point_cloud(points)  # This should have the same UUID
    time.sleep(0.1)

    # Log the point cloud shifted significantly
    points += 1
    viz.log_point_cloud(points)  # This should have a new UUID
    time.sleep(0.1)

    # Log a transformation frame
    tf = np.eye(4)
    tf[0, 3] = 1
    viz.log_tf(tf, scale=0.5)
    time.sleep(0.1)

    # Log a random trajectory
    trajectory = np.random.randn(10, 2)
    viz.log_trajectory(trajectory)
    time.sleep(0.1)

    # Log a color camera image
    image = np.random.rand(100, 100, 3)
    tf = np.eye(4)
    viz.log_camera(image, tf)
    time.sleep(0.1)

    # Log a depth camera image
    depth_image = np.ones(shape=(100, 100))
    viz.log_camera(depth_image, tf)
    time.sleep(0.1)
