import rerun as rr
from .camera import Camera
from .point_cloud import PointCloud
from .robot_model import RobotModel
from .trajectory import Trajectory


class Visualizer:
    def __init__(self):
        self.camera = Camera("camera")
        self.point_cloud = PointCloud("point_cloud")
        self.robot_model = RobotModel("robot")
        self.trajectory = Trajectory("trajectory")

    def visualize_camera(self, pose, rgb_image):
        self.camera.visualize(pose, rgb_image)

    def visualize_point_cloud(self, points, colors=None):
        self.point_cloud.visualize(points, colors)

    def visualize_robot_model(self, urdf_path, joint_positions):
        self.robot_model.visualize(urdf_path, joint_positions)

    def visualize_trajectory(self, waypoints):
        self.trajectory.visualize(waypoints)

    def visualize_all(
        self,
        camera_pose,
        rgb_image,
        points,
        colors,
        urdf_path,
        joint_positions,
        waypoints,
    ):
        self.visualize_camera(camera_pose, rgb_image)
        self.visualize_point_cloud(points, colors)
        self.visualize_robot_model(urdf_path, joint_positions)
        self.visualize_trajectory(waypoints)

    @staticmethod
    def init(name="visualization"):
        rr.init(name, spawn=True)

    @staticmethod
    def log_file_system():
        rr.log_file_system()
