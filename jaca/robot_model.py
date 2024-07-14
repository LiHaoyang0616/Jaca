import numpy as np
import rerun as rr
import scipy.spatial.transform as st
import trimesh
from PIL import Image
from urdf_parser_py import urdf as urdf_parser
from typing import Optional, List
import pathlib
import os
import time


class URDFVisualizer:
    """Class to visualize a URDF using Rerun."""

    def __init__(
        self, name, filepath: str, entity_path_prefix: Optional[str] = None
    ) -> None:
        self.urdf = urdf_parser.URDF.from_xml_file(filepath)
        self.urdf_path = os.path.dirname(filepath)
        self.entity_path_prefix = entity_path_prefix
        self.mat_name_to_mat = {mat.name: mat for mat in self.urdf.materials}
        self.root_pose = np.eye(4)
        self.name = name

        rr.init(self.name, spawn=True)
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
        )

    def set_root_pose(self, translation: np.ndarray, rotation: np.ndarray) -> None:
        """Set the root pose of the URDF."""
        self.root_pose[:3, 3] = translation
        self.root_pose[:3, :3] = st.Rotation.from_euler("xyz", rotation).as_matrix()
        print(self.root_pose)

    def update_joint(
        self,
        joint_name: str,
        translation: Optional[np.ndarray] = None,
        rotation: Optional[np.ndarray] = None,
    ) -> None:
        """Update the translation and/or rotation of a specific joint."""
        for joint in self.urdf.joints:
            if joint.name == joint_name:
                if translation is not None:
                    joint.origin.xyz = translation
                if rotation is not None:
                    joint.origin.rpy = rotation

    def link_entity_path(self, link: urdf_parser.Link) -> str:
        """Return the entity path for the URDF link."""
        root_name = self.urdf.get_root()
        link_names = self.urdf.get_chain(root_name, link.name)[0::2]  # skip the joints
        return self.add_entity_path_prefix("/".join(link_names))

    def joint_entity_path(self, joint: urdf_parser.Joint) -> str:
        """Return the entity path for the URDF joint."""
        root_name = self.urdf.get_root()
        joint_names = self.urdf.get_chain(root_name, joint.child)[
            0::2
        ]  # skip the links
        return self.add_entity_path_prefix("/".join(joint_names))

    def add_entity_path_prefix(self, entity_path: str) -> str:
        """Add prefix (if passed) to entity path."""
        if self.entity_path_prefix is not None:
            return f"{self.entity_path_prefix}/{entity_path}"
        return entity_path

    def visualize(self, observation_time: float = None) -> None:
        """Visualize a URDF file in Rerun."""
        if observation_time is None:
            observation_time = time.time()
        else:
            observation_time = observation_time
        self.set_time_seconds(observation_time=observation_time)

        # Log the root link with the current root pose
        root_link = self.urdf.links[0]
        root_entity_path = self.add_entity_path_prefix(self.urdf.get_root())
        self.log_link(root_entity_path, root_link, self.root_pose)

        for joint in self.urdf.joints:
            entity_path = self.joint_entity_path(joint)
            self.log_joint(entity_path, joint)

        for link in self.urdf.links[1:]:
            entity_path = self.link_entity_path(link)
            self.log_link(entity_path, link)

    def log_link(
        self,
        entity_path: str,
        link: urdf_parser.Link,
        transform: Optional[np.ndarray] = None,
    ) -> None:
        """Log a URDF link to Rerun."""
        if transform is None:
            transform = np.eye(4)
        for i, visual in enumerate(link.visuals):
            self.log_visual(entity_path + f"/visual_{i}", visual, transform)

    def log_joint(self, entity_path: str, joint: urdf_parser.Joint) -> None:
        """Log a URDF joint to Rerun."""
        translation = rotation = None

        if joint.origin is not None and joint.origin.xyz is not None:
            translation = joint.origin.xyz

        if joint.origin is not None and joint.origin.rpy is not None:
            rotation = st.Rotation.from_euler("xyz", joint.origin.rpy).as_matrix()

        # Split the entity path to add "agent"
        parts = entity_path.split("/")
        if len(parts) > 1:
            log_path = "/".join([parts[0], "agent"] + parts[1:])
        else:
            log_path = "/".join([parts[0], "agent"])
        rr.log(
            log_path,
            rr.Transform3D(translation=translation, mat3x3=rotation),
        )

    def log_visual(
        self, entity_path: str, visual: urdf_parser.Visual, parent_transform: np.ndarray
    ) -> None:
        """Log a URDF visual to Rerun."""
        material = None
        if visual.material is not None:
            if visual.material.color is None and visual.material.texture is None:
                material = self.mat_name_to_mat.get(visual.material.name)
            else:
                material = visual.material

        transform = np.eye(4)
        if visual.origin is not None and visual.origin.xyz is not None:
            transform[:3, 3] = visual.origin.xyz
        if visual.origin is not None and visual.origin.rpy is not None:
            transform[:3, :3] = st.Rotation.from_euler(
                "xyz", visual.origin.rpy
            ).as_matrix()

        transform = parent_transform @ transform
        mesh_or_scene = self.load_mesh_or_geometry(visual, transform)
        self.log_mesh_or_scene(entity_path, mesh_or_scene, material)

    def load_mesh_or_geometry(
        self, visual: urdf_parser.Visual, transform: np.ndarray
    ) -> trimesh.Trimesh:
        if isinstance(visual.geometry, urdf_parser.Mesh):
            resolved_path = self.resolve_ros_path(visual.geometry.filename)
            resolved_path = os.path.join(self.urdf_path, resolved_path)
            mesh_scale = visual.geometry.scale
            mesh_or_scene = trimesh.load_mesh(resolved_path)
            if mesh_scale is not None:
                transform[:3, :3] *= mesh_scale
        elif isinstance(visual.geometry, urdf_parser.Box):
            mesh_or_scene = trimesh.creation.box(extents=visual.geometry.size)
        elif isinstance(visual.geometry, urdf_parser.Cylinder):
            mesh_or_scene = trimesh.creation.cylinder(
                radius=visual.geometry.radius, height=visual.geometry.length
            )
        elif isinstance(visual.geometry, urdf_parser.Sphere):
            mesh_or_scene = trimesh.creation.icosphere(radius=visual.geometry.radius)
        else:
            rr.log(
                "",
                rr.TextLog("Unsupported geometry type: " + str(type(visual.geometry))),
            )
            mesh_or_scene = trimesh.Trimesh()

        mesh_or_scene.apply_transform(transform)
        return mesh_or_scene

    def log_mesh_or_scene(
        self,
        entity_path: str,
        mesh_or_scene: trimesh.Trimesh,
        material: Optional[urdf_parser.Material],
    ) -> None:
        if isinstance(mesh_or_scene, trimesh.Scene):
            for i, mesh in enumerate(self.scene_to_trimeshes(mesh_or_scene)):
                self.apply_material_and_log(entity_path + f"/{i}", mesh, material)
        else:
            self.apply_material_and_log(entity_path, mesh_or_scene, material)

    def scene_to_trimeshes(self, scene: trimesh.Scene) -> List[trimesh.Trimesh]:
        trimeshes = []
        scene_dump = scene.dump()
        geometries = [scene_dump] if not isinstance(scene_dump, list) else scene_dump
        for geometry in geometries:
            if isinstance(geometry, trimesh.Trimesh):
                trimeshes.append(geometry)
            elif isinstance(geometry, trimesh.Scene):
                trimeshes.extend(self.scene_to_trimeshes(geometry))
        return trimeshes

    def apply_material_and_log(
        self,
        entity_path: str,
        mesh: trimesh.Trimesh,
        material: Optional[urdf_parser.Material],
    ) -> None:
        if material is not None and not isinstance(
            mesh.visual, trimesh.visual.texture.TextureVisuals
        ):
            if material.color is not None:
                mesh.visual = trimesh.visual.ColorVisuals()
                mesh.visual.vertex_colors = material.color.rgba
            elif material.texture is not None:
                texture_path = self.resolve_ros_path(material.texture.filename)
                mesh.visual = trimesh.visual.texture.TextureVisuals(
                    image=Image.open(texture_path)
                )
        self.log_trimesh(entity_path, mesh)

    def log_trimesh(self, entity_path: str, mesh: trimesh.Trimesh) -> None:
        vertex_colors = albedo_texture = vertex_texcoords = None

        if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
            vertex_colors = mesh.visual.vertex_colors
        elif isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            trimesh_material = mesh.visual.material
            if mesh.visual.uv is not None:
                vertex_texcoords = mesh.visual.uv
                vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]
            if isinstance(trimesh_material, trimesh.visual.material.PBRMaterial):
                if trimesh_material.baseColorTexture is not None:
                    albedo_texture = self.pil_image_to_albedo_texture(
                        trimesh_material.baseColorTexture
                    )
                elif trimesh_material.baseColorFactor is not None:
                    vertex_colors = trimesh_material.baseColorFactor
            elif isinstance(trimesh_material, trimesh.visual.material.SimpleMaterial):
                if trimesh_material.image is not None:
                    albedo_texture = self.pil_image_to_albedo_texture(
                        trimesh_material.image
                    )
                else:
                    vertex_colors = mesh.visual.to_color().vertex_colors

        # Split the entity path to add "agent"
        parts = entity_path.split("/")
        if len(parts) > 1:
            log_path = "/".join([parts[0], "agent"] + parts[1:])
        else:
            log_path = "/".join([parts[0], "agent"])

        rr.log(
            log_path,
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                vertex_colors=vertex_colors,
                albedo_texture=albedo_texture,
                vertex_texcoords=vertex_texcoords,
            ),
            timeless=True,
        )

    @staticmethod
    def resolve_ros_path(path_str: str) -> str:
        """Resolve a ROS path to an absolute path."""
        if path_str.startswith("package://"):
            path = pathlib.Path(path_str)
            package_name = path.parts[1]
            relative_path = pathlib.Path(*path.parts[2:])
            package_path = URDFVisualizer.resolve_ros1_package(
                package_name
            ) or URDFVisualizer.resolve_ros2_package(package_name)
            if package_path is None:
                raise ValueError(f"Could not resolve {path}.")
            return str(package_path / relative_path)
        elif path_str.startswith("file://"):
            return path_str[len("file://") :]
        else:
            return path_str

    @staticmethod
    def resolve_ros2_package(package_name: str) -> Optional[str]:
        try:
            import ament_index_python

            try:
                return ament_index_python.get_package_share_directory(package_name)
            except ament_index_python.packages.PackageNotFoundError:
                return None
        except ImportError:
            return None

    @staticmethod
    def resolve_ros1_package(package_name: str) -> Optional[str]:
        try:
            import rospkg

            try:
                return rospkg.RosPack().get_path(package_name)
            except rospkg.ResourceNotFound:
                return None
        except ImportError:
            return None

    @staticmethod
    def pil_image_to_albedo_texture(image: Image.Image) -> np.ndarray:
        """Convert a PIL image to an albedo texture."""
        albedo_texture = np.asarray(image)
        if albedo_texture.ndim == 2:
            albedo_texture = np.stack([albedo_texture] * 3, axis=-1)
        return albedo_texture

    def set_time_sequence(self, time_line: str, time_sequence: np.ndarray) -> None:
        """Set the time sequence for the visualizer."""
        rr.set_time_sequence(time_line, time_sequence)

    def set_time_seconds(self, observation_time: float) -> None:
        """Set the time in seconds for the visualizer."""
        rr.set_time_seconds("real_clock", observation_time)


# Example usage:
if __name__ == "__main__":
    visualizer = URDFVisualizer("path/to/urdf/file.urdf", "test_prefix")
    visualizer.set_root_pose(np.array([0, 0, 0]), np.array([0, 0, 0]))
    visualizer.visualize(observation_time=time.time())

    # Update root pose and re-visualize
    visualizer.set_root_pose(np.array([1, 1, 1]), np.array([0.5, 0.5, 0.5]))
    visualizer.visualize(observation_time=time.time())

    # Update a specific joint and re-visualize
    visualizer.update_joint(
        "joint_name",
        translation=np.array([0.1, 0.2, 0.3]),
        rotation=np.array([0.1, 0.2, 0.3]),
    )
    visualizer.visualize(observation_time=time.time())
