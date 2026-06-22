import json
import os
import random

from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import searchForMaxIteration

__all__ = ["HierarchicalTetScene", "HierarchicalTetrahedraModel"]


def __getattr__(name):
    if name == "HierarchicalTetrahedraModel":
        from scene.hierarchical_tetrahedra_model import HierarchicalTetrahedraModel

        return HierarchicalTetrahedraModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _write_point_cloud_ply(path, point_cloud):
    if point_cloud is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    points = point_cloud.points
    normals = point_cloud.normals
    colors = point_cloud.colors
    if colors.dtype != "uint8":
        colors = (colors * 255.0).clip(0, 255).astype("uint8")
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, normal, color in zip(points, normals, colors):
            f.write(
                f"{point[0]} {point[1]} {point[2]} "
                f"{normal[0]} {normal[1]} {normal[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


class HierarchicalTetScene:
    def __init__(
        self,
        args: ModelParams,
        tets,
        load_iteration=None,
        shuffle=False,
        resolution_scales=[1.0],
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.tets = tets
        self.train_cameras = {}
        self.test_cameras = {}
        self.low_scale_init = False

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        scene_info = sceneLoadTypeCallbacks["MRI"](
            args.source_path,
            args.white_background,
            args.eval,
            args.init_mesh,
        )

        if not self.loaded_iter:
            input_ply_path = os.path.join(self.model_path, "input.ply")
            if scene_info.ply_path and os.path.exists(scene_info.ply_path):
                with open(scene_info.ply_path, "rb") as src_file, open(input_ply_path, "wb") as dest_file:
                    dest_file.write(src_file.read())
            else:
                _write_point_cloud_ply(input_ply_path, scene_info.point_cloud)

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for idx, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(idx, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras,
                resolution_scale,
                args,
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras,
                resolution_scale,
                args,
            )

        if self.loaded_iter:
            self.tets.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    f"iteration_{self.loaded_iter}",
                    "point_cloud.ply",
                )
            )
        else:
            self.tets.create_from_tetra(
                scene_info.tetrahedra,
                self.cameras_extent,
                low_scale_init=self.low_scale_init,
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.tets.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
