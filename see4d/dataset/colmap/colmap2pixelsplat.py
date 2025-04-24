from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from .read_write_model import Camera, Image, read_model, write_model

import ipdb
from typing import Tuple, List

def read_colmap_model(
    path: Path,
    device: torch.device = torch.device("cpu"),
    return_points: bool = False,
    return_cam_points: bool = False,
    norm_intrinsics: bool = True,
    ext="",
) -> Tuple[
    Float[Tensor, "frame 4 4"],  # extrinsics
    Float[Tensor, "frame 3 3"],  # intrinsics
    List[str],  # image names
]:
    cameras, images, points = read_model(path, ext=ext)
    # ipdb.set_trace()
    all_extrinsics = []
    all_intrinsics = []
    all_image_names = []

    for image in images.values():
        try:
            camera: Camera = cameras[image.camera_id]
        except:
            assert len(cameras) == 1, "The camera model should be either 1 or the same with image camera_id"
            camera = cameras[1]
        # ipdb.set_trace()
        # Read the camera intrinsics.
        intrinsics = torch.eye(3, dtype=torch.float32, device=device)
        if camera.model == "SIMPLE_PINHOLE" or camera.model == "SIMPLE_RADIAL":
            fx, cx, cy = camera.params[:3]
            fy = fx
        elif camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "OPENCV":
            fx, fy, cx, cy = camera.params[:4]
            
        intrinsics[0, 0] = fx
        intrinsics[1, 1] = fy
        intrinsics[0, 2] = cx
        intrinsics[1, 2] = cy
        if norm_intrinsics:
            intrinsics[0] /= camera.width
            intrinsics[1] /= camera.height
        all_intrinsics.append(intrinsics)

        # Read the camera extrinsics.
        qw, qx, qy, qz = image.qvec
        w2c = torch.eye(4, dtype=torch.float32, device=device)
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
        w2c[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
        w2c[:3, 3] = torch.tensor(image.tvec, dtype=torch.float32, device=device)
        extrinsics = w2c #w2c.inverse() # NOTE the extrinsics should be w2c to be used in pixelsplat
        all_extrinsics.append(extrinsics)

        # Read the image name.
        all_image_names.append(image.name)
        
    if return_points:
        return all_image_names, points

    if return_cam_points:
        return torch.stack(all_extrinsics), torch.stack(all_intrinsics), all_image_names, points
    
    return torch.stack(all_extrinsics), torch.stack(all_intrinsics), all_image_names


def write_colmap_model(
    path: Path,
    extrinsics: Float[Tensor, "frame 4 4"],
    intrinsics: Float[Tensor, "frame 3 3"],
    image_names: List[str],
    image_shape: Tuple[int, int],
) -> None:
    h, w = image_shape

    # Define the cameras (intrinsics).
    cameras = {}
    for index, k in enumerate(intrinsics):
        id = index + 1

        # Undo the normalization we apply to the intrinsics.
        k = k.detach().clone()
        k[0] *= w
        k[1] *= h

        # Extract the intrinsics' parameters.
        fx = k[0, 0]
        fy = k[1, 1]
        cx = k[0, 2]
        cy = k[1, 2]

        cameras[id] = Camera(id, "PINHOLE", w, h, (fx, fy, cx, cy))

    # Define the images (extrinsics and names).
    images = {}
    for index, (c2w, name) in enumerate(zip(extrinsics, image_names)):
        id = index + 1

        # Convert the extrinsics to COLMAP's format.
        w2c = c2w.inverse().detach().cpu().numpy()
        qx, qy, qz, qw = R.from_matrix(w2c[:3, :3]).as_quat()
        qvec = np.array((qw, qx, qy, qz))
        tvec = w2c[:3, 3]
        images[id] = Image(id, qvec, tvec, id, name, [], [])

    path.mkdir(exist_ok=True, parents=True)
    write_model(cameras, images, {}, path)
