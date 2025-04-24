import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import torch
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
import multiprocessing
import shutil

import sys
sys.path.append('/data/dylu/project/See4D-Dataset')
from see4d.dataset.colmap.colmap2pixelsplat import read_colmap_model
from see4d.dataset.syncam4d import Syncam4DDataset
# from .colmap.colmap2pixelsplat import read_colmap_model

def extrinsics_to_quaternion(extrinsics):
    # ipdb.set_trace()
    rotation_matrix = extrinsics[:, :3]
    translation_vector = extrinsics[:, 3]
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
    quaternion = np.roll(quaternion, shift=1)  # [qw, qx, qy, qz]
    return quaternion, translation_vector

def read_image(image):
    image = Image.open(BytesIO(image.numpy().tobytes()))
    return image


# Load your camera parameter tensor (Nx18)
# data_dict = torch.load(args.data)[0]  # Replace with your actual data loading

def build_colmap_format(dataset, args, item_idx, save_dir_full, width, height):
    data_dict = dataset.__getitem_dict__(item_idx)
    video = data_dict['video']
    for frame_idx in range(video.shape[0]):
        save_dir = os.path.join(save_dir_full, f"{data_dict['id']}")
        image_dir = os.path.join(save_dir, "images")
        camera_dir = os.path.join(save_dir, "sparse/0")
        if args.remove_sparse:
            if os.path.exists(camera_dir):
                try:
                    shutil.rmtree(camera_dir)
                except:
                    tqdm.write(f"Failed to remove {camera_dir}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(camera_dir, exist_ok=True)

        camera_parameters = data_dict['cameras']

        N = camera_parameters.shape[0]
        image_filenames = [f'frame_{i:04d}.png' for i in range(N)]  # Replace with your actual filenames

        image_list = []
        valid = True
        for i in range(N):
            image = Image.fromarray(video[i])
            if not image.size == (width, height):
                valid = False
                tqdm.write(f"Example {data_dict['key']}, Image size does not match the specified width and height")
                break
            else:
                image_list.append(image)
        
        if not valid:
            continue

        for i in range(N):
            if args.no_save_image:
                continue 
            image_list[i].save(os.path.join(image_dir, image_filenames[i]))
            
        # Prepare cameras.txt
        with open(os.path.join(camera_dir, 'cameras.txt'), 'w') as f:
            for i in range(1):
                ## only one is ok
                fx, fy, cx, cy = camera_parameters[i, :4]
                fx = fx * width
                fy = fy * height
                cx = cx * width
                cy = cy * height
                f.write(f"{i + 1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")
                f.write("\n")

        # Prepare images.txt
        with open(os.path.join(camera_dir, 'images.txt'), 'w') as f:
            for i in range(N):
                fx, fy, cx, cy = camera_parameters[i, :4]
                extrinsics = camera_parameters[i, 6:].reshape(3, 4)
                quaternion, translation = extrinsics_to_quaternion(extrinsics)
                qw, qx, qy, qz = quaternion
                tx, ty, tz = translation
                ## NOTE here the value before image name is the camera id, which should always be 1 for single camera
                f.write(f"{i + 1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {image_filenames[i]}\n\n") ## need to add one empty line after each camera
        # create an empty points3D.txt
        with open(os.path.join(camera_dir, 'points3D.txt'), 'w') as f:
            pass
        
        if args.check:
            cams, intrinsics, image_names = read_colmap_model(camera_dir, device=torch.device("cpu"), ext=".txt")
            fx = intrinsics[:, 0, 0]
            fy = intrinsics[:, 1, 1] 
            cx = intrinsics[:, 0, 2]
            cy = intrinsics[:, 1, 2]
            ins_tensor = torch.zeros((len(fx), 6))
            ins_tensor[:, 0] = fx
            ins_tensor[:, 1] = fy
            ins_tensor[:, 2] = cx
            ins_tensor[:, 3] = cy
            ext_matrix = cams[:,:3].reshape(len(fx), -1)
            # camera = torch.cat([ins_tensor, ext_matrix], dim=-1)
            camera = ext_matrix
            image_name2id = {name: i for i, name in enumerate(image_names)}
            sorted_image_names = sorted(image_names,)
            sortedids = [image_name2id[name] for name in sorted_image_names]
            camera = camera[sortedids]
            # ipdb.set_trace()
    
            if not torch.allclose(camera, data_dict['cameras'][:, 6:], atol=1e-05):
                tqdm.write(f"Failed check Extrinsics for {data_dict['key']}, max diff: {torch.max(torch.abs(camera - data_dict['cameras'][:, 6:]))}")
            else:
                tqdm.write(f"Check Extrinsics passed for {data_dict['key']}")
                
        tqdm.write(f"Processed {data_dict['id']}")

def process_data(dataset, args, chunk, worker_id):
    progress_bar = tqdm(chunk, desc=f"Worker {worker_id}")
    for data in chunk:
        item_idx, save_dir, width, height = data
        build_colmap_format(dataset, args, item_idx, save_dir, width, height)
        progress_bar.update(1)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, default="camera_parameters.npy", help="Path to the camera parameters")
    parser.add_argument("--save_dir", type=str, default="re10k-colmap-format", help="Path to the camera parameters")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--mp", type=int, default=32)
    parser.add_argument("--check", action="store_true", help="check if the saved file match the torch file")
    parser.add_argument("--no_save_image", action="store_true", help="save image files")
    parser.add_argument("--remove_sparse", action="store_true", help="remove existing sparse folder")
    args = parser.parse_args()
    
    dataset = Syncam4DDataset(root=args.data, mini=True)
    data_list = [i for i in range(dataset.__len__())]
    # data_list = sorted(glob(os.path.join(args.data, "*.mp4")))
    num_workers = min(cpu_count(), len(data_list), args.mp)
    data_args = [(data, args.save_dir, args.width, args.height) for data in data_list]

    if num_workers > 1:
        print(f"Using {num_workers} workers")
        chunk_size = len(data_args) // num_workers
        chunks = [data_args[i:i + chunk_size] for i in range(0, len(data_args), chunk_size)]
        processes = []
        for worker_id in range(num_workers):
            p = multiprocessing.Process(target=process_data, args=(dataset, args, chunks[worker_id], worker_id))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    else:
        process_data(dataset, args, data_args, 0)