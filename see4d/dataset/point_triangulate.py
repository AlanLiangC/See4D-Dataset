#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil
from glob import glob
from tqdm import tqdm
import torch
from PIL import Image
from io import BytesIO
import wandb
from datetime import datetime
import subprocess
import time
import multiprocessing
from multiprocessing import Pool, cpu_count
import json

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--image_path", "-i", required=True, type=str, help="path to images, it can be different from source_path")
parser.add_argument("--index_path", required=True, type=str)
# parser.add_argument("--output", "-o", required=True, type=str)
parser.add_argument("--stage", type=str, default="train")
parser.add_argument("--camera", default="PINHOLE", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
parser.add_argument("--mp", type=int, default=32)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0


def run_command(command):
    try:
        result = subprocess.run(command, text=True, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{' '.join(command)}' failed with return code {e.returncode}")
        return e.returncode

def read_camera_txt(camera_txt):
    # Read the camera file
    with open(camera_txt, 'r') as file:
        lines = file.readlines()

    # Remove any empty lines and new line characters
    lines = [line.strip() for line in lines if line.strip()]

    # Parse the camera parameters
    for line in lines:
        camera_id, model, width, height, fx, fy, cx, cy = line.split()
        camera_id = int(camera_id)
        width = int(width)
        height = int(height)
        fx = float(fx)
        fy = float(fy)
        cx = float(cx)
        cy = float(cy)

        print(f"Camera txt: {camera_txt}: Model: {model}, Width: {width}, Height: {height}, fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
    
    return camera_id, model, width, height, fx, fy, cx, cy




def colmap_point_triangulate(image_dir, source_path):
    '''
    colmap feature_extractor \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH/images

    colmap sequential_matcher \
        --database_path $PROJECT_PATH/database.db \
        --SiftMatching.use_gpu 1

    colmap point_triangulator \
        --database_path $PROJECT_PATH/database.db \
        --image_path $PROJECT_PATH/images \
        --input_path $PROJECT_PATH/sparse/0 \
        --output_path $PROJECT_PATH/sparse/0
    '''
    
    image_path = os.path.join(image_dir, os.path.basename(source_path))
    if os.path.exists(os.path.join(source_path, "database.db")):
        os.remove(os.path.join(source_path, "database.db"))
    
    ## Load camera model
    camera_id, camera_model, width, height, fx, fy, cx, cy = read_camera_txt(os.path.join(source_path, "sparse/0/cameras.txt"))
        
    # Feature extraction
    feat_extraction_cmd = [
        colmap_command, "feature_extractor",
        "--database_path", os.path.join(source_path, "database.db"),
        "--image_path", os.path.join(image_path, "images"),
        "--SiftExtraction.use_gpu", str(use_gpu),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1" ,
        "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}",
    ]
    exit_code = run_command(feat_extraction_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed")
    
    # Feature matching
    feat_matching_cmd = [
        colmap_command, "sequential_matcher",
        "--database_path", os.path.join(source_path, "database.db"),
        "--SiftMatching.use_gpu", str(use_gpu)
    ]
    exit_code = run_command(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed")

    # Bundle adjustment
    point_triangulate_cmd = [
        colmap_command, "point_triangulator",
        "--database_path", os.path.join(source_path, "database.db"),
        "--image_path", os.path.join(image_path, "images"),
        "--input_path", os.path.join(source_path, "sparse/0"),
        "--output_path", os.path.join(source_path, "sparse/0"),
    ]
    exit_code = run_command(point_triangulate_cmd)
    if exit_code != 0:
        logging.error(f"point_triangulator failed")


def process_data(image_path, data_list, worker_id):
    num_samples, success = 0, 0
    num_processed = 0
    if worker_id == 0:
        wandb.init(project="re10k-colmap", name=f"index{os.path.basename(args.index_path)}-{timestamp_str}-0", config=args, id=f"index{os.path.basename(args.index_path)}-{timestamp_str}-0")
    for work_dir in data_list:
        num_samples += 1
        start_time = time.time()
        tqdm.write(f"Processing {os.path.basename(work_dir)}")
        
        if not os.path.exists(os.path.join(work_dir, "sparse/0/images.txt")):
            txt_path = os.path.join(work_dir, "sparse/0/images.txt")
            tqdm.write(f"{ txt_path } does not exist")
            continue
        try:
            # if points3D.bin exists, skip
            if os.path.exists(os.path.join(work_dir, "sparse/0/points3D.bin")):
                num_processed += 1
                tqdm.write(f"Skipping {os.path.basename(work_dir)}, already processed")
                continue
            colmap_point_triangulate(image_path, work_dir)   
            end_time = time.time()
            success += 1
            if worker_id == 0:
                wandb.log({"time": end_time - start_time, 
                           "success": success, 
                           "num_processed": num_processed,
                           "all_samples": num_samples})
            progress_bar.set_postfix_str(f"time: {end_time - start_time:.2f}, success number {success}, success rate: {success / num_samples:.2f}")
        except Exception as e:
            logging.error(f"Error processing {os.path.basename(work_dir)}: {e}")
            continue
        
if __name__ == "__main__":
    # get timestep str
    current_timestamp = datetime.now()
    # Convert to string format
    timestamp_str = current_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
    # torch_list = os.listdir(os.path.join(args.source_path, args.stage))
    torch_list = sorted(os.listdir(args.source_path))
    num_samples = 0
    success = 0
    num_workers = min(cpu_count(), len(torch_list), args.mp)
    progress_bar = tqdm(torch_list)
    # data_list = [os.path.join(args.source_path, args.stage, data) for data in torch_list.keys() if os.path.exists(os.path.join(args.source_path, args.stage, data, "sparse/0/images.txt"))]
    data_list = [os.path.join(args.source_path, data) for data in torch_list.keys()]
    image_path = args.source_path

    # for data in tqdm(data_list):
    #     build_colmap_format(data)
    if num_workers > 1:
        print(f"Using {num_workers} workers")
        chunk_size = len(data_list) // num_workers
        chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
        processes = []
        for worker_id in range(num_workers):
            p = multiprocessing.Process(target=process_data, args=(image_path, chunks[worker_id], worker_id))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    else:
        process_data(image_path, data_list, 0)