import os
import json
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from decord import VideoReader

VIEWS = {
    'syncam4d': 36,
    'kubric4d': 16,
    'obj4d-10k': 18,
}

def load_video(video_dir, id, target_resolution=(256, 384), num_videos=36, background_color="random"):
    # Construct paths
    video_path = os.path.join(video_dir, 'videos', id + ".mp4")
    # metadata_path = os.path.join(video_dir, 'videos', id + ".json")

    # with open(metadata_path, "r") as f:
    #     metadata = json.load(f)

    vr = VideoReader(video_path)
    num_frames = len(vr)
    # num_videos = metadata["num_videos"]
    if num_videos == 36:
        frame_indices = list(range(0, num_frames, 2))
    else:
        frame_indices = list(range(0, num_frames))
    video = vr.get_batch(frame_indices).asnumpy()  # shape: [f, H, W, C], 0-255
    video = torch.from_numpy(video).float() / 255.0  # now in [0,1]

    mask_path = video_path.split(".")[0] + '_mask.mp4'
    if os.path.exists(mask_path):
        mvr = VideoReader(mask_path)
        masks = mvr.get_batch(frame_indices).asnumpy()
        masks = torch.from_numpy(masks).float() / 255.0
        if background_color == "white":
            bg_color = torch.ones(1,1,1,3, dtype=torch.float32)
        elif background_color == "black":
            bg_color = torch.zeros(1,1,1,3, dtype=torch.float32)
        else:
            bg_color = torch.randint(0, 256, size=(1,1,1,3), dtype=torch.float32) / 255.0 
        video =  masks * video + (1 - masks) * bg_color

    num_frames, H, W, C = video.shape

    # Process based on dataset
    if num_videos == 36:  # syncam4d
        # For syncam4d, reshape frames to form videos.
        video = video.reshape(num_frames, 6, H//6, 6, W//6, C)
        video = rearrange(video, "f m h n w c -> (m n) f c h w")
    else:
        video_chunks = video.chunk(num_videos, dim=2)  # along width
        video = torch.stack(video_chunks, dim=0)  # [v, f, H, W, C]
        video = rearrange(video, "v f h w c -> v f c h w")
    
    # print(f'Loaded {video.shape[0]} videos with {video.shape[1]} frames each. Resolution: {video.shape[3]}x{video.shape[4]}')

    # Assume video has shape [v, f, c, H, W]
    v, f, c, H, W = video.shape
    # Merge view and frame dimensions
    video_merged = video.view(v * f, c, H, W)
    # Resize to target resolution (new_H, new_W)
    video_resized = F.interpolate(video_merged, size=target_resolution, mode="bilinear", align_corners=False)
    # Reshape back to [v, f, c, new_H, new_W]
    video = video_resized.view(v, f, c, target_resolution[0], target_resolution[1])

    return video

# --- Base Dataset Class for 4D Data ---
class Base4DDataset(Dataset):
    def __init__(self, root, dataset_name, split="train", num_frames_sample=8, target_resolution=(256,384), mini=False):
        """
        root: base directory (e.g. "/dataset/yyzhao/Sync4D")
        dataset_name: one of 'syncam4d', 'kubric4d', or 'obj4d-10k'
        split: "train" or "test" (uses index_train.json or index_test.json)
        num_frames_sample: number of contiguous frames to sample per view
        target_resolution: tuple (H, W) to resize frames
        """
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        index_file = f"index_{split}.json"
        self.index_file = os.path.join(root, dataset_name, index_file)
        with open(self.index_file, 'r') as f:
            self.ids = json.load(f) # list of sequence ids
        if mini:
            self.ids = self.ids[:10]
        self.num_frames_sample = num_frames_sample
        self.target_resolution = target_resolution
        self.num_videos = VIEWS[dataset_name]

    def __len__(self):
        return len(self.ids)

    def __getitem_dict__(self, idx):
        data_dict = {}

        seq = self.ids[idx]
        # video
        video_dir = os.path.join(self.root, self.dataset_name)
        video = load_video(video_dir, seq, self.target_resolution, self.num_videos)
        video = video[:,0,...].permute(0,2,3,1)
        data_dict['video'] = (video.cpu().numpy()*255).astype(np.uint8)
        # camera
        pose_path = os.path.join(video_dir, 'videos', seq + ".json")
        with open(pose_path, 'rb') as f:
            pose_dict = json.load(f) # list of sequence ids
        data_dict['cameras'] = np.array(pose_dict['cameras'])
        data_dict['id'] = seq
        # v, f, c, H, W = video.shape
        return data_dict

    def __getitem__(self, idx):

        seq = self.ids[idx]
        # load & cache

        video_dir = os.path.join(self.root, self.dataset_name)
        video = load_video(video_dir, seq, self.target_resolution, self.num_videos)
        video = video[:,0,...].permute(0,2,3,1)
        # v, f, c, H, W = video.shape
        return (video.cpu().numpy()*255).astype(np.uint8)


class Syncam4DDataset(Base4DDataset):
    def __init__(self, root, split="train", num_frames_sample=8, target_resolution=(512,512), mini=False):
        # For syncam4d, assume 36 views and group_size=9.
        super().__init__(root, 'syncam4d', split, num_frames_sample, target_resolution=target_resolution, mini=mini)