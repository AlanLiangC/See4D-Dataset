import os
import torch, os, imageio
from torchvision.transforms import v2
from einops import rearrange
import torchvision
from PIL import Image
import numpy as np
import random
from pathlib import Path
import loguru

class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        self.logger = loguru.logger()
        self.data_root = data_root
        self.slpit = split
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.prepare_dataset()
        
    def prepare_dataset(self):
        path_list = []
        cam_cats = os.listdir(Path(self.data_root) / self.slpit)
        for _cam in cam_cats:
            cam_path = Path(self.data_root) / self.slpit / _cam
            path_list.extend([str(cam_path) + f'/{scene_name}' for scene_name in  os.listdir(cam_path)])
        self.path = path_list

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = np.array(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames


    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def __getitem__(self, idx):
        try:
            path = os.path.join(self.path[idx], 'videos')
            video_path_list = os.listdir(path)
            video_path = os.path.join(path, video_path_list[random.randint(0,9)])
            video = self.load_video(video_path)
        except:
            self.logger.info(f'Video {path} is broken.')
            video_idx = random.randint(0, self.__len__()-1)
            return self.__getitem__(video_idx)
        return video
    
    def __len__(self):
        return len(self.path)

if __name__ == '__main__':
    dataset = TextVideoDataset(
        data_root='data/MultiCamVideo-Dataset',
        split='train',
        frame_interval=10,
        num_frames=8
    )
    item = dataset.__getitem__(0)
