import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from loguru import logger
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.append('/data/dylu/project/See4D-Dataset')
from see4d.dataset.syncam4d import Syncam4DDataset, Kubric4DDataset
from see4d.submodules.ben2 import BEN_Base, images2mp4

class BEN2_Inferencer:
    def __init__(self, ckpt_path, max_workers=1):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model()
        self.dataset = self.init_dataset()

    def init_model(self):
        model = BEN_Base.from_pretrained(self.ckpt_path)  # If you have networking issues, try pre-download the HF checkpoint dir and change the path here to a local directory
        model = model.to(self.device).eval()
        return model
    
    def init_dataset(self):
        dataset = Kubric4DDataset(root='/dataset/yyzhao/Sync4D')
        return dataset

    def inference_sample(self, index=None, images=None):
        assert not ((index is None) and (images) is None)
        if index is not None:
            self.inference_from_item(index)
        elif images is not None:
            rel = self.inference_from_images(images)
        else:
            raise NotImplementedError()
        
    def parallel_execute(self, tasks):
        for task in tasks:
            self.executor.submit(self.inference_sample, task)

    def inference_from_item(self, index):
        if not hasattr(self, 'dataset'):
            self.dataset = self.init_dataset()
        data_dict = self.dataset.__getitem_video__(index) # N 3 H W
        video_path = data_dict['video_path']
        seq_id = data_dict['id']
        self.model.segment_video(
            video_path= str(video_path),
            output_path=f"./data/temp/{seq_id}", # Outputs will be saved as foreground.webm or foreground.mp4. The default value is "./"
            fps=0, # If this is set to 0 CV2 will detect the fps in the original video. The default value is 0.
            refine_foreground=False,  #refine foreground is an extract postprocessing step that increases inference time but can improve matting edges. The default value is False.
            batch=1,  # We recommended that batch size not exceed 3 for consumer GPUs as there are minimal inference gains. The default value is 1.
            print_frames_processed=True,  #Informs you what frame is being processed. The default value is True.
            webm = False, # This will output an alpha layer video but this defaults to mp4 when webm is false. The default value is False.
            rgb_value= (0, 255, 0) # If you do not use webm this will be the RGB value of the resulting background only when webm is False. The default value is a green background (0,255,0).
        )

    def gen_mask_for_sample(self, index=None, images=None, to_video=False):
        assert not ((index is None) and (images) is None)
        if index is not None:
            self.gen_mask_for_item(index, to_video)
        elif images is not None:
            pass # TODO: for images directly
        else:
            raise NotImplementedError()

    def gen_mask_for_item(self, index, to_video=False):
        if not hasattr(self, 'dataset'):
            self.dataset = self.init_dataset()
        data_dict = self.dataset.__getitem_video__(index, to_video) # N 3 H W
        for view_id, images in enumerate(data_dict['video']):
            images = [Image.fromarray(image)  for image in data_dict['video'][view_id]]
            _, masks = self.model.inference_mask_rgb(images)
            mask_list = [np.array(mask_image)>0 for mask_image in masks]
            mask_array = np.stack(mask_list).astype(np.bool_)
            # save_mask_path = Path(self.dataset.root) / self.dataset.dataset_name / 'mask'
            save_mask_path = Path('data/Sync4D') / self.dataset.dataset_name / 'mask'

            save_mask_path.mkdir(parents=True, exist_ok=True)
            np.save(save_mask_path / f"{data_dict['id']}_mask_{view_id}.npy", mask_array)
        logger.info(f"{index} / {self.dataset.__len__()} is saved.")
        # if to_video:
        #     output_path = Path(f"data/temp/{data_dict['id']}")
        #     output_path.mkdir(parents=True, exist_ok=True)
        #     foreground_video_path = output_path / f"{data_dict['id']}_foreground.mp4"
        #     images2mp4(foregrounds, output_path=str(foreground_video_path), fps=10)
        #     mask_video_path = output_path / f"{data_dict['id']}_mask.mp4"
        #     images2mp4(masks, output_path=str(mask_video_path), fps=10)

    def inference_from_item_parallel(self, index_list):
        for index in index_list:
            self.inference_sample(index)

if __name__ == "__main__":
    import multiprocessing
    from multiprocessing import Pool
    # test depth
    # vda_egine = VDA_Inferencer()

    # test fast3r
    ben2_egine = BEN2_Inferencer(ckpt_path='pretrained_models/ben2')
    num_index = [i for i in range(ben2_egine.dataset.__len__())][:200]
    for index in num_index:
        ben2_egine.gen_mask_for_sample(index)
    # ben2_egine.inference_sample(index = num_index[0])
    # ben2_egine.parallel_execute(num_index)
    # indexes = [0,10,20,30,40,50,60,70,80]
    # for i, index in tqdm(enumerate(indexes)):
    #     ben2_egine.gen_mask_for_item(index=index, to_video=True)
    # num_workers = 2
    # print(f"Using {num_workers} workers")
    # # chunk_size = ben2_egine.dataset.__len__() // num_workers
    # chunk_size = 200 // num_workers

    # chunks = [num_index[i:i + chunk_size] for i in range(0, len(num_index), chunk_size)]
    # processes = []
    # for worker_id in range(num_workers):
    #     p = multiprocessing.Process(target=inference_from_item_parallel, args=(ben2_egine, chunks[worker_id]))
    #     processes.append(p)
    #     p.start()

    # for p in processes:
    #     p.join()