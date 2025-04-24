import torch
# import sys
# sys.path.append('/data/dylu/project/See4D-Dataset')
from ..submodules.video_depth_anything.video_depth import VideoDepthAnything
from ..dataset.syncam4d import Syncam4DDataset
from ..dataset.processer import image_process
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule


class VDA_Inferencer:
    def __init__(self, encoder='vits'):
        self.encoder = encoder
        self.model = self.init_model()

    def init_model(self):

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        model = VideoDepthAnything(**model_configs[self.encoder])
        model.load_state_dict(torch.load(f'./pretrained_models/VDA/video_depth_anything_{self.encoder}.pth', map_location='cpu'), strict=True)
        model = model.to('cuda').eval()
        return model
    

    def inference_sample(self, frames, target_fps): # TODO: Only for images as input now
        depths, fps = self.model.infer_video_depth(frames, target_fps, input_size=200, device='cuda')
        return depths
    

class Fast3R_Inferencer:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.lit_module = self.init_model()

    def init_model(self):
        model = Fast3R.from_pretrained(self.ckpt_path)  # If you have networking issues, try pre-download the HF checkpoint dir and change the path here to a local directory
        # save_dir = "pretrained_models/fast3r"
        # model.save_pretrained(save_dir)
        # model = Fast3R.from_pretrained(self.ckpt_path)
        model = model.to(self.device)
        lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
        model, lit_module = model.eval(), lit_module.eval()
        return model, lit_module

    def init_dataset(self):
        dataset = Syncam4DDataset(root='/dataset/yyzhao/Sync4D')
        return dataset

    def inference_sample(self, index=None, images=None):
        assert not ((index is None) and (images) is None)
        if index is not None:
            rel = self.inference_from_item(index)
        elif images is not None:
            rel = self.inference_from_images(images)
        else:
            raise NotImplementedError()
        return rel

    def inference_from_item(self, index):
        if not hasattr(self, 'dataset'):
            self.dataset = self.init_dataset()
        images = self.dataset.__getitem__(index) # N 3 H W
        rel = self.inference_from_images(images)
        return rel

    def inference_from_images(self, images): # TODO: Only for image.path list now. like: ["path/to/image1.jpg", "path/to/image2.jpg", "path/to/image3.jpg"]
        images = image_process.load_images(images, size=512, verbose=True)

        # --- Run Inference ---
        output_dict, profiling_info = inference(
            images,
            self.model,
            self.device,
            dtype=torch.float32,  # or use torch.bfloat16 if supported
            verbose=True,
            profiling=True,
        )

        # --- Estimate Camera Poses ---
        # This step estimates the camera-to-world (c2w) poses for each view using PnP.
        poses_c2w_batch, estimated_focals = MultiViewDUSt3RLitModule.estimate_camera_poses(
            output_dict['preds'],
            niter_PnP=100,
            focal_length_estimation_method='first_view_from_global_head'
        )
        # poses_c2w_batch is a list; the first element contains the estimated poses for each view.
        camera_poses = poses_c2w_batch[0]

        # Print camera poses for all views.
        for view_idx, pose in enumerate(camera_poses):
            print(f"Camera Pose for view {view_idx}:")
            print(pose.shape)  # np.array of shape (4, 4), the camera-to-world transformation matrix

        # --- Extract 3D Point Clouds for Each View ---
        # Each element in output_dict['preds'] corresponds to a view's point map.
        for view_idx, pred in enumerate(output_dict['preds']):
            point_cloud = pred['pts3d_in_other_view'].cpu().numpy()
            print(f"Point Cloud Shape for view {view_idx}: {point_cloud.shape}")  

        return output_dict, profiling_info 

if __name__ == "__main__":
    # test depth
    # vda_egine = VDA_Inferencer()

    # test fast3r
    fast3r_egine = Fast3R_Inferencer(ckpt_path='pretrained_models/fast3r')
    fast3r_egine.inference_sample(index=0)