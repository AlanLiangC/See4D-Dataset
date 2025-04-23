import torch
import sys
sys.path.append('/home/alan/AlanLiang/Projects/AlanLiang/See4D-Dataset')
from see4d.submodules.video_depth_anything.video_depth import VideoDepthAnything
from fast3r.dust3r.utils.image import load_images
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
        depths, fps = self.engine.infer_video_depth(frames, target_fps, input_size=200, device='cuda')
        return depths
    

class Fast3R_Inferencer:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.model, self.lit_module = self.init_model()

    def init_model(self):
        # model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")  # If you have networking issues, try pre-download the HF checkpoint dir and change the path here to a local directory
        model = Fast3R.from_pretrained(self.ckpt_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
        model, lit_module = model.eval(), lit_module.eval()
        return model, lit_module

    def inference_sample(self):
        pass

if __name__ == "__main__":
    # test depth
    vda_egine = VDA_Inferencer()

    # test fast3r
    fast3r_egine = Fast3R_Inferencer(ckpt_path='pretrained_models/fast3r/model.safetensors')