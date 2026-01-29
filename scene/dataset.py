from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                R,T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask=None
                control_vec = None
                depth = None
                camera_idx = 0
                sample_idx = 0
            except:
                caminfo = self.dataset[index]
                image = caminfo.image
                R = caminfo.R
                T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                control_vec = caminfo.control_vec if hasattr(caminfo, 'control_vec') else None
                mask = caminfo.mask
                camera_idx = caminfo.camera_idx
                depth = caminfo.depth if hasattr(caminfo, 'depth') else None
                sample_idx = caminfo.sample_idx if hasattr(caminfo, 'sample_idx') else 0
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=None,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              control_vec=control_vec,mask=mask,camera_idx=camera_idx,depth=depth,sample_idx=sample_idx)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
