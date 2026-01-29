import os
import json
import numpy as np 
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import NamedTuple

from utils.graphics_utils import focal2fov
from utils.general_utils import PILtoTorch


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    control_vec : np.array
    mask: np.array
    camera_idx: int
    depth: np.array
    sample_idx: int


class ToyArmDataset(Dataset):
    def __init__(self,
                 datadir, 
                 split="train",
                 train_cameras=None,
                 test_cameras=None,
                 video_cameras=None,
                 train_samples=None,
                 test_samples=None,
                 video_samples=None,
                 ratio=1.0,
                 preload_images=False):
        self.datadir = os.path.expanduser(datadir)
        self.split = split
        self.ratio = ratio
        self.preload_images = preload_images
        
        if train_cameras is None:
            # train_cameras = list(range(20, 30))
            # train_cameras = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            # train_cameras = [7, 10, 12, 15, 18, 20, 23, 28]
            train_cameras = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
            # train_cameras = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            # train_cameras = None
            
        if test_cameras is None:
            test_cameras = [7]
            # test_cameras = None
            
        if video_cameras is None:
            # video_cameras = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            # video_cameras = [7, 10, 12, 15, 18, 20, 23, 28]
            video_cameras = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10]
            # video_cameras = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23] 
            # video_cameras = None  
        
        if train_samples is None:
            # train_samples = [0, 1]
            train_samples = [0, 2, 4, 6, 8, 10, 12, 14]
            
        if test_samples is None:
            # test_samples = [8]
            test_samples = [3, 7, 11, 13]
        
        if video_samples is None:
            # video_samples = [0, 1]
            video_samples = [0, 2, 4, 6, 8, 10, 12, 14]
            
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras
        self.video_cameras = video_cameras
        
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.video_samples = video_samples
   
        self._load_metadata()

        self._filter_frames()

        if self.preload_images:
            print(f"[Warning] Preloading {len(self.frames)} images to memory...")
            self._preload_all_images()
            
    def _load_metadata(self):
        transforms_path = os.path.join(self.datadir, "transforms.json")
        
        if not os.path.exists(transforms_path):
            raise FileNotFoundError(f"transforms.json not found: {transforms_path}")
        
        print(f"Loading Toy Arm metadata from {transforms_path}...")
        with open(transforms_path, 'r') as f:
            data = json.load(f)
        
        if 'cameras' not in data or 'frames' not in data:
            raise ValueError("transforms.json must contain 'cameras' and 'frames' keys.")
        
        self.cameras_meta = data['cameras']
        self.frames_meta = data['frames']
        
        print(f"  Found {len(self.cameras_meta)} cameras")
        print(f"  Found {len(self.frames_meta)} frames")
        
        cam0 = self.cameras_meta[0]
        self.width = cam0['w']
        self.height = cam0['h']
        self.focal_x = cam0['fl_x']
        self.focal_y = cam0['fl_y']
        self.cx = cam0['cx']
        self.cy = cam0['cy']
        
        if self.ratio != 1.0:
            self.width = int(self.width * self.ratio)
            self.height = int(self.height * self.ratio)
            self.focal_x = self.focal_x * self.ratio
            self.focal_y = self.focal_y * self.ratio
            self.cx = self.cx * self.ratio
            self.cy = self.cy * self.ratio
            
        self.FovX = focal2fov(self.focal_x, self.width)
        self.FovY = focal2fov(self.focal_y, self.height)
        
        all_times = [frame['time'] for frame in self.frames_meta]
        self.min_time = min(all_times)
        self.max_time = max(all_times)
        self.time_range = self.max_time - self.min_time
        print(f"  Time range: [{self.min_time}, {self.max_time}]")

    def _filter_frames(self):
        self.frames = []

        for frame in self.frames_meta:
            cam_idx = frame['camera_idx']
            sample_idx = frame.get('sample_idx', None)

            if self.split == "train":
                if self.train_cameras is not None:
                    if cam_idx in self.train_cameras and sample_idx in self.train_samples:
                        self.frames.append(frame)
                else:     
                    if sample_idx in self.train_samples:
                        self.frames.append(frame)
                        
            elif self.split == "test":
                if self.test_cameras is not None:
                    if cam_idx in self.test_cameras and sample_idx in self.test_samples:
                        self.frames.append(frame)
                else:
                    if sample_idx in self.test_samples:
                        self.frames.append(frame)
                        
            elif self.split == "video":
                if self.video_cameras is not None:
                    if cam_idx in self.video_cameras and sample_idx in self.video_samples:
                        self.frames.append(frame)
                else:
                    if sample_idx in self.video_samples:
                        self.frames.append(frame)
                        
        self.frames.sort(key=lambda f: (f.get('sample_idx', 0), f['camera_idx'], f.get('time', 0)))
        print(f"  Filtered to {len(self.frames)} frames for {self.split} split")
    
        if self.split == "train":
            train_cams = sorted(set([f['camera_idx'] for f in self.frames]))
            train_samps = sorted(set([f.get('sample_idx', -1) for f in self.frames]))
            print(f"    Train cameras: {train_cams}")
            print(f"    Train samples: {train_samps if self.train_samples is not None else 'all'}")
        
        elif self.split == "test":
            test_cams = sorted(set([f['camera_idx'] for f in self.frames]))
            test_samps = sorted(set([f.get('sample_idx', -1) for f in self.frames]))
            print(f"    Test cameras: {test_cams}")
            print(f"    Test samples: {test_samps if self.test_samples is not None else 'all'}")
            
        elif self.split == "video":
            video_cams = sorted(set([f['camera_idx'] for f in self.frames]))
            video_samps = sorted(set([f.get('sample_idx', -1) for f in self.frames]))
            print(f"    Video cameras: {video_cams}")
            print(f"    Video samples: {video_samps if self.video_samples is not None else 'all'}")
        
        unique_cams = sorted({f['camera_idx'] for f in self.frames})
        self.poses = unique_cams if unique_cams else [0]      
                
    def _preload_all_images(self):
        self.preloaded_images = {}
        for idx in tqdm(range(len(self.frames)), desc=f"Preloading {self.split} images"):
            frame = self.frames[idx]
            image_path = os.path.join(self.datadir, frame['file_path'])
            image = Image.open(image_path)
            
            if self.ratio != 1.0:
                image = image.resize((self.width, self.height), Image.LANCZOS)
            
            image = PILtoTorch(image, None)
            self.preloaded_images[idx] = image
            
    def _load_image(self, index):
        if self.preload_images:
            return self.preloaded_images[index]
        
        frame = self.frames[index]
        image_path = os.path.join(self.datadir, frame['file_path'])

        try:
            image = Image.open(image_path)
            
            if self.ratio != 1.0:
                image = image.resize((self.width, self.height), Image.LANCZOS)
                
            image = PILtoTorch(image, None)
            return image
        except Exception as e:
            print(f"Failed to load image  {image_path}: {e}")
            return torch.zeros(3, self.height, self.width, dtype=torch.float32)
        
    def _load_depth(self, index):
        frame = self.frames[index]
        depth_path = frame.get('depth_file_path')
        
        if depth_path is None:
            return None
        
        depth_full_path = os.path.join(self.datadir, depth_path)
        
        if not os.path.exists(depth_full_path):
            return None
        
        depth_pil = Image.open(depth_full_path)
        depth = np.array(depth_pil, dtype=np.float32)
        
        if self.ratio != 1.0:
            depth_pil = depth_pil.resize((self.width, self.height), Image.NEAREST)
            depth = np.array(depth_pil, dtype=np.float32)
            
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        
        return depth_tensor
        
        
    def _get_camera_params(self, index):
        frame = self.frames[index]
        camera_idx = frame['camera_idx']
        camera_meta = self.cameras_meta[camera_idx]
        
        transform_matrix = np.array(camera_meta['transform_matrix'], dtype=np.float32)
        
        c2w = transform_matrix
        
        R_x_180 = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
        c2w = c2w @ R_x_180
        
        w2c = np.linalg.inv(c2w)
        
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        
        return R, T
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        frame = self.frames[index]
        
        # Load image
        image = self._load_image(index)
        
        # Load depth
        depth = self._load_depth(index)
        
        # Get camera parameters
        R, T = self._get_camera_params(index)
        time = frame['time']
        control_vec = frame['joint_pos']
        
        # Get paths
        image_path = os.path.join(self.datadir, frame['file_path'])
        image_name = Path(image_path).stem
        
        # Create CameraInfo
        cam_info = CameraInfo(
            uid=index,
            R=R,
            T=T,
            FovY=self.FovY,
            FovX=self.FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=self.width,
            height=self.height,
            time=time,
            control_vec=control_vec,
            mask=None,
            camera_idx=frame['camera_idx'],
            depth=depth,
            sample_idx=frame.get('sample_idx', 0)
        )
        
        return cam_info
    
    def load_pose(self, index):
        return self._get_camera_params(index)
    
    def load_control_vec(self, index):
        frame = self.frames[index]
        return self._normalize_control_vec(frame['joint_pos'])

