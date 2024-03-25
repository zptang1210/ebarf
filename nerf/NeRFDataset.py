import os
import cv2
import glob
import json
import tqdm
import pickle
import numpy as np
import shutil
from scipy.spatial.transform import Slerp, Rotation

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
from .utils import get_rays, get_event_rays
from utils.pose_utils import *
from utils.plot_utils import *
from utils.event_utils import *

import hdf5plugin

# NeRF dataset
import json
import matplotlib
matplotlib.use('Agg')

from .NGPDataset import NGPDataset


class NeRFDataset(NGPDataset):
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, select_frames=None, get_rays_evs_on_collate=True):
        super().__init__(opt, device, type=type, downscale=downscale, n_test=n_test, select_frames=select_frames)
        self.get_rays_evs_on_collate = get_rays_evs_on_collate # * This should be False when using ebarf since we don't predict evCam's views!

    def collate(self, index):
        B = len(index) # always 1

        poses = self.poses[index].to(self.device) # [B, 4, 4]
        error_map = None if self.error_map is None else self.error_map[index]
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map)
        
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

        if (self.mode == "tumvie" or self.mode == "eds") and self.type == "val" and self.get_rays_evs_on_collate: 
            poses_evCam = self.poses_evCam_atValIdxs[index, ...].to(self.device)
            rays = get_rays(poses_evCam, self.intrinsics_evs, self.H_ev, self.W_ev, self.num_rays, error_map)
            results['rays_evs_o'] = rays['rays_o']
            results['rays_evs_d'] = rays['rays_d']

        if self.type == "val" and self.e2vid:
            results['images'] = self.e2vid_gts[index].to(self.device) # overwriting image to compute psnr against it

        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0) 
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        return loader
