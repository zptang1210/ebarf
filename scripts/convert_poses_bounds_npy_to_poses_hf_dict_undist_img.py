import os, sys
import pickle
import numpy as np
import torch
import torch.nn.functional as torch_F

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from nerf.pose_utils import camera

PATH = './poses_bounds.npy'
TIMESTAMP_FILE = './timestamps.txt'

def parse_cameras_and_bounds(path):
    fname = path #"{}/poses_bounds.npy".format(path)
    data = torch.tensor(np.load(fname),dtype=torch.float32)
    # parse cameras (intrinsics and poses)
    cam_data = data[:,:-2].view([-1,3,5]) # [N,3,5]
    poses_raw = cam_data[...,:4] # [N,3,4] # *down right backwards
    poses_raw[...,0],poses_raw[...,1] = poses_raw[...,1],-poses_raw[...,0] # * right up backwards

    poses_raw[...,1], poses_raw[...,2] = -poses_raw[...,1], -poses_raw[...,2]

    raw_H,raw_W,focal = cam_data[0,:,-1]
    
    intr = torch.tensor([[focal,0,raw_W/2],
                         [0,focal,raw_H/2],
                         [0,0,1]]).float()

    # parse depth bounds
    bounds = data[:,-2:] # [N,2]

    # scale = 10 / (bounds.max() - bounds.min()) # todo
    # scale = 1./(bounds.min()*0.75) # not sure how this was determined
    scale = 0.01818
    print('scale', scale)
    poses_raw[...,3] *= scale 
    bounds *= scale
    # roughly center camera poses
    # poses_raw = center_camera_poses(poses_raw)
    return poses_raw,bounds, intr

def center_camera_poses(poses):
    # compute average pose
    center = poses[...,3].mean(dim=0)
    v1 = torch_F.normalize(poses[...,1].mean(dim=0),dim=0)
    v2 = torch_F.normalize(poses[...,2].mean(dim=0),dim=0)
    v0 = v1.cross(v2)
    pose_avg = torch.stack([v0,v1,v2,center],dim=-1)[None] # [1,3,4]
    # apply inverse of averaged pose
    poses = camera.pose.compose([poses,camera.pose.invert(pose_avg)])
    return poses

if __name__ == '__main__':
    poses_hf, bounds, intr = parse_cameras_and_bounds(PATH)
    print(intr)
    timestamps_us = torch.tensor(np.loadtxt(TIMESTAMP_FILE), dtype=torch.float32)
    # print(poses_hf.shape, bounds.shape, timestamps_us.shape)
    timestamps = timestamps_us * 1e3 # to ns
    print(poses_hf.dtype, timestamps.dtype)

    poses_hf_dict = {
        'poses_hf': poses_hf,
        'tss_poses_hf_ns': timestamps,
        'raw_poses_hf': poses_hf,
        'raw_tss_poses_hf_ns': timestamps,
        'comment': 'flyingroom poses computed by colmap from undistorted images.'
    }

    with open('poses_hf_flyingroom_from_undistorted_e2vid.pickle', 'wb') as fout:
        pickle.dump(poses_hf_dict, fout)

