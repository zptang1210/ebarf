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
from .utils import get_rays, get_event_rays, get_universal_workspace_path
from utils.pose_utils import *
from utils.plot_utils import *
from utils.event_utils import *

import hdf5plugin

# NeRF dataset
import json
import matplotlib
matplotlib.use('Agg')

#####################
# General Helpers
#####################
def load_intrinsics(hwf):
    H, W, focal = hwf
    H, W = int(H), int(W)
    cx = W / 2.0
    cy = H / 2.0
    print(f"cx={cx}, cy={cy}, H={H}, W={W}")
    return H, W, focal, cx, cy

#####################
# Loading tumvie
#####################
def create_poses_bounds_tumvie(all_poses_evClk_us, tss_imgs_evClk_us, bounds, num_imgs, H_final, W_final, focal_final, T_imu_marker=None, T_imu_cam=None, prec_angle=0.8, prec_trans=0.04):
    """
    Input: 
    all_poses_evClk_us: poses list (stamp_us, x, y, z, wx, wy, wz, ww)

    Output: 
    list of (r1, r2, r3, t, hwf, min_depth, max_depth)-entries, i.e. the llff-data format for poses
    """
    assert len(tss_imgs_evClk_us) == num_imgs
    assert len(bounds) == num_imgs

    tss_all_poses_us = [t[0] for t in all_poses_evClk_us]
    tss_all_poses_us, all_trafos = quatList_to_poses_hom_and_tss(all_poses_evClk_us)

    all_trafos_c2w = [T_mocap_marker @ np.linalg.inv(T_imu_marker) @ T_imu_cam for T_mocap_marker in all_trafos]
    all_rots = [r[0, :3, :3] for r in all_trafos_c2w]
    all_trans = [t[0, :3, 3] for t in all_trafos_c2w]
    all_trafos_c2w_list = all_poses_evClk_us.copy()
    all_trafos_c2w_list[:, 1:4] = np.asarray(all_trans)
    all_trafos_c2w_list[:, 4:] = np.asarray(([R.from_matrix(r).as_quat() for r in all_rots]))

    print("\nCreating poses_bounds tumvie (rot, trans, hwf, min_depth, max_depth)")
    poses_bounds = []  # save poses in llff format
    pbar = tqdm.tqdm(total=num_imgs)
    skipped = 0
    for i in range(num_imgs):
        if tss_all_poses_us[0] - tss_imgs_evClk_us[i] > 0:
            print(f"Moving ts {i}.jpg by {(tss_all_poses_us[0] - tss_imgs_evClk_us[i])*1e3} ms to first pose-ts (still creating pose_bounds-entry)")
            skipped += 1
            tss_imgs_evClk_us[i] = tss_all_poses_us[0]
        if tss_all_poses_us[-1] - tss_imgs_evClk_us[i] < 0:
            print(f"Moving ts {i}.jpg by {(tss_all_poses_us[0] - tss_imgs_evClk_us[i])*1e3} ms to last pose-ts (still creating pose_bounds-entry)")
            skipped += 1
            tss_imgs_evClk_us[i] = tss_all_poses_us[-1]
            
        rot_slerp, trans_slerp = interpol_poses_slerp(
            tss_all_poses_us, all_rots, all_trans, tss_imgs_evClk_us[i]
        )

        hwf = np.array((H_final, W_final, focal_final))
        rthwf = np.concatenate((rot_slerp, trans_slerp[..., np.newaxis], hwf[..., np.newaxis]), axis=1)
        # (r1, r2, r3, t, hwf).ravel = (3, 5).ravel = 15 + min_depth, max_depth
        poses_bounds.append(np.concatenate([rthwf.ravel(), np.array([bounds[i][0], bounds[i][1]])], 0))
        pbar.update(1)
    
    print(f"Interpolated {len(poses_bounds)} poses from total of {len(all_poses_evClk_us)}")
    assert skipped <= 2
    return poses_bounds

#####################
# Preprocess poses
#####################
def preprocess_poses_sphere(quatlist_m2w_rdf_us, outpath, T_imu_rgbCam, T_imu_marker, bound=4):
    # All cameras should look at center of sphere. Recentering done in rgbCam-system. 
    assert quatlist_m2w_rdf_us.shape[0] > 2
    assert quatlist_m2w_rdf_us.shape[1] == 8

    # 1) T_world_marker => T_world_rgbCam
    tss_us, poses_m2w = quatList_to_poses_hom_and_tss(quatlist_m2w_rdf_us)
    T_marker_rgbCam = np.linalg.inv(T_imu_marker) @ T_imu_rgbCam # this is T_ev_rgb for eds
    poses_c2w_rdf = [T_world_marker @ T_marker_rgbCam for T_world_marker in poses_m2w]
    poses = np.asarray(poses_c2w_rdf) # (N, 4, 4)

    # 2) Spherification in c2w-space
    poses = preprocess_poseArr_sphere(poses)
    # [debug]: uncomment for visualization
    # visualize_poses(poses[::100,...], bound=bound)

    # 3) Save results for debugging
    quatlist_rgb2world = poses_hom_to_quatlist(poses, tss_us)
    np.savetxt(os.path.join(outpath, "pp_mocap_rdf_rgb2world.txt"), quatlist_rgb2world, header="# preprocessed mocap poses (T_world_rgbCam, rdf): time(us) px py pz qx qy qz qw")

    # 4) transform back to marker-space (poses_bds is transformed to rgb, and self.poses_hf to event)
    quatlist_marker2world = poses_hom_to_quatlist(np.asarray([T_world_rgb @ np.linalg.inv(T_marker_rgbCam) for T_world_rgb in poses]), tss_us)
    return np.asarray(quatlist_marker2world)

def preprocess_poseArr_sphere(poses):
    N = len(poses)
    print(f'[INFO] average radius before re-centering = {np.linalg.norm(poses[:, :3, 3], axis=-1).mean()}')
    poses[:, :3, :] = recenter_poses2(poses[:, :3, :])
    print(f'[INFO] average radius after re-centering = {np.linalg.norm(poses[:, :3, 3], axis=-1).mean()}')
    
    # [debug]: uncomment for visualization (of original poses)
    # ps = poses
    # ps = poses[:, :3, :] * 20
    # plot_poses(poses[::200, :3, :], l=0.5, title="before flip")

    poses[:, 0:3, 1] *= -1 # flip the y and z axis
    poses[:, 0:3, 2] *= -1 # now poses are in rub (confirmed by plotting)
    poses = poses[:, [1, 0, 2, 3], :] # swap y and z 
    poses[:, 2, :] *= -1 # flip whole world upside down  

    up = poses[:, 0:3, 1].sum(0) 
    up = up / np.linalg.norm(up)
    R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    poses = R @ poses
    # [debug]: uncomment for visualization (of transformed poses)
    # plot_poses(poses[:, :3, :], l=10, title="after rot")
    # visualize_poses(poses[::200], bound=bound)

    print("computing center of attention for hf-poses...")
    idxs = np.random.randint(0, N, size=(100)) # subsample since we have too many hf-poses
    poses_sub = poses[idxs]
    pbar = tqdm.tqdm(total=len(poses_sub)**2)
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for i in range(len(poses_sub)):
        mf = poses_sub[i, :3, :]
        for j in range(len(poses_sub)):
            mg = poses_sub[j, :3, :]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.01:
                totp += p * w
                totw += w
            pbar.update(1)
    totp /= totw
    print(f'[INFO] totp = {totp}')
    poses[:, :3, 3] -= totp
    avglen = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
    print(f'[INFO] avg. radius after aligning to center of attention = {avglen}')
    poses[:, :3, 3] *= 1.0 / avglen 
    print(f'[INFO] avg. radius after rescaling to approximately 1 = {np.linalg.norm(poses[:, :3, 3], axis=-1).mean()}')

    return poses



class NGPDataset(Dataset):
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, select_frames=None):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.datadir
        self.mode = opt.mode # esim, tumvie, eds, evo
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.images_corrupted = opt.images_corrupted
        self.training = self.type in ['train', 'all'] # what is 'trainval' in current main? is it to load all data for training, i.e. cheating?
        self.num_rays = self.opt.num_rays if self.training else -1
    
        self.use_events = opt.events
        self.event_only = opt.event_only
        self.e2vid = opt.e2vid
        self.negative_event_sampling = opt.negative_event_sampling
        self.pp_poses_sphere = opt.pp_poses_sphere
        self.out_dim_color = opt.out_dim_color 
        self.rand_pose = opt.rand_pose
        self.hotpixs = opt.hotpixs
        self.acc_max_num_evs = opt.acc_max_num_evs
        self.precompute_evs_poses = opt.precompute_evs_poses
        self.poses_hf = None
        self.code_dir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
        self.imgdir = None

        # Experiment logging
        self.workspace = get_universal_workspace_path(opt)
        print(f"Dataloader uses {self.workspace} as workspace.")

        if select_frames is not None:
            if type == 'train':
                idxs = select_frames["train_idxs"]
            elif type == 'val':
                idxs = select_frames["val_idxs"]
            elif type == 'all':
                idxs = select_frames["train_idxs"] + select_frames["val_idxs"]
            else: 
                idxs = select_frames["train_idxs"] + select_frames["val_idxs"]
            self.frame_idxs = np.asarray(idxs)
        self.process_id = str(os.getpid())
        self.slurm_id = str(os.environ.get("SLURM_JOBID"))
        self.transform_filepath = os.path.join(self.workspace, 'transform_' + self.slurm_id + "_" + self.process_id + "_" + self.type + ".json") if self.mode != "colmap" else os.path.join(opt.datadir, "transforms.json")
        
        if self.mode == 'esim':
            if self.e2vid:
                p = glob.glob(os.path.join(self.root_path, f"e2vids/e2vid_up{self.e2vid}_*/e2calib/")) # "e2calib_upsample4/"
                assert len(p) == 1
                self.imgdir = p[0] 
                assert os.path.exists(self.imgdir)
                assert ("e2calib" in self.imgdir) or ("e2vid" in self.imgdir)
            
            self.convert_esim_to_posesBds_and_hfPoses()
            if type == 'train' or not os.path.exists(self.transform_filepath):
                 self.create_transform_json_from_posesBds()
            with open(self.transform_filepath, 'r') as f:
                transform = json.load(f)
        elif self.mode == 'tumvie':
            self.camId = 0
            self.camIdEvs = 2
            assert self.camId == 0 or self.camId == 1
            assert self.camIdEvs == 2 or self.camIdEvs == 3
            self.imgdir = "left_images_undistorted/" if self.camId == 0 else "right_images_undistorted/"
            if self.e2vid:
                p = glob.glob(os.path.join(self.root_path, f"e2vids/left/e2vid_up{self.e2vid}_*/e2calib_undistorted/")) 
                assert len(p) == 1
                self.imgdir = p[0] 
                assert os.path.exists(self.imgdir)
                assert ("e2calib" in self.imgdir) or ("e2vid" in self.imgdir)
                self.camId = 2
                self.camIdEvs = 2
                if not self.event_only and self.use_events:  # using both frame & event camera
                    self.camIdEvs = 2

            with open(os.path.join(self.root_path, "calib_undist.json"), 'r') as f:
                self.calibdata = json.load(f)["value0"]
            with open(os.path.join(self.root_path, "mocap-imu-calib.json"), 'r') as f:
                self.calibdata.update(json.load(f)["value0"])

            shutil.copy(os.path.join(self.root_path, "calibration.json"), os.path.join(self.workspace, "calibration.json"))
            shutil.copy(os.path.join(self.root_path, "calib_undist.json"), os.path.join(self.workspace, "calib_undist.json"))
            shutil.copy(os.path.join(self.root_path, "mocap-imu-calib.json"), os.path.join(self.workspace, "mocap-imu-calib.json"))
            
            T_imu_rgb = quat_dict_to_pose_hom(self.calibdata["T_imu_cam"][self.camId])
            T_imu_ev = quat_dict_to_pose_hom(self.calibdata["T_imu_cam"][self.camIdEvs])
            self.T_ev_rgb = np.linalg.inv(T_imu_ev) @ T_imu_rgb

            self.convert_tumvie_to_posesBds_and_hfPoses()
            if not os.path.exists(self.transform_filepath):
                self.create_transform_json_from_posesBds()
            with open(self.transform_filepath, 'r') as f:
                transform = json.load(f)
        elif self.mode == 'eds':
            self.camId = 0
            self.camIdEvs = 1
            self.calibstr = "calib0"
            self.imgdir = f"images_undistorted_{self.calibstr}/"

            if self.e2vid:
                p = glob.glob(os.path.join(self.root_path, f"e2vids/e2vid_up{self.e2vid}_*/e2calib_undistorted/"))
                assert len(p) == 1
                self.imgdir = p[0] 
                assert os.path.exists(self.imgdir)
                assert ("e2calib" in self.imgdir) or ("e2vid" in self.imgdir)
                self.camId = 1
                self.camIdEvs = 0
                if not self.event_only and self.use_events: 
                    self.camIdEvs = 1

            if self.calibstr == "calib1":
                self.T_rgb_imu = np.asarray([ # calib1::cam0
                            [-0.9990674261177589, 0.003631371785536113, 0.04302430951674526, 0.029775744033325717], 
                            [-0.0028069257561474303, -0.9998115832434417, 0.019207268937644115, -3.945506719509616e-05],
                            [0.043085951750390296, 0.019068590697760665, 0.9988893780647411, -0.05058322878791149], 
                            [0.0, 0.0, 0.0, 1.0]])

                self.T_ev_imu = np.asarray([ # calib1::cam1
                            [-0.9995991738524179, 0.005137352230635228, 0.027840604261081696, 0.025511174632699758], 
                            [-0.005338788412348343, -0.9999600732256928, -0.007165842082780113, 0.00040093104150374986],
                            [0.027802679220750436, -0.007311604921325729, 0.9995866903183648, -0.06778644321079669], 
                            [0.0, 0.0, 0.0, 1.0]])
                self.T_ev_rgb = self.T_ev_imu @ np.linalg.inv(self.T_rgb_imu) # == T_cn_cnm1 == T_cam1_cam0 (calib1)
            elif self.calibstr == "calib0":
                self.T_ev_rgb = np.asarray([  # calib0
                    [0.9998964430808897, -0.0020335804041023736, -0.014246672065022661, -0.00011238613157578769],
                    [0.001703024953250547, 0.9997299470300024, -0.023176123864880376, -0.0005981481496958399],
                    [0.014289955220253567, 0.02314946137886846, 0.9996298813149167, -0.004416681577516066],
                    [0.0, 0.0, 0.0, 1.0]
                ])

            with open(os.path.join(self.root_path, f"calib_undist_{self.calibstr}.json"), 'r') as f:
                self.calibdata = json.load(f)
            
            self.convert_EDS_to_posesBds_and_hfPoses()
            if not os.path.exists(self.transform_filepath):
                self.create_transform_json_from_posesBds()
            with open(self.transform_filepath, 'r') as f:
                transform = json.load(f)
        elif self.mode == 'evo':
            self.camId = 0
            self.camIdEvs = 1
            self.calibstr = "calib0"
            self.imgdir = f"images_undistorted_{self.calibstr}/"

            if self.e2vid:
                raise NotImplementedError('e2vid not supported yet.')

            self.T_ev_rgb = quat_dict_to_pose_hom({"px": 0.0, "py": 0.0, "pz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0})

            with open(os.path.join(self.root_path, f"calib_undist_{self.calibstr}.json"), 'r') as f:
                self.calibdata = json.load(f)
            
            self.convert_EVO_to_posesBds_and_hfPoses()
            if not os.path.exists(self.transform_filepath):
                self.create_transform_json_from_posesBds()
            with open(self.transform_filepath, 'r') as f:
                transform = json.load(f)
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # [debug]: uncomment to plot the poses here
        # plotted_poses = []
        # for p in transform["frames"]:
        #     plotted_poses.append(np.asarray(p["transform_matrix"])[:3, :4])
        # plot_coord_sys_c2w(np.asarray(plotted_poses), plot_kf=True)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we will read image to get H, W.
            self.H = self.W = None
        
        if (opt.mode == "tumvie" or opt.mode == "eds" or opt.mode == "evo"):
            self.W_ev = transform['w_evs']
            self.H_ev = transform['h_evs']
            assert self.W_ev == 1280 if self.mode == "tumvie" else self.W_ev == self.W
            assert self.H_ev == 720 if self.mode == "tumvie" else self.H_ev == self.H
        if opt.mode == "esim":
            self.W_ev, self.H_ev = self.W, self.H
        
        # reading images
        frames = transform["frames"]
        frames = sorted(frames, key=lambda x: x["file_path"])
        frames = [frames[i] for i in self.frame_idxs]
        self.load_kfs_data(frames)
        self.poses_hf = [{"ts_ns": p["ts_ns"], "pose_c2w": nerf_matrix_to_ngp(p["pose_c2w"], scale=self.scale)[:3, :]} for p in self.poses_hf]

        if "11_all_characters" in self.root_path:
            self.update_poses(x=-1.5, y=-0.5, z=-0.75)
        elif "00_peanuts_dark" in self.root_path:
            if not self.pp_poses_sphere:
                self.update_poses(x=-1, y=-0.5, z=-1)
        elif "ShakeCarpet1" in self.root_path:
            self.update_poses(x=0, y=0, z=0.3)
        # [debug] uncomment to plot final poses.
        # plot_poses(self.poses[::5, :3, :], l=0.5, title="final")
        # visualize_poses(self.poses, bound=self.bound)
        # ps_hf = np.asarray([p["pose_c2w"] for p in self.poses_hf])
        # visualize_poses(ps_hf[::100, ...], bound=self.bound)

        ############ [debugging]: saving final self.poses and self.poses_hf
        ps = np.asarray([p["pose_c2w"] for p in self.poses_hf])
        ts_ns = np.asarray([p["ts_ns"] for p in self.poses_hf])
        quatlist_hf_ns = poses_hom_to_quatlist(get_hom_trafos(ps[:, :3,:3], ps[:, :3, 3]), ts_ns)
        np.savetxt(os.path.join(self.workspace, f"{self.type}_final_quatlist_hf_ns.txt"), quatlist_hf_ns, header="stamps in nanoseconds, px, py, pz, qx, qy, qz, qw")
        quatlist_kfs_ns = poses_hom_to_quatlist(self.poses, self.tss_imgs_us[self.frame_idxs])
        np.savetxt(os.path.join(self.workspace, f"{self.type}_final_quatlist_kfs_ns.txt"), quatlist_kfs_ns, header="stamps in nanoseconds, px, py, pz, qx, qy, qz, qw")
        ############
 
        self.poses = torch.from_numpy(self.poses) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]

        if self.e2vid:
            self.e2vid_gts = torch.from_numpy(np.stack(self.e2vid_gts, axis=0))

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        print(f'[INFO] radius after adding data-set-specific offset = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy(), bound=self.bound)

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # loading intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('cannot read focal!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.H / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.W / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        if opt.events or opt.mode == "tumvie" or opt.mode == "eds" or opt.mode == "evo":
            self.intrinsics_evs = np.array([transform['fl_x_evs'], transform['fl_y_evs'], transform['cx_evs'], transform['cy_evs']])
            assert self.intrinsics_evs[2] > 100 and self.intrinsics_evs[2] < 2000
            assert self.intrinsics_evs[3] > 100 and self.intrinsics_evs[3] < 2000
        if self.preload:
            self.intrinsics = torch.from_numpy(self.intrinsics).to(self.device)

        if (self.mode == "tumvie" or self.mode == "eds" or self.mode == "evo") and self.type == "val": # Event Camera poses
            tss_poses_hf_ns = np.asarray([p["ts_ns"] for p in self.poses_hf]) # self.poses_hf is in Event-View
            rots_hf = [p["pose_c2w"][:3, :3] for p in self.poses_hf]
            trans_hf = [p["pose_c2w"][:3, 3] for p in self.poses_hf]
            rots, trans = interpol_poses_slerp(tss_poses_hf_ns, rots_hf, trans_hf, self.tss_imgs_us[self.frame_idxs]*1000)
            poses_evCam_atValIdxs = get_hom_trafos(rots, trans).astype(np.float32)
            self.poses_evCam_atValIdxs = torch.from_numpy(poses_evCam_atValIdxs[:, :3, :])
            quatlist_hf_ns = poses_hom_to_quatlist(poses_evCam_atValIdxs, self.tss_imgs_us[self.frame_idxs]*1000)
            np.savetxt(os.path.join(self.workspace, f"{self.type}_final_evCam_quatlist_atValTss_ns.txt"), quatlist_hf_ns, header="stamps in nanoseconds, px, py, pz, qx, qy, qz, qw")

        # Sanity checks
        assert self.H > 100 and self.H < 4000
        assert self.W > 100 and self.W < 4000
        assert cx > 50 and cx < 2000
        assert cy > 50 and cy < 2000

    def update_poses(self, x=0, y=0, z=0):
        self.poses[:, 0, 3] += x
        self.poses[:, 1, 3] += y
        self.poses[:, 2, 3] += z

        poses_dict = []
        for i in range(len(self.poses_hf)):
            pupdated = self.poses_hf[i]["pose_c2w"]
            pupdated[0, 3] += x
            pupdated[1, 3] += y
            pupdated[2, 3] += z
            poses_dict.append({"pose_c2w": pupdated, "ts_ns": self.poses_hf[i]["ts_ns"]})
        self.poses_hf = poses_dict
        print(f"[IMPORTANT]: updated self.poses and self.hf_poses with {x}, {y}, {z}")

    def convert_esim_to_posesBds_and_hfPoses(self):
        c2w, _ = read_poses_bounds(os.path.join(self.root_path, "poses_bounds.npy"))
        hwf = c2w[0, :3, -1]
        H, W, f = hwf

        poses_gt_ns = np.loadtxt(glob.glob(os.path.join(self.root_path, "*poses_all*.txt"))[0], skiprows=1)
        tss_gt_ns = poses_gt_ns[:, 0]
        assert np.all(tss_gt_ns == sorted(tss_gt_ns))
        assert poses_gt_ns.shape[0] > 100
        assert poses_gt_ns.shape[1] == 8

        if self.e2vid:
            tss_imgs_ns = np.loadtxt(os.path.join(self.imgdir, "timestamps.txt")) * 1000
        else:
            if self.type == "train" and self.images_corrupted:
                imgfolder = "images_corrupted" # only use the corrupted images for training, not for testing
            else: 
                imgfolder = "images" # use non-corrupted images for psnr-computation
            tss_imgs_ns = np.loadtxt(os.path.join(self.root_path, imgfolder, "image_stamps_ns.txt"))
        self.tss_imgs_us = tss_imgs_ns * 1e-3

        bds = np.zeros((len(tss_imgs_ns), 2))
        imglist = sorted(glob.glob(os.path.join(self.root_path, "images", "*jpg")))
        if len(imglist) > 0:
            img0 = cv2.imread(imglist[0])
        else:
            img0 = cv2.imread(sorted(glob.glob(os.path.join(self.root_path, "images", "*.png")))[0])
        assert img0.shape[0] == H and img0.shape[1] == W
        H, W = img0.shape[0], img0.shape[1]
        
        tss_all_poses_ns, poses_hom = quatList_to_poses_hom_and_tss(poses_gt_ns)
        if self.pp_poses_sphere:
            poses_hom = preprocess_poseArr_sphere(poses_hom) 
        quatlist_ns = poses_hom_to_quatlist(poses_hom, tss_all_poses_ns)

        pb_path = os.path.join(self.workspace, f"poses_bounds_{self.type}.npy")
        poses_bounds = create_poses_bounds_esim(quatlist_ns, tss_imgs_ns, bds, len(self.tss_imgs_us), H, W, focal_final=f)
        np.save(pb_path, poses_bounds)

        #####################
        plotting_poses_bounds(self.workspace, self.tss_imgs_us, poses_bounds)
        #####################

        if not self.pp_poses_sphere:
            poses_hom = rub_from_rdf(poses_hom[:, :3, :])
        poses_dict = []
        for i in range(poses_hom.shape[0]):
            poses_dict.append({"pose_c2w": poses_hom[i, :, :], "ts_ns": tss_all_poses_ns[i]})
        self.poses_hf = poses_dict

    def convert_EVO_to_posesBds_and_hfPoses(self):
        assert (not self.pp_poses_sphere)
        assert (not self.e2vid)
        self.convert_EDS_to_posesBds_and_hfPoses()

    def convert_EDS_to_posesBds_and_hfPoses(self):
        poses_gt_us = np.loadtxt(os.path.join(self.root_path, "stamped_groundtruth_us.txt"), skiprows=1)
        tss_gt_us = poses_gt_us[:, 0]
        assert np.all(tss_gt_us == sorted(tss_gt_us))
        assert poses_gt_us.shape[0] > 100
        assert poses_gt_us.shape[1] == 8

        if self.e2vid:
            tss_imgs_us = np.loadtxt(os.path.join(self.imgdir, "timestamps.txt"))
        else:
            tss_imgs_us = np.loadtxt(os.path.join(self.root_path, "images_timestamps_us.txt"))
        self.tss_imgs_us = tss_imgs_us
        bds = np.zeros((len(tss_imgs_us), 2))
        img0 = cv2.imread(os.path.join(self.root_path, "images", "frame_0000000000.png"))
        H, W = img0.shape[0], img0.shape[1]

        T_imu_marker = quat_dict_to_pose_hom({"px": 0.0, "py": 0.0, "pz": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0})
        T_ev_rgb = T_imu_marker.copy() if self.e2vid else np.copy(self.T_ev_rgb)

        # Preprocessing: 
        if self.pp_poses_sphere:
            poses_gt_us = preprocess_poses_sphere(poses_gt_us, self.workspace, T_ev_rgb.squeeze(), T_imu_marker.squeeze(), bound=self.bound)

        pb_path = os.path.join(self.workspace, "poses_bounds.npy")
        if self.type == "train" and (not os.path.exists(pb_path) or True):
            poses_bounds = create_poses_bounds_tumvie(poses_gt_us, tss_imgs_us, bds, len(tss_imgs_us), H, W, focal_final=-1, T_imu_marker=T_imu_marker, T_imu_cam=T_ev_rgb, prec_angle=1.5, prec_trans=0.06)
            np.save(pb_path, poses_bounds)            
            #####################
            plotting_poses_bounds(self.workspace, tss_imgs_us, poses_bounds)
            #####################

        tss_all_poses_ns, all_trafos_c2w = quatList_to_poses_hom_and_tss(poses_gt_us)
        tss_all_poses_ns = [t * 1000 for t in tss_all_poses_ns]
        if not self.pp_poses_sphere:
            all_trafos_c2w = rub_from_rdf(all_trafos_c2w[:, :3, :])
        check_rot_batch(all_trafos_c2w[:, :3, :])

        poses_dict = []
        for i in range(all_trafos_c2w.shape[0]):
            poses_dict.append({"pose_c2w": all_trafos_c2w[i, :, :], "ts_ns": tss_all_poses_ns[i]})
        self.poses_hf = poses_dict # contains raw poses at recording frequency in event-cam system for EDS

    def convert_tumvie_to_posesBds_and_hfPoses(self):
        poses_gt_us = np.loadtxt(glob.glob(os.path.join(self.root_path, "*mocap*.txt"))[0], skiprows=1)
        assert "pp_mocap" not in glob.glob(os.path.join(self.root_path, "*mocap*.txt"))[0]
            
        tss_gt_us = poses_gt_us[:, 0]
        assert np.all(tss_gt_us == sorted(tss_gt_us))
        if not np.median(np.abs(1./np.diff(tss_gt_us)*1e6 - 120.0)) < 1:
            assert np.median(np.abs(1./np.diff(tss_gt_us)*1e6 - 120.0)) < 1
        assert poses_gt_us.shape[0] > 100
        assert poses_gt_us.shape[1] == 8

        if self.e2vid:
            tss_imgs_left_us = np.loadtxt(os.path.join(self.imgdir, "timestamps.txt"))
        else:
            tss_imgs_left_us = np.loadtxt(os.path.join(self.root_path, self.imgdir, "image_timestamps_left.txt"))
            tss_imgs_right_us = np.loadtxt(os.path.join(self.root_path, "right_images", "image_timestamps_right.txt"))
            if len(tss_imgs_right_us) > 0:
                assert len(tss_imgs_left_us) == len(tss_imgs_right_us)
                assert np.abs(tss_imgs_left_us-tss_imgs_right_us).max() < 1 # 1e-6
        
        tss_imgs_us = tss_imgs_left_us
        self.tss_imgs_us = tss_imgs_us

        bds = np.zeros((len(tss_imgs_us), 2))
        if os.path.isfile(os.path.join(self.root_path, self.imgdir, "00000.jpg")):
            img0 = cv2.imread(os.path.join(self.root_path, self.imgdir, "00000.jpg"))
        else:
            img0 = cv2.imread(glob.glob(os.path.join(self.root_path, self.imgdir, "*.png"))[0])
        H, W = img0.shape[0], img0.shape[1]

        T_imu_rgbCam = quat_dict_to_pose_hom(self.calibdata["T_imu_cam"][self.camId]) # camId == 2 if self.e2vid => hence, we get the correct poses! (we then have event intrnisics saved in self.intriniscs)
        T_imu_marker = quat_dict_to_pose_hom(self.calibdata["T_imu_marker"])
        if self.pp_poses_sphere:
            poses_gt_us = preprocess_poses_sphere(poses_gt_us, self.workspace, T_imu_rgbCam.squeeze(), T_imu_marker.squeeze(), bound=self.bound)

        pb_path = os.path.join(self.workspace, "poses_bounds.npy")
        if self.type == "train" and (not os.path.exists(pb_path) or True):
            poses_bounds = create_poses_bounds_tumvie(poses_gt_us, tss_imgs_us, bds, len(tss_imgs_us), H, W, focal_final=-1, T_imu_marker=T_imu_marker, T_imu_cam=T_imu_rgbCam)
            np.save(pb_path, poses_bounds)
            #####################
            plotting_poses_bounds(self.workspace, tss_imgs_us, poses_bounds)
            #####################

        tss_all_poses_ns, all_trafos = quatList_to_poses_hom_and_tss(poses_gt_us)
        tss_all_poses_ns = [t * 1000 for t in tss_all_poses_ns]
        T_imu_evCam = quat_dict_to_pose_hom(self.calibdata["T_imu_cam"][self.camIdEvs])

        all_trafos_c2w = np.asarray([T_mocap_marker @ np.linalg.inv(T_imu_marker) @ T_imu_evCam for T_mocap_marker in all_trafos]).squeeze()[:, :3, :]
        if not self.pp_poses_sphere:
            all_trafos_c2w = rub_from_rdf(all_trafos_c2w[:, :3, :]) # * final coord is RDF
        check_rot_batch(all_trafos_c2w)

        poses_dict = []
        for i in range(all_trafos_c2w.shape[0]):
            poses_dict.append({"pose_c2w": all_trafos_c2w[i, :, :], "ts_ns": tss_all_poses_ns[i]})
        self.poses_hf = poses_dict # contains raw poses (in event-coord-sys) at recording frequency

    def create_transform_json_from_posesBds(self):
        path = self.root_path
        if self.mode == "esim":
            poses_c2w, bds = read_poses_bounds(os.path.join(self.workspace, F"poses_bounds_{self.type}.npy"), invert=False)
            poses_c2w = poses_c2w if self.pp_poses_sphere else rub_from_rdf(poses_c2w)
        elif self.mode == "tumvie":
            poses_c2w, bds = read_poses_bounds(os.path.join(self.workspace, F"poses_bounds.npy"), invert=False)
            poses_c2w = poses_c2w if self.pp_poses_sphere else rub_from_rdf(poses_c2w)
        elif self.mode == "eds":
            poses_c2w, bds = read_poses_bounds(os.path.join(self.workspace, F"poses_bounds.npy"), invert=False)
            poses_c2w = poses_c2w if self.pp_poses_sphere else rub_from_rdf(poses_c2w)
        elif self.mode == "evo":
            poses_c2w, bds = read_poses_bounds(os.path.join(self.workspace, F"poses_bounds.npy"), invert=False)
            poses_c2w = poses_c2w if self.pp_poses_sphere else rub_from_rdf(poses_c2w)
        else:
            print(f"Unknown mode {self.mode}. Exiting")
            sys.exit()

        if self.imgdir is not None:
            img_folder = self.imgdir
        else:
            if self.images_corrupted and self.type == 'train':
                img_folder = "images_corrupted/" # only for train we chose corrupted images, for val we want to compute clean images for GT-evaluation
            else:
                img_folder = "images/"
            
        imgs_path = sorted(glob.glob(os.path.join(path, img_folder) + "*.png"))
        if len(imgs_path) == 0:
            imgs_path = sorted(glob.glob(os.path.join(path, img_folder) + "*.jpg"))
            if len(imgs_path) == 0: 
                print("could not find images")
                sys.exit()
        assert np.abs(len(imgs_path) - poses_c2w.shape[0]) == 0

        hwf = poses_c2w[0, :3, -1]
        if self.mode == "tumvie" or self.mode == "eds" or self.mode == "evo":
            intr = self.calibdata["intrinsics_undistorted"][self.camId]
            if self.mode == "tumvie" and self.e2vid:
                intr = {"fx": 982.70606462, "fy": 982.55614817, "cx":  632.7907291, "cy": 250.45539802 } # only used once, but problem with this is the border => might be even fairer to undistort separately

            fl_x = intr["fx"]
            fl_y = intr["fy"]
            cx, cy = intr["cx"], intr["cy"]
            img0dim = cv2.imread(imgs_path[0]).shape
            H, W = img0dim[0], img0dim[1]
        
            assert self.downscale == 1 
            intr_evs = self.calibdata["intrinsics_undistorted"][self.camIdEvs]
            fl_x_evs, fl_y_evs = intr_evs["fx"], intr_evs["fy"]
            cx_evs, cy_evs = intr_evs["cx"], intr_evs["cy"]
            self.intrinsics_evs = np.array([fl_x_evs, fl_y_evs, cx_evs, cy_evs])
        else:
            H, W, focal, cx, cy = load_intrinsics(hwf)
            fl_x, fl_y = focal, focal
            fl_x_evs, fl_y_evs = focal, focal
            cx_evs, cy_evs = cx, cy
            assert focal == hwf[2]
        assert H == hwf[0]
        assert W == hwf[1]
        W_ev = 1280 if self.mode == "tumvie" else W
        H_ev = 720 if self.mode == "tumvie" else H

        angle_x = np.arctan(W/(fl_x*2))*2
        angle_y = np.arctan(H/(fl_y*2))*2
        fovx = angle_x*180/np.pi
        fovy = angle_y*180/np.pi
        k1, k2, p1, p2 = 0, 0, 0, 0
        print(f"camera:\n\tres={W,H}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

        out={
            "camera_angle_x":angle_x,
            "camera_angle_y":angle_y,
            "fl_x":fl_x,
            "fl_y":fl_y,
            "k1":k1,
            "k2":k2,
            "p1":p1,
            "p2":p2,
            "cx":cx,
            "cy":cy,
            "w":W,
            "h":H,
            "h_evs": H_ev,
            "w_evs": W_ev, 
            "fl_x_evs": fl_x_evs, 
            "fl_y_evs": fl_y_evs,
            "cx_evs": cx_evs, 
            "cy_evs": cy_evs,
            "frames":[]
        }
        for i, c2w in enumerate(poses_c2w):
            frame={"file_path":imgs_path[i] ,"transform_matrix": c2w[:, :4]}
            out["frames"].append(frame)

        for f in out["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()
        
        with open(self.transform_filepath, "w") as outfile:
            json.dump(out, outfile, indent=2)

    def load_kfs_data(self, frames):
        self.poses = []
        self.images = []
        if self.e2vid:
            self.e2vid_gts = []

            if self.mode == "tumvie":
                gtdir = "left_images_undistorted/" if self.camId == 0 else "right_images_undistorted/"
            elif self.mode == "eds":
                gtdir = f"images_undistorted_{self.calibstr}/"
            elif self.mode == "evo":
                gtdir = f"images_undistorted_{self.calibstr}/"
            elif self.mode == "esim":
                gtdir = "images/"

            gts_path = sorted(glob.glob(os.path.join(self.root_path, gtdir) + "*.png"))
            if len(gts_path) == 0:
                gts_path = sorted(glob.glob(os.path.join(self.root_path, gtdir) + "*.jpg"))
                if len(gts_path) == 0: 
                    print("could not find gt-images")
                    sys.exit()

        # Debugging only
        ps = []
        for p in frames:
            ps.append(np.array(p["transform_matrix"]))
        ps = np.stack(ps) # (N, 3, 4)        
        print(f'[INFO] average radius before scaling with {self.scale} = {np.linalg.norm(ps[:, :3, 3], axis=-1).mean()}')

        for i, f in enumerate(frames):
            f_path = f['file_path']

            # there are non-exist paths in fox...
            if not os.path.exists(f_path):
                continue
            
            pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale)

            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
            if self.H is None or self.W is None:
                self.H = image.shape[0] // self.downscale
                self.W = image.shape[1] // self.downscale

            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255 # [H, W, 3/4]

            if self.out_dim_color == 1:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[...,None]

            self.poses.append(pose)
            self.images.append(image)
            if self.e2vid:
                if self.mode == "esim":
                    tss_imgs_us = np.loadtxt(os.path.join(self.root_path, "images", "image_stamps_ns.txt")) / 1000
                    tss_e2vs_us = np.loadtxt(os.path.join(self.root_path, self.imgdir, "timestamps.txt"))
                elif self.mode == "eds":
                    tss_imgs_us = np.loadtxt(os.path.join(self.root_path, "images_timestamps_us.txt"))
                    tss_e2vs_us = np.loadtxt(os.path.join(self.imgdir, "timestamps.txt"))
                elif self.mode == "evo":
                    tss_imgs_us = np.loadtxt(os.path.join(self.root_path, "images_timestamps_us.txt"))
                    tss_e2vs_us = np.loadtxt(os.path.join(self.imgdir, "timestamps.txt"))
                elif self.mode == "tumvie":
                    tss_imgs_us = np.loadtxt(os.path.join(self.root_path, "left_images_undistorted/image_timestamps_left.txt")) # us
                    tss_e2vs_us = np.loadtxt(os.path.join(self.imgdir, "timestamps.txt"))

                ts_e2v = tss_e2vs_us[self.frame_idxs[i]]
                idx_closest_gt_img = np.argmin(np.abs(ts_e2v-tss_imgs_us))
                gt_frame = cv2.imread(gts_path[idx_closest_gt_img], cv2.IMREAD_UNCHANGED)

                if self.H is None or self.W is None:
                    self.H = gt_frame.shape[0] // self.downscale
                    self.W = gt_frame.shape[1] // self.downscale

                # add support for the alpha channel as a mask.
                if gt_frame.shape[-1] == 3: 
                    gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2RGB)
                else:
                    gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_BGRA2RGBA)
                gt_frame = cv2.resize(gt_frame, (self.W, self.H), interpolation=cv2.INTER_AREA)
                gt_frame = gt_frame.astype(np.float32) / 255 # [H, W, 3/4]

                if self.out_dim_color == 1:
                    gt_frame = cv2.cvtColor(gt_frame, cv2.COLOR_RGB2GRAY)[...,None]
                self.e2vid_gts.append(gt_frame)

        self.poses = np.stack(self.poses, axis=0).astype(np.float32)
        print(f'[INFO] average radius after scaling with {self.scale} = {np.linalg.norm(self.poses[:, :3, 3], axis=-1).mean()}')
        # [debug]: uncomment
        # visualize_poses(self.poses, bound=self.bound)
