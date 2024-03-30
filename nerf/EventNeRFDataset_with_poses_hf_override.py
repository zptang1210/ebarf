import os
import cv2
import glob
import json
import tqdm
import pickle
import numpy as np
import shutil
from scipy.spatial.transform import Slerp, Rotation

import pathvalidate
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

from .EventNeRFDataset import *

class EventNeRFDataset_with_poses_hf_override(EventNeRFDataset):
    def __init__(self, opt, poses_hf_dict_for_override, device, type='train', downscale=1, n_test=10, select_frames=None, cached_data=None):
        """
        Input
        events_in: (N, 5)
        
        Returns: 
        saves self.events = (N, 5) where events are sorted per pixel (and in time). 
        saves self.poses_evs = (N, 3, 4)
        saves self.poses_hf: k-mocap-measuremtns {"ts_ns": scalar, "pose_c2w": (3, 4)}, where k << N
        """
        super(EventNeRFDataset, self).__init__(opt, device, type=type, downscale=downscale, n_test=n_test, select_frames=select_frames)

        self.accumulate_evs = opt.accumulate_evs
        self.batch_size_evs = opt.batch_size_evs
        self.out_dim_color = opt.out_dim_color
        evs_batches_ns_tmp, no_events = self.load_events_at_frame_idxs(opt.datadir, self.frame_idxs, mode=opt.mode)
        assert len(self.frame_idxs) == len(evs_batches_ns_tmp)

        # * comment this out since self.poses_hf will be overridden
        # ####################
        # plotting_poses_hf(self.workspace, self.poses_hf)
        # ####################
    
        self.num_evs = {}
        self.idx_no_successor = {}
        self.num_successor_evs = {}
        self.counter = 0
        
        self.events = {}
        self.poses_evs = {}
        self.no_evs = {}
        self.xy_numEvs_Idx = {}

        # Setting up Event-Pose-Interpolators
        # * override
        if poses_hf_dict_for_override is None:
            self.tss_poses_hf_ns = np.stack([p["ts_ns"] for p in self.poses_hf])  
            self.rots_hf = np.stack([p["pose_c2w"][:3, :3] for p in self.poses_hf])  
            self.trans_hf = np.stack([p["pose_c2w"][:3, 3] for p in self.poses_hf])
        else:
            new_poses_hf_size = poses_hf_dict_for_override['tss_poses_hf_ns'].shape[0]
            print(f'INFO: poses_hf is overridden... (#poses_hf = {new_poses_hf_size})')
            self.tss_poses_hf_ns = poses_hf_dict_for_override['tss_poses_hf_ns'].detach().cpu().numpy()
            self.rots_hf = poses_hf_dict_for_override['poses_hf'][:, :3, :3].detach().cpu().numpy()
            self.trans_hf = poses_hf_dict_for_override['poses_hf'][:, :3, 3].detach().cpu().numpy()
            self.poses_hf = None # * set it to avoid future access
        self.rot_interpolator = Slerp(self.tss_poses_hf_ns, R.from_matrix(self.rots_hf)) 
        self.trans_interpolator = interp1d(x=self.tss_poses_hf_ns, y=self.trans_hf, axis=0, kind="cubic", bounds_error=True)

        # * THE REST OF THE CODE OF INIT METHOD SHOULD KEEP THE SAME AS ITS PARENT CLASS!

        # [todo]: precompute once
        # * cache_data is None: we want to compute all the data anyway. cache_data is a path: we want to load the data if the path is valid, otherwise, we will compute the data and save the cache.
        # * if cached_data is None or the stored file path doesn't exists, then we have to compute all the data from scratch. After that, we can save the computed data to cache_data
        # * otherwise, we can load the cache to prevent the following time-consuming computation in the for loop
        # * first we need to check if the evs_batches_ns_tmp and no_events are the same as the one stored in the cache file
        # * then we load self.xy_numEvs_idx, self.num_evs, self.idx_no_successor, self.num_successor_evs (!={} when self.accumulate_evs is True),
        # * self.events, self.poses_evs (!={} when precompute_evs_poses is True).
        # * notice that self.no_evs (!={} when self.negative_event_sampling is True) is not cached since it's computed after the for loop.
        if cached_data is None or not os.path.exists(cached_data):
            print('* Compute init data of EventNeRFDataset...')
            evs_batches_ns_tmp_backup = list(evs_batches_ns_tmp) # * evs_batches_ns_tmp will shrink in each for iteration, back it up for caching.
            print(f"Starting to compute {len(evs_batches_ns_tmp)} evs_dict_xy")
            N_evs_batches = len(evs_batches_ns_tmp)
            for i in range(N_evs_batches):
                current_frame = self.frame_idxs[i]

                events_in = evs_batches_ns_tmp.pop(0)
                events_in = events_in.astype(np.float32)
                events_in = np.asarray(sorted(events_in, key=lambda x: x[2]))   

                # create evs_dict_xy with key: (x,y) and value: ev-tuple (x, y, z, t_ns, pol)
                evs_dict_xy = {}
                for ev in events_in:
                    key_xy = (ev[0], ev[1])
                    if key_xy in evs_dict_xy.keys():
                        evs_dict_xy[key_xy].append(ev.tolist())
                    else:
                        evs_dict_xy[key_xy] = [ev.tolist()]
                # filter dictonary s.t. > 1 ev per pixel
                evs_dict_xy = dict((k, v) for k, v in evs_dict_xy.items() if len(v) > 1) 
                # del events_in # * disable this because evs_batches_ns_tmp_backup is not deep copied
                
                # compute pair of (numEvs, Index) for each pixel (where there is >1 event)
                xys_mtNevs = list(evs_dict_xy.keys())
                num_evs_at_xy = np.asarray([len(evs_dict_xy[xy]) for xy in xys_mtNevs])
                xys_mtNevs = np.asarray(xys_mtNevs).astype(np.uint32)
                self.xy_numEvs_Idx[current_frame] = np.concatenate((num_evs_at_xy[:, None], np.append(0, np.cumsum(num_evs_at_xy)[:-1])[:, None]), axis=1)
                assert np.all(num_evs_at_xy > 1)

                # save the Index of last event at pixel xy
                cumnum_evs_at_xy = np.cumsum(num_evs_at_xy) # (M)
                self.num_evs[current_frame] = cumnum_evs_at_xy[-1]
                # idx_no_successor is index (in [0, num_evs-1]) for which there is no following event at same xy
                self.idx_no_successor[current_frame] = cumnum_evs_at_xy - 1 # (M)
                
                if self.accumulate_evs:
                    num_successor_evs = np.zeros(self.num_evs[current_frame]).astype(np.int64)
                    j = 0
                    for id in range(self.num_evs[current_frame]):
                        if id >= cumnum_evs_at_xy[j]:
                            j += 1
                        num_successor_evs[id] = cumnum_evs_at_xy[j] - id - 1 # -1 to substract itself. np.Tensor (M,)
                    self.num_successor_evs[current_frame] = num_successor_evs
                
                # flatten evs_dict_xy to linear self.events np.array (N, 5) 
                for xy in list(evs_dict_xy.keys()):
                    evs = evs_dict_xy[xy]
                    for ev in evs:
                        if current_frame in self.events:
                            self.events[current_frame].append(ev)
                        else:
                            self.events[current_frame] = [ev]
                    del evs_dict_xy[xy]  # delete each key, to keep max-memory low

                evs_dict_xy.clear()
                del evs_dict_xy
                self.events[current_frame] = np.asarray(self.events[current_frame]).astype(np.float32)
                assert self.num_evs[current_frame] == self.events[current_frame].shape[0]

                if self.precompute_evs_poses:
                    # [alternative] option2: pre-interpolate (fast, but large memory requirement)
                    eval_tss_evs_ns = self.events[current_frame][:, 2].copy() 
                    rots = self.rot_interpolator(eval_tss_evs_ns).as_matrix().astype(np.float32) 
                    trans = self.trans_interpolator(eval_tss_evs_ns).astype(np.float32)
                    # [debug]: uncomment to plot event-poses
                    # plotting_poses_evs(self.workspace, rots, trans, eval_tss_evs_ns)
                    N = rots.shape[0]
                    pose_N_3_4 = np.zeros((N, 3, 4)).astype(np.float32)
                    pose_N_3_4[:N, :3, :3] = rots.copy().astype(np.float32)  # (N, 3, 3)
                    pose_N_3_4[:N, :3, 3:4] = np.expand_dims(trans, axis=-1).copy().astype(np.float32) # (N, 3, 1)
                    self.poses_evs[current_frame] = pose_N_3_4.copy()
                    del rots
                    del trans
                    del eval_tss_evs_ns
                print(f"Batch {i+1}/{(N_evs_batches)} dict from events and interpolated poses per event")
        
            # * save the above-mentioned variables into a cache file
            if cached_data is not None:
                print('* Save the computed init data to cache file for future reusage...')
                cache = {'evs_batches_ns_tmp': evs_batches_ns_tmp_backup,
                         'no_events': no_events,
                         'xy_numEvs_Idx': self.xy_numEvs_Idx,
                         'num_evs': self.num_evs,
                         'idx_no_successor': self.idx_no_successor,
                         'num_successor_evs': self.num_successor_evs,
                         'events': self.events,
                         'poses_evs': self.poses_evs
                         }
                with open(cached_data, 'wb') as fout:
                    pickle.dump(cache, fout)
                
        else:
            # * load the cache
            print('* Load the cached init data for EventNeRFDataset...')
            with open(cached_data, 'rb') as fin:
                cache = pickle.load(fin)

            # * check if the saved cache data matches the current session
            print('* Check if the saved cache data matches the current session...')
            evs_batches_ns_tmp_pkl = cache['evs_batches_ns_tmp']
            if len(evs_batches_ns_tmp_pkl) == len(evs_batches_ns_tmp):
                are_equal = all(np.array_equal(a, b) for a, b in zip(evs_batches_ns_tmp_pkl, evs_batches_ns_tmp))
                assert are_equal, 'The cached file does not match the current session due to unmatched evs_batches_ns.'

            if cache['no_events'] is not None:
                # todo
                raise NotImplementedError('not implemented the module for negative sampling yet...')

            # * load data
            self.evs_batches_ns_tmp = cache['evs_batches_ns_tmp']
            self.no_events = cache['no_events']
            self.xy_numEvs_Idx = cache['xy_numEvs_Idx']
            self.num_evs = cache['num_evs']
            self.idx_no_successor = cache['idx_no_successor']
            self.events = cache['events']
            self.poses_evs = cache['poses_evs']

        # float32-cast
        if self.negative_event_sampling:
            self.no_evs = no_events
            for fid, _ in self.no_evs.items():
                for k in self.no_evs[fid]:
                    if k == "coords":
                        for j in range(len(self.no_evs[fid][k])):
                            self.no_evs[fid][k][j] = self.no_evs[fid][k][j].astype(np.float32) # (N, 3, 4)
                            self.no_evs[fid][k][j] = torch.from_numpy(self.no_evs[fid][k][j]).to(device)
                    elif k == "tss_bds":
                        for kk, _ in self.no_evs[fid][k].items():
                            for j in range(len(self.no_evs[fid][k][kk])):
                                self.no_evs[fid][k][kk] = np.asarray(self.no_evs[fid][k][kk]).astype(np.float32)

        # float32-cast
        for key, evs in self.events.items():
            self.events[key] = self.events[key].astype(np.float32) # (N, 4)

        # Preloading event data to GPU
        for key, evs in self.events.items():
            evs_batch = torch.from_numpy(evs).to(device)
            self.events[key] = evs_batch

        if self.precompute_evs_poses:
            # float32-cast
            for key, evs in self.events.items():
                self.poses_evs[key] = self.poses_evs[key].astype(np.float32) # (N, 3, 4)
            
            # Preloading pose data to GPU
            for key, poses in self.poses_evs.items():
                poses_batch = torch.from_numpy(np.asarray(poses)).to(device)
                self.poses_evs[key] = poses_batch

    def collate(self, index):
        results = super().collate(index)
        # * rays for images are based on the original poses_hf not being overriden. should not use it!
        del results['rays_o']
        del results['rays_d']
        return results