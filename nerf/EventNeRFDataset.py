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
import hdf5plugin
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

#####################
# Loading esim 
#####################
def load_contiguous_evs_batches_esim_ns(eventdir, idxs, us=False, hwf=None):
    """
    Inputs:
    eventdir: Dir with npys
    idxs: List of N indices [idx1, ..., idxN], specifies which batches to load.
    
    Description:
    Function collects N event-batches, using all intermediate events.
    Checks if pols in (-1, 1), checks if xy in (HxW). Timestamps in nanosecond. 

    Output:
    list of np.arrays (num_evs_batch, 4) where 4 =  (x, y, ts_ns, p)
    """

    assert len(idxs) > 0
    event_npys = [os.path.join(eventdir, f) for f in sorted(os.listdir(eventdir)) if f.endswith(".npy")]

    if len(idxs) == 1:
        event_batches = [np.load(event_npys[idxs[0]])]
    else:
        event_batches = []
        diff_idxs = np.diff(idxs) # (N-1)
        assert np.all(diff_idxs>0)
        for i, diff_idx in enumerate(diff_idxs):
            sub_batches = []
            # collecting sub-batches, in (idxs[i], idxs[i]+1, ..., idxs[i]+diff_idx[i]-1)
            for k in range(0, diff_idx, 1): 
                evs = np.load(event_npys[idxs[i] + k])
                sub_batches.append(evs)
            event_batches.append(np.concatenate((sub_batches)))
            print(f"loaded event batch {i}/{len(diff_idxs)}")
        if idxs[i] + k < idxs[-1]:
            event_batches.append(np.load(event_npys[idxs[-1]]))
            print(f"loaded (remaining) event batch {i+1}/{len(diff_idxs)}")
        assert len(idxs) == len(event_batches)

    event_batches = [evs[:, :4] for evs in event_batches]
    event_batches = np.asarray(event_batches, dtype=object)
    check_evs_shapes(event_batches, tuple_size=4)

    if us:
        mask = (1, 1, 1000.0, 1)  # (x, y, ts_us, p)
        event_batches = [ev * mask for ev in event_batches] # convert to ns

    if hwf is not None:
        check_evs_coord_range(event_batches, W=hwf[1], H=hwf[0])
    else:
        check_evs_coord_range(event_batches, W=1280, H=720)
    if should_transform_pol(event_batches):
        event_batches = transform_pol(event_batches)

    # transform polarity into (-1, 1)
    check_evs_pol(event_batches, pol_neg=-1, pol_pos=1)
    check_evs_shapes(event_batches, tuple_size=4)

    return event_batches.tolist()

def load_event_data_esim(datadir, idxs, hwf=None, img_folder="images"):
    # Loading events
    eventdir = os.path.join(datadir, "events")
    if not os.path.exists(eventdir):
        print(eventdir, "does not exist!")
        sys.exit()
    events_ns = load_contiguous_evs_batches_esim_ns(eventdir, idxs, us=False, hwf=hwf)
    total_num_evs = np.array([events_ns[i].shape[0] for i in range(len(events_ns))]).sum()
    print(f"loaded events with first batch´s shape {events_ns[0].shape}. Total_num_evs = {total_num_evs/1e6:.3f} million")
    return events_ns

#####################
# Loading tumvie
#####################
def load_event_data_tumvie(path, idxs, hotpixs=False, H=720, W=1280, img_folder="left_images"):
    idxss = sorted(idxs)

    if "left" in img_folder:
        suffix = "left"
    else:
        suffix = "right"
    if hotpixs:
        suffix = f"{suffix}_hotpixs"    
    
    # load events
    h5file = glob.glob(os.path.join(path, f'*events_{suffix}.h5'))[0]
    evs_h5 = h5py.File(h5file, "r")

    # load undistortion
    h5file = glob.glob(os.path.join(path, f'*rectify_map_{suffix}.h5'))[0]
    rmap = h5py.File(h5file, "r")
    rectify_map = np.array(rmap["rectify_map"])
    rmap.close()

    # load timestamps
    tss_imgs_us = np.loadtxt(os.path.join(path, img_folder, f"image_timestamps_{suffix}.txt"))
    dT_ms_trigger_period = np.diff(tss_imgs_us).mean()/1e3 
    assert dT_ms_trigger_period > 3 and dT_ms_trigger_period < 100 # dt_ms on tumvie is 50ms
    tss_imgs_us = tss_imgs_us[idxss]

    # compute center timestamps (events associated with image at time t0 are taken from (t0 - 0.5dT, t0 + 0.5dT))
    tss_evs_centers_us = np.insert(tss_imgs_us, 0, tss_imgs_us[0]-2*dT_ms_trigger_period*1e3)
    tss_evs_centers_us = np.insert(tss_evs_centers_us, len(tss_evs_centers_us), tss_evs_centers_us[-1]+2*dT_ms_trigger_period*1e3)
    tss_evs_centers_us = tss_evs_centers_us[:-1] + np.diff(tss_evs_centers_us)/2.
    assert np.all(np.diff(tss_evs_centers_us)>0)

    event_slicer = EventSlicer(evs_h5)
    print(f"Events span from {event_slicer.get_start_time_us()/1e6:.3f}secs to {event_slicer.get_final_time_us()/1e6:.3f}secs")
    evs_out, durs_ms, evs_hists, evs_hists_undist, coords = [], [], [], [], []
    pos, neg = 0, 0
    dT_us = 0
    
    # for very long event durations (> max_dT_us), subsample events, since tumvie is high resolution
    ev_window_dT_us = (tss_evs_centers_us[-1] - tss_evs_centers_us[0])
    max_dT_us = 10*1e6 
    if ev_window_dT_us > max_dT_us:
        no_evs_dT_us = ev_window_dT_us - max_dT_us
        dT_us = no_evs_dT_us / (2 * len(idxss)) # assumes equal event window selection
        print(f"Not using all events due to memory constraints! \
            \nUsing dT_us={dT_us*1e-3:.3f}ms since requested ev-window is {ev_window_dT_us*1e-6:.3f}secs long \
            but can use maximum of max_dT {max_dT_us*1e-6:.3f}secs.")

    for i, ts_us in enumerate(tss_imgs_us):
        start_time_us = tss_evs_centers_us[i] + dT_us
        end_time_us = tss_evs_centers_us[i+1] - dT_us

        durs_ms.append(end_time_us/1e3-start_time_us/1e3)
        ev_batch = event_slicer.get_events(start_time_us, end_time_us)
        assert durs_ms[-1] > 0
        assert np.abs(ev_batch["t"][-1]-end_time_us) <= 50
        assert np.abs(ev_batch["t"][0]-start_time_us) <= 50
    
        N = len(ev_batch["t"])
        coord = np.zeros((N, 2))
        coord[:, 0] = ev_batch["x"]
        coord[:, 1] = ev_batch["y"]
        tmp = np.zeros((N, 4))
        rect = rectify_map[ev_batch["y"], ev_batch["x"]]
        tmp[:, 0] = rect[..., 0] 
        tmp[:, 1] = rect[..., 1]  
        tmp[:, 2] = ev_batch["t"] * 1000 # nanosec
        tmp[:, 3] = ev_batch["p"]
        tmp[:, 3] = tmp[:, 3] * 2 - 1
        pos += np.sum(tmp[:, 3]>0)
        neg += np.sum(tmp[:, 3]<0)
        assert ev_batch["x"].min() >= 0.0
        assert ev_batch["x"].max() <= W-1
        assert ev_batch["y"].min() >= 0.0
        assert ev_batch["y"].max() <= H-1
        assert np.all(tmp[:, 2] > 0.0)
        print(f"median x-deviation of undistorting event camera: {np.median(np.abs(ev_batch['x']-rect[..., 0]))}")
        print(f"median y-deviation of undistorting event camera: {np.median(np.abs(ev_batch['y']-rect[..., 1]))}")

        img = render_ev_accumulation(ev_batch["x"], ev_batch["y"], ev_batch["p"], H, W)
        evs_hists.append(img)
        img = render_ev_accumulation(tmp[:, 0], tmp[:, 1], ev_batch["p"], H, W)
        evs_hists_undist.append(img)

        evs_out.append(tmp)
        coords.append(coord)
        print(f"Got {tmp.shape[0]/1e6} million events per {durs_ms[i]}ms (in ({(start_time_us)/1e6}, {(end_time_us)/1e6})), \
               centered at frame {idxss[i]} ({(tss_imgs_us[i])/1e6} secs). pos/neg = {np.sum(tmp[:, 3]>0)/np.sum(tmp[:, 3]<0)})")
    evs_h5.close()

    check_evs_pol(evs_out, pol_neg=-1, pol_pos=1, idx_pol=3)
    hists = {"hists": evs_hists, "hists_undist": evs_hists_undist}
    posneg = pos/neg

    print(f"Duration (ms) expected vs. measured: {np.abs(np.asarray(durs_ms).sum() - (tss_imgs_us[-1]-tss_imgs_us[0])/1e3 - 100)} ms")
    print(f"Got total events of {np.asarray(durs_ms).sum()}ms, with pos/neg = {posneg}")
    
    return evs_out, hists, coords, rectify_map, tss_evs_centers_us

#####################
# Loading eds
#####################
def load_event_data_EDS(path, idxs, calibstr, hotpixs=False, H=480, W=640):
    idxss = sorted(idxs)
    
    # loading evs
    h5file = os.path.join(path, 'events.h5')
    if hotpixs:
        h5file = glob.glob(os.path.join(path, 'events_hotpixs_*.h5'))[0]
    evs = h5py.File(h5file, "r")
    event_slicer = EventSlicer(evs)
    print(f"Total {(event_slicer.get_start_time_us()-event_slicer.t_offset)/1e6}secs \
           to {(event_slicer.get_final_time_us()-event_slicer.t_offset)/1e6}secs.")

    # loadings undistortion
    h5file = glob.glob(os.path.join(path, f'rectify_map_{calibstr}.h5'))[0]
    rmap = h5py.File(h5file, "r")
    rectify_map = np.array(rmap["rectify_map"])  # (H, W, 2)
    rmap.close()

    tss_imgs_us = np.loadtxt(os.path.join(path, "images_timestamps_us.txt"))
    dT_ms_trigger_period = np.diff(tss_imgs_us).mean()/1e3
    assert dT_ms_trigger_period > 3 and dT_ms_trigger_period < 50
    assert tss_imgs_us[0] - (evs["t"][0]) < 1e6 and tss_imgs_us[0] - (evs["t"][0]) > 0
    assert tss_imgs_us[-1] - (evs["t"][-1]) < 1e6 and tss_imgs_us[-1] - (evs["t"][-1]) > 0
    tss_imgs_us = tss_imgs_us[idxss] # * corrected this bug by removing extra []
    tss_evs_centers_us = np.insert(tss_imgs_us, 0, tss_imgs_us[0]-2*dT_ms_trigger_period*1e3)
    tss_evs_centers_us = np.insert(tss_evs_centers_us, len(tss_evs_centers_us), tss_evs_centers_us[-1]+2*dT_ms_trigger_period*1e3)
    tss_evs_centers_us = tss_evs_centers_us[:-1] + np.diff(tss_evs_centers_us)/2.
    assert np.all(np.diff(tss_evs_centers_us)>0)

    coords, evs_out, durs_ms, evs_hists, evs_hists_undist = [], [], [], [], []
    pos, neg = 0, 0
    for i, ts_us in enumerate(tss_imgs_us):
        start_time_us = tss_evs_centers_us[i]
        end_time_us = tss_evs_centers_us[i+1]
        durs_ms.append(end_time_us/1e3-start_time_us/1e3)
        ev_batch = event_slicer.get_events(start_time_us, end_time_us)
        if ev_batch is None:
            print(f"Found no events in {(start_time_us)/1e6:.3f}secs to {(end_time_us)/1e6:.3f}secs ({durs_ms[i]:.3f} ms duration) at frame {idxss[i]}.jpg")
            continue
        assert np.abs(ev_batch["t"][-1]-end_time_us) <= 50
        assert np.abs(ev_batch["t"][0]-start_time_us) <= 900
            
        N = len(ev_batch["t"])
        tmp = np.zeros((N, 4))
        rect = rectify_map[ev_batch["y"], ev_batch["x"]]
        tmp[:, 0] = rect[..., 0]
        tmp[:, 1] = rect[..., 1]
        tmp[:, 2] = (ev_batch["t"]) * 1000 # us -> nanosecs
        tmp[:, 3] = ev_batch["p"]
        tmp[:, 3] = tmp[:, 3] * 2 - 1
        pos += np.sum(tmp[:, 3]>0)
        neg += np.sum(tmp[:, 3]<0)
        coord = np.zeros((N, 2))
        coord[:, 0] = ev_batch["x"] 
        coord[:, 1] = ev_batch["y"] 
        assert ev_batch["x"].min() >= 0.0
        assert ev_batch["x"].max() <= W-1
        assert ev_batch["y"].min() >= 0.0
        assert ev_batch["y"].max() <= H-1
        assert np.all(tmp[:, 2] >= 0.0)
        print(f"median x-deviation of undistorting event camera: {np.median(np.abs(ev_batch['x']-rect[..., 0]))}")
        print(f"median y-deviation of undistorting event camera: {np.median(np.abs(ev_batch['y']-rect[..., 1]))}")

        img = render_ev_accumulation(ev_batch["x"], ev_batch["y"], ev_batch["p"], H, W)
        evs_hists.append(img)
        img = render_ev_accumulation(tmp[:, 0], tmp[:, 1], ev_batch["p"], H, W)
        evs_hists_undist.append(img)

        evs_out.append(tmp)
        coords.append(coord)
        print(f"Got {tmp.shape[0]/1e6:{3}.{2}} million events per {durs_ms[i]:{4}.{3}} ms (in ({(start_time_us)/1e6}, {(end_time_us)/1e6})), centered at frame {idxss[i]} ({(tss_imgs_us[i])/1e6} secs. pos/neg = {np.sum(tmp[:, 3]>0)/np.sum(tmp[:, 3]<0)})")
    evs.close()

    check_evs_pol(evs_out, pol_neg=-1, pol_pos=1, idx_pol=3)
    hists = {"hists": evs_hists, "hists_undist": evs_hists_undist}
    posneg = pos/neg
    print(f"Got total events of {np.asarray(durs_ms).sum()} milisecs, with pos/neg = {posneg}")
    
    return evs_out, hists, coords, rectify_map, tss_evs_centers_us


class EventNeRFDataset(NGPDataset):
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, select_frames=None, cached_data=None):
        """
        Input
        events_in: (N, 5)
        
        Returns: 
        saves self.events = (N, 5) where events are sorted per pixel (and in time). 
        saves self.poses_evs = (N, 3, 4)
        saves self.poses_hf: k-mocap-measuremtns {"ts_ns": scalar, "pose_c2w": (3, 4)}, where k << N
        """
        super().__init__(opt, device, type=type, downscale=downscale, n_test=n_test, select_frames=select_frames)

        self.accumulate_evs = opt.accumulate_evs
        self.batch_size_evs = opt.batch_size_evs
        self.out_dim_color = opt.out_dim_color
        evs_batches_ns_tmp, no_events = self.load_events_at_frame_idxs(opt.datadir, self.frame_idxs, mode=opt.mode)
        assert len(self.frame_idxs) == len(evs_batches_ns_tmp)

        ####################
        plotting_poses_hf(self.workspace, self.poses_hf)
        ####################
    
        self.num_evs = {}
        self.idx_no_successor = {}
        self.num_successor_evs = {}
        self.counter = 0
        
        self.events = {}
        self.poses_evs = {}
        self.no_evs = {}
        self.xy_numEvs_Idx = {}

        # Setting up Event-Pose-Interpolators
        self.tss_poses_hf_ns = np.stack([p["ts_ns"] for p in self.poses_hf])  
        self.rots_hf = np.stack([p["pose_c2w"][:3, :3] for p in self.poses_hf])  
        self.trans_hf = np.stack([p["pose_c2w"][:3, 3] for p in self.poses_hf]) 
        self.rot_interpolator = Slerp(self.tss_poses_hf_ns, R.from_matrix(self.rots_hf)) 
        self.trans_interpolator = interp1d(x=self.tss_poses_hf_ns, y=self.trans_hf, axis=0, kind="cubic", bounds_error=True)

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
            print(cache.keys())
            self.evs_batches_ns_tmp = cache['evs_batches_ns_tmp']
            self.no_events = cache['no_events']
            self.xy_numEvs_Idx = cache['xy_numEvs_Idx']
            self.num_evs = cache['num_evs']
            self.idx_no_successor = cache['idx_no_successor']
            self.events = cache['events']
            self.poses_evs = cache['poses_evs']

        # * redundant
        # # Setting up Event-Pose-Interpolators
        # self.tss_poses_hf_ns = np.stack([p["ts_ns"] for p in self.poses_hf])  
        # self.rots_hf = np.stack([p["pose_c2w"][:3, :3] for p in self.poses_hf])  
        # self.trans_hf = np.stack([p["pose_c2w"][:3, 3] for p in self.poses_hf]) 
        # self.rot_interpolator = Slerp(self.tss_poses_hf_ns, R.from_matrix(self.rots_hf)) 
        # self.trans_interpolator = interp1d(x=self.tss_poses_hf_ns, y=self.trans_hf, axis=0, kind="cubic", bounds_error=True)

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

    def load_events_at_frame_idxs(self, path, idxs, mode, hwf=None):
        if self.images_corrupted and self.type == 'train':
            img_folder = "images_corrupted"
        else:
            img_folder = "images" # use non-blurred images for eval
        
        rectify_map = np.stack(np.meshgrid(np.arange(self.W_ev), np.arange(self.H_ev)), axis=2)
        if mode == "esim":
            evs_batches_ns = load_event_data_esim(path, idxs, hwf=hwf, img_folder=img_folder)
            tss_centers_us = [1e-3*(evs[0, 2]) for evs in evs_batches_ns]
            tss_centers_us.append(evs_batches_ns[-1][-1, 2]*1e-3)
            coords = [cs[:, :2] for cs in evs_batches_ns]
        elif mode == "tumvie":
            evs_batches_ns, hists, coords, rectify_map, tss_centers_us = load_event_data_tumvie(path, idxs, self.hotpixs, self.H_ev, self.W_ev, img_folder=self.imgdir)
        elif mode == "eds":
            evs_batches_ns, hists, coords, rectify_map, tss_centers_us = load_event_data_EDS(path, idxs, self.calibstr, self.hotpixs, H=self.H_ev, W=self.W_ev)
        else: 
            sys.exit()
        self.rectify_map = rectify_map

        # Compute no_events per event-batch: locations and respective time interval (t0,t1)
        no_evs_out = None
        if self.negative_event_sampling:
            # no-event is defined in chunk_len_ms (e.g. 20ms to 100ms) timeinterval 
            # if this window is too small (e.g.1ms), it becomes meaningless,
            # because then every pixel is no-event and L2=L1=0 is valid solution
            # if too big, then almost no no-events detected.
            chunk_len_ms = 20
                        
            no_evs_out = {}
            durs_ms = []
            for i in range(len(evs_batches_ns)):
                fidx = self.frame_idxs[i]
                no_evs_out[fidx] = {}
                no_evs_out[fidx]["coords"] = []
                no_evs_out[fidx]["tss_bds"] = {}
                no_evs_out[fidx]["tss_bds"]["N_ev_chunks"] = []
                no_evs_out[fidx]["tss_bds"]["start_time_us"] = []
                no_evs_out[fidx]["tss_bds"]["end_time_us"] = []
                no_evs_out[fidx]["tss_bds"]["dt_us"] = []

                start_time_us = tss_centers_us[i]
                end_time_us = tss_centers_us[i+1]
                assert end_time_us > start_time_us
                durs_ms.append(end_time_us/1e3-start_time_us/1e3)

                # sub-chunks for no-event search
                N_ev_chunks = int(durs_ms[i]/chunk_len_ms) + 1
                dt_us = 1e3*durs_ms[i]/N_ev_chunks

                xsall = coords[i][:, 0].astype(np.uint32)
                ysall = coords[i][:, 1].astype(np.uint32)
                ts_iter = start_time_us
                for j in range(N_ev_chunks):    
                    ts_mask = (evs_batches_ns[i][:, 2]*1e-3 >= ts_iter) & (evs_batches_ns[i][:, 2]*1e-3 < (ts_iter+dt_us))
                    xstmp = xsall[ts_mask]
                    ystmp = ysall[ts_mask]
                    if N_ev_chunks == 1:
                        assert (ts_mask-np.ones(len(evs_batches_ns[i][:, 2]))).sum() <= 1

                    # idxs_no_evs linearly save no-event-location as (1...HW)
                    idxs_no_evs = np.linspace(1, self.H_ev*self.W_ev, self.H_ev*self.W_ev).astype(np.uint32)
                    idxs_evs = ystmp * self.W_ev + xstmp # in (0...HW-1)
                    idxs_no_evs[idxs_evs] = 0 # mark the event locations with 0s
                    N_noevs = (idxs_no_evs>0).sum()
                    if len(np.unique(idxs_evs))  == self.W_ev*self.H_ev:
                        assert N_noevs == 0
                    print(f"Total of {N_noevs/(self.H_ev*self.W_ev):.3f} no-events per pixel at frame {fidx}_subchunk_{j}")
                    
                    # keep only the no_evs
                    idxs_no_evs = idxs_no_evs[idxs_no_evs>0]
                    # subsample no_evs adaptively, due to OOM: if many no-events, we keep many. if many chunks, we keep few per chunk.
                    idxs_no_evs = np.random.choice(idxs_no_evs, size=int(N_noevs/N_ev_chunks), replace=False) 
                    N_noevs = (idxs_no_evs>0).sum()
                    print(f"We keep around {N_noevs/(self.H_ev*self.W_ev)} no-events per pixel.")

                    ys, xs = (idxs_no_evs-1) // self.W_ev, (idxs_no_evs-1) % self.W_ev # ys,xs in (0...HW-1)
                    rect = rectify_map[ys, xs]  # (N_noevs, 2)  
                    no_evs_batch = np.zeros((N_noevs, 2))
                    no_evs_batch[:, 0] = rect[:, 0]
                    no_evs_batch[:, 1] = rect[:, 1]
                    if len(no_evs_batch) == 0:
                        no_evs_batch = np.zeros((1, 2)) # 1 dummy event to not break downstream-code

                    no_evs_out[fidx]["coords"].append(no_evs_batch)
                    no_evs_out[fidx]["tss_bds"]["start_time_us"].append(ts_iter)
                    no_evs_out[fidx]["tss_bds"]["end_time_us"].append(ts_iter+dt_us)
                    ts_iter += dt_us

                no_evs_out[fidx]["tss_bds"]["N_ev_chunks"].append(N_ev_chunks)
                no_evs_out[fidx]["tss_bds"]["dt_us"].append(dt_us)

        if mode == "tumvie" or mode == "eds":
            # [debug]: visualize distorted and undisorted events
            os.makedirs(os.path.join(self.workspace, "loaded_events_undist_viz"), exist_ok=True)
            N_hists = len(hists["hists"])
            for i in range(N_hists):
                cv2.imwrite(os.path.join(self.workspace, "loaded_events_undist_viz", "%06d" % self.frame_idxs[N_hists-i-1] + ".png"), hists["hists"].pop())
                cv2.imwrite(os.path.join(self.workspace, "loaded_events_undist_viz", "%06d_undist" % self.frame_idxs[N_hists-i-1] + ".png"), hists["hists_undist"].pop())
            
        print("Done with event and pose query.")
        return evs_batches_ns, no_evs_out

    def collate(self, index):
        B = len(index)
        fidx = self.frame_idxs[index[0]] 

        if self.accumulate_evs:
            eidx = np.random.randint(0, self.num_evs[fidx], (self.batch_size_evs)) # (M == self.batch_size_evs)
            # filter events with no successor (temporally last events at a pixel in a event-batch)
            eidx = np.asarray([eidx[i]-1 if (eidx[i] in self.idx_no_successor[fidx]) else eidx[i] for i in range(len(eidx))])

            # sample random (more widespread) event from interval
            eidx_end = []
            sum_pols = []
            for ev_id_start in eidx:
                num_successors = self.num_successor_evs[fidx][ev_id_start]
                if self.acc_max_num_evs:
                    num_successors = np.minimum(num_successors, self.acc_max_num_evs+1)
            
                # ev_id_start+1: at least 1 event apart. randint is [, ).
                ev_id_end = np.random.randint(ev_id_start+1, ev_id_start+1+num_successors, (1))[0]
                # [alternative]: use fixed accumulation windows
                # ev_id_end = ev_id_start+num_successors 

                ps = self.events[fidx][(ev_id_start+1):(ev_id_end+1), 3] # ev_id_end >= ev_id_start + 1
                sum_pols.append(ps.sum())
                eidx_end.append(ev_id_end)

                # [debug]
                # assert num_successors > 0
                # assert ev_id_end >= (ev_id_start+1)
                # assert len(ps) >= 1
                # assert self.events[fidx][ev_id_start, 0] == self.events[fidx][ev_id_end, 0]
                # assert self.events[fidx][ev_id_start, 1] == self.events[fidx][ev_id_end, 1]
                # assert self.events[fidx][ev_id_start, 2] < self.events[fidx][ev_id_end, 2]

            pols = torch.stack(sum_pols).unsqueeze(0) # (1, M)
            eidx_end = np.asarray(eidx_end) # (M,)
        else:
            num_evs_xy = self.xy_numEvs_Idx[fidx][:, 0]
            eidx = (np.random.rand(num_evs_xy.shape[0]) * num_evs_xy - 1).astype(int) + self.xy_numEvs_Idx[fidx][:, 1]
            eidx = np.random.choice(eidx, size=self.batch_size_evs, replace=self.batch_size_evs>len(eidx))
            eidx_end = eidx + 1 # take direct successor event by default
            pols = self.events[fidx][eidx+1, 3].unsqueeze(0)

        xs = self.events[fidx][eidx, 0].unsqueeze(0) # (1, M)
        ys = self.events[fidx][eidx, 1].unsqueeze(0) # (1, M)

        if not self.precompute_evs_poses:
            # [alternative] Option1 (slow): computing poses online
            eval_tss_evs_ns = self.events[fidx][eidx, 2].detach().cpu() 
            rots = self.rot_interpolator(eval_tss_evs_ns).as_matrix()
            trans = self.trans_interpolator(eval_tss_evs_ns)
            poses1 = torch.Tensor(get_hom_trafos(rots, trans)[:, :3, :]).to(self.device).unsqueeze(0)

            eval_tss_evs_ns = self.events[fidx][eidx_end, 2].detach().cpu() 
            rots = self.rot_interpolator(eval_tss_evs_ns).as_matrix()
            trans = self.trans_interpolator(eval_tss_evs_ns)
            poses2 = torch.Tensor(get_hom_trafos(rots, trans)[:, :3, :]).to(self.device).unsqueeze(0)
        else:
            # [alternative] option2: pre-interpolate (fast, but large memory requirement)
            poses1 = self.poses_evs[fidx][eidx, ...].unsqueeze(0) # (1, M, 3, 4)
            poses2 = self.poses_evs[fidx][eidx_end, ...].unsqueeze(0) # (1, M, 3, 4)

        rays_evs = get_event_rays(xs, ys, poses1, poses2, self.intrinsics_evs) # (B, Nevs, 3)
        poses = self.poses[index].to(self.device) # [B, 4, 4]
        error_map = None if self.error_map is None else self.error_map[index]
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'rays_evs_o1': rays_evs["rays_evs_o1"], 
            'rays_evs_d1': rays_evs["rays_evs_d1"], 
            'rays_evs_o2': rays_evs["rays_evs_o2"], 
            'rays_evs_d2': rays_evs["rays_evs_d2"],
            'pols': pols,
        }

        if self.negative_event_sampling:
            N_noevs = int(self.batch_size_evs * 0.5)
            N_chunks_noevs = int(self.no_evs[fidx]["tss_bds"]["N_ev_chunks"][0])
            chunk_j = np.random.randint(0, N_chunks_noevs)
            N_noevs_j = self.no_evs[fidx]["coords"][chunk_j].shape[0]

            neidx = np.random.randint(0, N_noevs_j, (N_noevs)) # (N_noevs)
            xsno = self.no_evs[fidx]["coords"][chunk_j][neidx,0].unsqueeze(0) # (1, N_noevs)
            ysno = self.no_evs[fidx]["coords"][chunk_j][neidx,1].unsqueeze(0) # (1, N_noevs)

            # get time interval for jth chunk
            t0_us_j, t1_us_j = self.no_evs[fidx]["tss_bds"]["start_time_us"][chunk_j], self.no_evs[fidx]["tss_bds"]["end_time_us"][chunk_j]
            dt_us_j = t1_us_j - t0_us_j
            # sample random start and end times
            tss_sampled = t0_us_j + dt_us_j * np.random.random((N_noevs, 2))
            tss_sampled = np.sort(tss_sampled, axis=1)

            # get poses at t1
            tss1 = tss_sampled[:, 0] * 1000
            rots1 = self.rot_interpolator(tss1).as_matrix()
            trans1 = self.trans_interpolator(tss1)
            poses_no_evs1 = torch.from_numpy(get_hom_trafos(rots1, trans1)[:, :3, :].astype(np.float32)).to(self.device)

            # get poses at t2
            tss2 = tss_sampled[:, 1] * 1000
            rots2 = self.rot_interpolator(tss2).as_matrix()
            trans2 = self.trans_interpolator(tss2)
            poses_no_evs2 = torch.from_numpy(get_hom_trafos(rots2, trans2)[:, :3, :].astype(np.float32)).to(self.device)

            rays_noevs = get_event_rays(xsno, ysno, poses_no_evs1.unsqueeze(0), poses_no_evs2.unsqueeze(0), self.intrinsics_evs) # (B, Nevs, 3)
            results["rays_no_evs_o1"] = rays_noevs["rays_evs_o1"]
            results["rays_no_evs_d1"] = rays_noevs["rays_evs_d1"]
            results["rays_no_evs_o2"] = rays_noevs["rays_evs_o2"]
            results["rays_no_evs_d2"] = rays_noevs["rays_evs_d2"]

            # [debug]: uncomment to visualize no-event-coordinates
            # save_path_cor2 = os.path.join(self.workspace, "validation", 'no_evs_locations', f'fid_{fidx}_chunk_{chunk_j:04d}_{self.counter}.png')
            # self.counter += 1
            # os.makedirs(os.path.dirname(save_path_cor2), exist_ok=True) 
            # xss = torch.cat((xsno.squeeze(), xs.squeeze())).detach().cpu()
            # yss = torch.cat((ysno.squeeze(), ys.squeeze())).detach().cpu()
            # pols = torch.cat((torch.zeros_like(xsno.squeeze()), torch.ones_like(xs.squeeze()))).detach().cpu()
            # no_evs_img = render_ev_accumulation(np.asarray(xss), np.asarray(yss), np.asarray(pols), self.H_ev, self.W_ev)
            # cv2.imwrite(save_path_cor2, no_evs_img)    

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results
        
    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0) 
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        return loader
    