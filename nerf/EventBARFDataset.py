import pickle
import numpy as np
import torch
from .pose_utils.camera import lie, pose
from .utils import get_rays
from .EventNeRFDataset import EventNeRFDataset

class EventBARFDataset(EventNeRFDataset):
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, select_frames=None, use_cache=True):
        super().__init__(opt, device, type=type, downscale=downscale, n_test=n_test, select_frames=select_frames, use_cache=use_cache)
        assert opt.event_only, 'only support event_only mode.'

        # * get the actual poses_hf that will be used in EBARF training
        if opt.override_poses_hf:
            assert opt.poses_hf_load_path is not None
            print(f'* override poses_hf from {opt.poses_hf_load_path}...')
            with open(opt.poses_hf_load_path, 'rb') as fin:
                poses_hf_dict_for_override = pickle.load(fin)
            print('* comment stored in the loaded poses_hf:', poses_hf_dict_for_override['comment'])
            self.poses_hf_dict_final = poses_hf_dict_for_override
        else:
            if opt.noise > 0:
                print(f'* randomly generate noised poses_hf based on {opt.noise}...')
                noised_poses_hf_dict = self._compute_noised_poses_hf(opt.noise)
                self.poses_hf_dict_final = noised_poses_hf_dict
            else:
                self.poses_hf_dict_final = self.get_gt_poses_hf_dict()

        if opt.poses_hf_save_path is not None:
            print(f'* save the final poses_hf for ebarf training to {opt.poses_hf_save_path}...')
            with open(opt.poses_hf_save_path, 'wb') as fout:
                pickle.dump(self.poses_hf_dict_final, fout)            

    def collate(self, index):
        B = len(index)
        fidx = self.frame_idxs[index[0]]

        if self.accumulate_evs:
            raise NotImplementedError('accumulate_evs is not supported yet.')
        else:
            num_evs_xy = self.xy_numEvs_Idx[fidx][:, 0]
            # todo: The first line is from the original code, but I believe the second line is correct. test it later.
            eidx = (np.random.rand(num_evs_xy.shape[0]) * num_evs_xy - 1).astype(int) + self.xy_numEvs_Idx[fidx][:, 1]
            # eidx = (np.random.rand(num_evs_xy.shape[0]) * (num_evs_xy - 1)).astype(int) + self.xy_numEvs_Idx[fidx][:, 1]
            eidx = np.random.choice(eidx, size=self.batch_size_evs, replace=self.batch_size_evs>len(eidx))
            eidx_end = eidx + 1 # take direct successor event by default
            pols = self.events[fidx][eidx+1, 3].unsqueeze(0)

        event1 = self.events[fidx][eidx, :].detach()
        event1_tss_ns = event1[:, 2]

        event2 = self.events[fidx][eidx_end, :].detach()
        event2_tss_ns = event2[:, 2]
                              
        if self.precompute_evs_poses:
            raise ValueError('precomputed_evs_poses cannot be set to True.')
        
        poses = self.poses[index].to(self.device) # [B, 4, 4] # * notice that self.poses are computed based on raw poses_hf.
        error_map = None if self.error_map is None else self.error_map[index]
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map)

        results = {
            'batch_size': len(index),
            'H': self.H,
            'W': self.W,
            'evs1': event1,
            'evs1_tss_ns': event1_tss_ns,
            'evs2': event2,
            'evs2_tss_ns': event2_tss_ns,
            'pols': pols,
            # 'rays_o': rays['rays_o'], # * not needed since we don't plan to train with images
            # 'rays_d': rays['rays_d']
        }

        if self.negative_event_sampling:
            raise NotImplementedError('negative_event_sampling is not supported yet.')

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images

        if error_map is not None:
            # results['index'] = index
            # results['inds_coarse'] = rays['inds_coarse']
            raise NotImplementedError('error_map is not supported yet.')
        
        return results
    
    def get_gt_poses_hf_dict(self):
        raw_poses_hf_dict = __class__.decompose_raw_poses_hf(self.poses_hf)
        raw_poses_hf = raw_poses_hf_dict['raw_poses_hf']
        raw_tss_poses_hf_ns = raw_poses_hf_dict['raw_tss_poses_hf_ns']
        poses_hf_dict = {
            'poses_hf': raw_poses_hf,
            'tss_poses_hf_ns': raw_tss_poses_hf_ns,
            'raw_poses_hf': raw_poses_hf,
            'raw_tss_poses_hf_ns': raw_tss_poses_hf_ns,
            'comment': 'ground truth poses_hf.'
        }
        return poses_hf_dict
    
    def get_final_poses_hf_dict(self, verbose=True):
        if verbose:
            print('* final poses_hf for training: #poses_hf =', self.poses_hf_dict_final['tss_poses_hf_ns'].shape[0])
        return self.poses_hf_dict_final
    
    def _compute_noised_poses_hf(self, noise_dev=1e-3):
        raw_poses_hf_dict = __class__.decompose_raw_poses_hf(self.poses_hf)
        raw_tss_poses_hf_ns, raw_poses_hf = raw_poses_hf_dict['raw_tss_poses_hf_ns'], raw_poses_hf_dict['raw_poses_hf']
        se3_noise = torch.randn(len(raw_tss_poses_hf_ns), 6) * noise_dev
        poses_noise = lie.se3_to_SE3(se3_noise)
        noised_poses_hf = pose.compose([poses_noise, raw_poses_hf])
        # * raw_poses_hf is the unchanged original data from the mocap system. poses_hf is the starting point of training.
        noised_poses_hf_dict = {'tss_poses_hf_ns': raw_tss_poses_hf_ns, 'poses_hf': noised_poses_hf,
                                'raw_tss_poses_hf_ns': raw_tss_poses_hf_ns, 'raw_poses_hf': raw_poses_hf,
                                'comment': f'add noise of {noise_dev} to the raw poses_hf.'}

        return noised_poses_hf_dict

    @staticmethod
    def decompose_raw_poses_hf(raw_poses_hf):
        raw_tss_poses_hf_ns = torch.tensor(np.stack([p["ts_ns"] for p in raw_poses_hf]), dtype=torch.float32, requires_grad=False).detach()
        raw_poses_hf = torch.tensor(np.stack([p["pose_c2w"][:3, :] for p in raw_poses_hf]), dtype=torch.float32, requires_grad=False).detach()
        raw_poses_hf_dict = {'raw_tss_poses_hf_ns': raw_tss_poses_hf_ns, 'raw_poses_hf': raw_poses_hf}
        return raw_poses_hf_dict