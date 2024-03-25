import numpy as np
import torch
from .utils import get_rays
from .EventNeRFDataset import EventNeRFDataset

class EventBARFDataset(EventNeRFDataset):
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, select_frames=None, cached_data=None):
        super().__init__(opt, device, type=type, downscale=downscale, n_test=n_test, select_frames=select_frames, cached_data=cached_data)
        assert opt.event_only, 'only support event_only mode.'

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
        
        poses = self.poses[index].to(self.device) # [B, 4, 4]
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
    
    def get_poses_hf(self):
        return self.poses_hf