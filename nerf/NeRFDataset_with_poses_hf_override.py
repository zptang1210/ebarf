import torch
from .NeRFDataset import NeRFDataset
from .pose_utils.pose_interpolator import PoseInterpolator

class NeRFDataset_with_poses_hf_override(NeRFDataset):
    def __init__(self, opt, device, poses_hf_dict_for_override, type='train', downscale=1, n_test=10, select_frames=None, get_rays_evs_on_collate=False):
        super().__init__(opt, device, type=type, downscale=downscale, n_test=n_test, select_frames=select_frames, get_rays_evs_on_collate=get_rays_evs_on_collate)

        # * compute self.poses again with the new poses_hf
        # * notice that poses_hf usually refers to T_mocap_evCam...
        tss = poses_hf_dict_for_override['tss_poses_hf_ns']
        poses_hf = poses_hf_dict_for_override['poses_hf']

        interpolator = PoseInterpolator(tss, poses_hf)
        tss_imgs_ns = self.tss_imgs_us * 1000
        # print('tss', tss)
        # print('tss_img', tss_imgs_ns)
        for i in range(len(tss_imgs_ns)):
            if tss_imgs_ns[i] < tss[0]:
                tss_imgs_ns[i] = tss[0]
            if tss_imgs_ns[i] > tss[-1]:
                tss_imgs_ns[i] = tss[-1]
        with torch.no_grad():
            new_poses = interpolator.interpolate_poses(torch.tensor(tss_imgs_ns, dtype=torch.float32))
        # print('newposes', new_poses, new_poses.shape)
        self.poses = new_poses[self.frame_idxs, :, :]

        # * set self.images to None because we don't have gt for comparison
        # self.images = None

        assert not self.get_rays_evs_on_collate, 'not support get_rays_evs_on_collate yet.'
            
        