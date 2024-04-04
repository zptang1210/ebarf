import torch
from .network_barf import BARFNetwork
from .pose_utils.pose_interpolator import PoseInterpolator

class BARFNetwork_with_point_aug(BARFNetwork):
    def __init__(self,
                 barf_net):
        super(BARFNetwork, self).__init__()

        self.nerf = barf_net.nerf
        self.raw_tss_poses_hf_ns, self.raw_poses_hf = barf_net.raw_tss_poses_hf_ns, barf_net.raw_poses_hf
        
        aug_tss_poses_hf_ns, aug_poses_hf = __class__.augmentPoint(barf_net)
        self.register_buffer('tss_poses_hf_ns', aug_tss_poses_hf_ns)
        self.register_buffer('poses_hf', aug_poses_hf)

        self.se3_refine = torch.nn.Embedding(len(self.tss_poses_hf_ns), 6)
        torch.nn.init.zeros_(self.se3_refine.weight)

        self.register_buffer('intrinsics_evs', barf_net.intrinsics_evs)
        self.out_dim_color = barf_net.out_dim_color

    @torch.no_grad()
    def augmentPoint(barf_net):
        tss_poses_hf_ns = barf_net.tss_poses_hf_ns.detach().clone()
        with torch.no_grad():
            poses_hf_ref = barf_net.compute_refined_poses_hf().detach().clone()

        middle_tss = (tss_poses_hf_ns[:-1] + tss_poses_hf_ns[1:]) / 2
        aug_point_num = len(tss_poses_hf_ns) + len(middle_tss)
        aug_tss_poses_hf_ns = torch.zeros(aug_point_num, dtype=tss_poses_hf_ns.dtype)
        aug_tss_poses_hf_ns[::2] = tss_poses_hf_ns
        aug_tss_poses_hf_ns[1::2] = middle_tss

        interpolator = PoseInterpolator(tss_poses_hf_ns, poses_hf_ref)
        middle_poses = interpolator.interpolate_poses(middle_tss)
        aug_poses_hf = torch.zeros((aug_point_num, 3, 4), dtype=poses_hf_ref.dtype)
        aug_poses_hf[::2, :, :] = poses_hf_ref
        aug_poses_hf[1::2, :, :] = middle_poses

        return aug_tss_poses_hf_ns, aug_poses_hf
