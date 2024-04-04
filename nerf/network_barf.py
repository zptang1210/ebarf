import numpy as np
import torch
from .pose_utils.camera import lie, pose
from .pose_utils.pose_interpolator import PoseInterpolator
from .network import NeRFNetwork
from nerf.utils import get_event_rays

class BARFNetwork(torch.nn.Module):
    def __init__(self,
                 poses_hf_dict, # a dict with keys ts_ns and pose_c2w
                 intrinsics_evs,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 disable_view_direction=False,
                 out_dim_color=3,
                 **kwargs
                 ):
        super().__init__()
        self.nerf = NeRFNetwork(encoding=encoding,
                         encoding_dir=encoding_dir,
                         encoding_bg=encoding_bg,
                         num_layers=num_layers,
                         hidden_dim=hidden_dim,
                         geo_feat_dim=geo_feat_dim,
                         num_layers_color=num_layers_color,
                         hidden_dim_color=hidden_dim_color,
                         num_layers_bg=num_layers_bg,
                         hidden_dim_bg=hidden_dim_bg,
                         bound=bound,
                         disable_view_direction=disable_view_direction,
                         out_dim_color=out_dim_color,
                         **kwargs
                         )

        tss_poses_hf_ns, poses_hf= poses_hf_dict['tss_poses_hf_ns'], poses_hf_dict['poses_hf']
        raw_tss_poses_hf_ns, raw_poses_hf = poses_hf_dict['raw_tss_poses_hf_ns'], poses_hf_dict['raw_poses_hf']
        self.register_buffer('tss_poses_hf_ns', tss_poses_hf_ns)
        self.register_buffer('poses_hf', poses_hf)
        self.raw_poses_hf = raw_poses_hf
        self.raw_tss_poses_hf_ns = raw_tss_poses_hf_ns

        self.se3_refine = torch.nn.Embedding(len(self.tss_poses_hf_ns), 6)
        torch.nn.init.zeros_(self.se3_refine.weight)

        if isinstance(intrinsics_evs, np.ndarray): intrinsics_evs = torch.from_numpy(intrinsics_evs)
        self.register_buffer('intrinsics_evs', intrinsics_evs)
        self.out_dim_color = out_dim_color
    
    def compute_refined_poses_hf(self):
        poses_displacement = lie.se3_to_SE3(self.se3_refine.weight)
        poses_hf_ref = pose.compose([poses_displacement, self.poses_hf])
        return poses_hf_ref

    @staticmethod
    def get_optimizer_and_scheduler(lr, lr_pose, iters):
        optimizer = lambda model: torch.optim.Adam(model.nerf.get_params(lr), betas=(0.9, 0.99), eps=1e-15)
        scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / iters, 1))

        # todo try various methods
        optimizer_pose = lambda model: torch.optim.Adam(model.se3_refine.parameters(), lr=lr_pose)
        scheduler_pose = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / iters, 1))

        return optimizer, scheduler, optimizer_pose, scheduler_pose


    def forward(self, data):
        poses_hf_ref = self.compute_refined_poses_hf()
        interpolator = PoseInterpolator(self.tss_poses_hf_ns, poses_hf_ref)

        evs1_tss_ns, evs2_tss_ns = data['evs1_tss_ns'], data['evs2_tss_ns']
        poses1 = interpolator.interpolate_poses(evs1_tss_ns)
        poses1 = poses1.unsqueeze(0) # add batch dim
        poses2 = interpolator.interpolate_poses(evs2_tss_ns)
        poses2 = poses2.unsqueeze(0)

        xs = data['evs1'][:, 0].unsqueeze(0)
        ys = data['evs1'][:, 1].unsqueeze(0)

        rays_evs = get_event_rays(xs, ys, poses1, poses2, self.intrinsics_evs)
        rays_evs_o1 = rays_evs["rays_evs_o1"]
        rays_evs_o2 = rays_evs["rays_evs_o2"]
        rays_evs_d1 = rays_evs["rays_evs_d1"]
        rays_evs_d2 = rays_evs["rays_evs_d2"]

        B = data['batch_size']
        bg_color_evs = torch.rand((B, 1, self.out_dim_color)).to(rays_evs_o1.device)

        outputs1 = self.nerf.render(rays_evs_o1, rays_evs_d1, staged=False, bg_color=bg_color_evs, perturb=True, out_dim_color=self.out_dim_color) #**vars(self.opt))
        outputs2 = self.nerf.render(rays_evs_o2, rays_evs_d2, staged=False, bg_color=bg_color_evs, perturb=True, out_dim_color=self.out_dim_color) #**vars(self.opt))

        return (outputs1, outputs2)
    