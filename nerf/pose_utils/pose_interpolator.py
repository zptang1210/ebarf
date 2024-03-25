import importlib
import numpy as np
import torch
from pytorch3d import transforms
from nerf.pose_utils.quaternion import slerp
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

class PoseInterpolator:
    def __init__(self, tss_hf, poses_hf, scipy_debug=False):
        '''
        tss_hf: tensor.float32 [N,]
        poses_hf: tensor.float32 [N, 3, 4]
        '''
        self.tss_hf = tss_hf
        self.poses_hf = poses_hf
        self.scipy_debug = scipy_debug

        coeffs = natural_cubic_spline_coeffs(self.tss_hf, self.poses_hf[:, :3, 3])
        self.spline = NaturalCubicSpline(coeffs)

        self.camera_module = importlib.import_module('nerf.pose_utils.camera')

        if self.scipy_debug:
            self.scipy_interpolate_rotation = Slerp(self.tss_hf.detach().cpu().numpy(), R.from_matrix(self.poses_hf[:, :3, :3].detach().cpu().numpy()))
            self.scipy_interpolate_translation = interp1d(x=self.tss_hf.detach().cpu().numpy(),
                                                            y=self.poses_hf[:, :3, 3].detach().cpu().numpy(),
                                                            axis=0, kind='cubic', bounds_error=True)

    def interpolate_rotation(self, tss):
        '''
        tss: tensor.float32 [M,]
        return: rotation matrix tensor.float32 [M, 3, 3]
        '''
        tss = tss.contiguous()
        ip = torch.searchsorted(self.tss_hf, tss)
        left_idx, right_idx = ip-1, ip
        left_ts = self.tss_hf[left_idx]
        right_ts = self.tss_hf[right_idx]
        left_rot = self.poses_hf[left_idx, :3, :3]
        left_quaternion = transforms.matrix_to_quaternion(left_rot)
        right_rot = self.poses_hf[right_idx, :3, :3]
        right_quaternion = transforms.matrix_to_quaternion(right_rot)

        amount = (tss - left_ts) / (right_ts - left_ts)
        quat = slerp(left_quaternion, right_quaternion, amount)
        res = transforms.quaternion_to_matrix(quat)

        if self.scipy_debug:
            atol = 1e-05
            res_scipy = self.interpolate_rotation_scipy(tss)
            for idx in range(res_scipy.shape[0]):
                if not np.allclose(res[idx, :, :].detach().cpu().numpy(), res_scipy[idx, :, :], atol=atol):
                    error_prompt = "Rotation interpolation doesn't match Scipy's result.\n" +\
                        f'idx={idx} (atol={atol}):\n{res[idx, :, :]}\n' +\
                        f'{res_scipy[idx, :, :]}\n{np.isclose(res[idx, :, :].detach().cpu().numpy(), res_scipy[idx, :, :], atol=atol)}'
                    raise ValueError(error_prompt)

        return res

    def interpolate_rotation_scipy(self, tss):
        '''
        tss: tensor or ndarray float32 [M,]
        return: ndarray [M, 3, 3]
        '''
        if torch.is_tensor(tss):
            tss = tss.detach().cpu().numpy()
        return self.scipy_interpolate_rotation(tss).as_matrix().astype(np.float32, casting='same_kind')
    
    def interpolate_translation(self, tss):
        '''
        return tensor.float32 (M, 3)'''
        tss = tss.contiguous()
        res = self.spline.evaluate(tss)

        if self.scipy_debug:
            atol = 1e-05
            res_scipy = self.interpolate_translation_scipy(tss)
            for idx in range(res_scipy.shape[0]):
                if not np.allclose(res[idx, :].detach().cpu().numpy(), res_scipy[idx, :], atol=atol):
                    error_prompt = "Translation interpolation doesn't match Scipy's result.\n" +\
                        f'idx={idx} (atol={atol}):\n{res[idx, :]}\n' +\
                        f'{res_scipy[idx, :]}\n{np.isclose(res[idx, :].detach().cpu().numpy(), res_scipy[idx, :], atol=atol)}'
                    raise ValueError(error_prompt)                 

        return res
    
    def interpolate_translation_scipy(self, tss):
        '''
        return ndarray (M, 3)
        '''
        if torch.is_tensor(tss):
            tss = tss.detach().cpu().numpy()
        return self.scipy_interpolate_translation(tss).astype(np.float32, casting='same_kind')
    
    def interpolate_poses(self, tss):
        rot = self.interpolate_rotation(tss)
        trans = self.interpolate_translation(tss)
        poses = self.camera_module.pose(rot, trans)
        return poses
