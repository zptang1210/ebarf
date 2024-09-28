import sys
import json
import torch
import pickle
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import pytransform3d.visualizer as pv

display_valid_idxs_only = True

def check_number_in_ranges(evs_timespan_us, tss_ns):
    start_tss_ns = evs_timespan_us[:, 0] * 1000
    end_tss_ns = evs_timespan_us[:, 1] * 1000

    tss_ns_expanded = tss_ns.unsqueeze(1).repeat(1, start_tss_ns.shape[0]) # shape (tss_ns.shape, start_tss_ns.shape)
    later_than_start_tss = start_tss_ns <= tss_ns_expanded
    before_end_tss = tss_ns_expanded <= end_tss_ns

    valid_idxs = torch.any(torch.logical_and(later_than_start_tss, before_end_tss), dim=1) # shape (tss_ns.shape,)
    return valid_idxs

def nearest_rotation_matrix(A):
    U, S, Vt = np.linalg.svd(A)
    return np.dot(U, Vt)

if __name__ == '__main__':
    with open('records/mocapdesk2_devo_ebarf_tmp.pickle', 'rb') as fin:
        devo_dict = pickle.load(fin)

    with open('records/gt_poses_hf.pickle', 'rb') as fin:
        gt_dict = pickle.load(fin)

    print(devo_dict.keys())

    devo_poses = devo_dict['raw_poses_hf']
    devo_ts = devo_dict['raw_tss_poses_hf_ns']

    ref_poses = devo_dict['poses_hf'] # refined poses by ebarf
    ref_ts = devo_dict['tss_poses_hf_ns']

    gt_poses = gt_dict['poses_hf']
    gt_ts = gt_dict['tss_poses_hf_ns']

    evs_timespan_us = devo_dict['evs_timespan_us'].clone()
    print('raw evs_timespan_us', evs_timespan_us)

    # # cut the devo trajectory involved timespan out of the gt.
    # idx1 = torch.searchsorted(gt_ts, devo_ts[0])
    # idx2 = torch.searchsorted(gt_ts, devo_ts[-1])
    # gt_poses = gt_poses[idx1:idx2, :, :]
    # gt_ts = gt_ts[idx1:idx2]


    fig = pv.figure()

    # gt_valid_idxs = check_number_in_ranges(evs_timespan_us, gt_ts)
    # idx1 = torch.searchsorted(gt_ts, devo_ts[0])
    # idx2 = torch.searchsorted(gt_ts, devo_ts[-1])
    # for i in range(idx1, idx2, 5):
    #     if display_valid_idxs_only and not gt_valid_idxs[i]: continue
    #     gt_trans_mat = gt_poses[i, :, :].numpy()
    #     gt_trans_mat = np.concatenate([gt_trans_mat, np.array([0, 0, 0, 1]).reshape(1, -1)], axis=0)
    #     fig.plot_transform(A2B=gt_trans_mat, s=0.006)
    #     if i+1 < gt_poses.shape[0]:
    #         fig.plot((gt_poses[i, :3, 3], gt_poses[i+1, :3, 3]), (1, 0, 0))

    devo_valid_idxs = check_number_in_ranges(evs_timespan_us, devo_ts)
    for i in range(len(devo_ts)):
        if display_valid_idxs_only and not devo_valid_idxs[i]: continue
        print('-',devo_ts[i])
        col_trans_mat = devo_poses[i, :, :].numpy()
        col_trans_mat[:3, :3] = nearest_rotation_matrix(col_trans_mat[:3, :3])
        col_trans_mat = np.concatenate([col_trans_mat, np.array([0, 0, 0, 1]).reshape(1, -1)], axis=0)
        fig.plot_transform(A2B=col_trans_mat, s=0.002)
        if i+1 < devo_poses.shape[0]:
            fig.plot((devo_poses[i, :3, 3], devo_poses[i+1, :3, 3]), (0, 0, 1))

    # ref_valid_idxs = check_number_in_ranges(evs_timespan_us, ref_ts)
    # for i in range(len(ref_ts)):
    #     if display_valid_idxs_only and not ref_valid_idxs[i]: continue
    #     print('*', ref_ts[i])
    #     col_trans_mat = ref_poses[i, :, :].numpy()
    #     col_trans_mat[:3, :3] = nearest_rotation_matrix(col_trans_mat[:3, :3])
    #     col_trans_mat = np.concatenate([col_trans_mat, np.array([0, 0, 0, 1]).reshape(1, -1)], axis=0)
    #     fig.plot_transform(A2B=col_trans_mat, s=0.01)
    #     if i+1 < ref_poses.shape[0]:
    #         fig.plot((ref_poses[i, :3, 3], ref_poses[i+1, :3, 3]), (0, 1, 0))

    fig.show()
