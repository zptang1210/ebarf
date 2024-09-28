import sys
import json
import torch
import pickle
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.camera as pc
import pytransform3d.visualizer as pv

def nearest_rotation_matrix(A):
    U, S, Vt = np.linalg.svd(A)
    return np.dot(U, Vt)

def is_valid_rotation_matrix(R, tolerance=1e-6):
    # Check if matrix is orthogonal
    orthogonal = np.allclose(np.dot(R.T, R), np.eye(3), atol=tolerance)
    assert orthogonal
    # Check if determinant is 1
    proper = np.isclose(np.linalg.det(R), 1, atol=tolerance)
    assert proper
    return orthogonal and proper

if __name__ == '__main__':
    with open('records/tmp/poses_hf_mocapdesk2_devo_partial.pickle', 'rb') as fin:
        pose_dict = pickle.load(fin)


    raw_poses = pose_dict['raw_poses_hf']
    raw_ts = pose_dict['raw_tss_poses_hf_ns']

    ref_poses = pose_dict['poses_hf'] # refined poses by ebarf
    ref_ts = pose_dict['tss_poses_hf_ns']

    fig = pv.figure()

    # for i in range(len(raw_ts)):
    #     print(i)
    #     col_trans_mat = raw_poses[i, :, :].numpy()
    #     print(col_trans_mat)
    #     # col_trans_mat[:3, :3] = nearest_rotation_matrix(col_trans_mat[:3, :3])
    #     print(col_trans_mat)
    #     col_trans_mat = np.concatenate([col_trans_mat, np.array([0, 0, 0, 1]).reshape(1, -1)], axis=0)
    #     fig.plot_transform(A2B=col_trans_mat, s=0.002)
    #     if i+1 < raw_poses.shape[0]:
    #         fig.plot((raw_poses[i, :3, 3], raw_poses[i+1, :3, 3]), (0, 0, 1))

    for i in range(len(ref_ts)):
        col_trans_mat = ref_poses[i, :, :].numpy()
        print(col_trans_mat)
        is_valid_rotation_matrix(col_trans_mat[:3, :3])
        # col_trans_mat[:3, :3] = nearest_rotation_matrix(col_trans_mat[:3, :3])
        col_trans_mat = np.concatenate([col_trans_mat, np.array([0, 0, 0, 1]).reshape(1, -1)], axis=0)
        fig.plot_transform(A2B=col_trans_mat, s=0.01)
        if i+1 < ref_poses.shape[0]:
            fig.plot((ref_poses[i, :3, 3], ref_poses[i+1, :3, 3]), (0, 1, 0))

    fig.show()
