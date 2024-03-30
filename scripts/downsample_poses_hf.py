import pickle
import torch

poses_hf_load_path = 'output/poses_hf_cache/noised_poses_hf_1e-2.pickle'
poses_hf_save_path = 'output/poses_hf_cache/ds_poses_hf_30hz.pickle'

if __name__ == '__main__':
    with open(poses_hf_load_path, 'rb') as fin:
        poses_hf_dict = pickle.load(fin)

    tss_poses_hf_ns = poses_hf_dict['tss_poses_hf_ns'].detach().cpu()
    # * raw was chosen because we want to downsample fromn the gt poses_hf
    # * if we want to downsample the altered (e.g. noised) poses_hf, we change it to poses_hf
    poses_hf = poses_hf_dict['raw_poses_hf'].detach().cpu() 

    tss_diff = torch.diff(tss_poses_hf_ns)
    # print('diff of tss_poses_hf_ns:\n', tss_diff)

    acc = 0
    threshold = 33333333.3 #ns =  1/30s 
    ds_idxs = [0] # always insert the first point
    for idx, diff in enumerate(tss_diff):
        if acc > threshold:
            acc = 0
            ds_idxs.append(idx+1)
        else:
            acc += diff
    if ds_idxs[-1] != len(tss_poses_hf_ns)-1:
        ds_idxs.append(len(tss_poses_hf_ns)-1)
    
    print('# downsampled points:', len(ds_idxs))
    # print('selected idxs', ds_idxs)

    ds_idxs = torch.tensor(ds_idxs)

    ds_tss_poses_hf_ns = tss_poses_hf_ns[ds_idxs]
    ds_poses_hf = poses_hf[ds_idxs, :, :]

    # # uncomment them for debugging
    # print('shape of tss_poses_hf_ns and downsampled tss_poses_hf_ns:', tss_poses_hf_ns.shape ,ds_tss_poses_hf_ns.shape)
    # print('diff of downsampled tss_poses_hf_ns:\n', torch.diff(ds_tss_poses_hf_ns))

    ds_poses_hf_dict = {
        'poses_hf': ds_poses_hf,
        'tss_poses_hf_ns': ds_tss_poses_hf_ns,
        'raw_poses_hf': poses_hf,
        'raw_tss_poses_hf_ns': tss_poses_hf_ns,
        'comment': 'downsample the raw poses_hf to 30Hz.'
    }

    with open(poses_hf_save_path, 'wb') as fout:
        pickle.dump(ds_poses_hf_dict, fout)