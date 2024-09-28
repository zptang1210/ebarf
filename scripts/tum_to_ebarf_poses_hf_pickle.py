import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation

def read_tum_file(file_path):
    poses = []
    timestamps = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 8:  # Ensure we have timestamp + 7 pose elements
                timestamp = float(parts[0])
                pose = list(map(float, parts[1:]))
                timestamps.append(timestamp)
                poses.append(pose)
    return poses, timestamps

def is_valid_rotation_matrix(R, tolerance=1e-6):
    # Check if matrix is orthogonal
    orthogonal = np.allclose(np.dot(R.T, R), np.eye(3), atol=tolerance)
    # Check if determinant is 1
    proper = np.isclose(np.linalg.det(R), 1, atol=tolerance)
    return orthogonal and proper

def pose_to_matrix(pose):
    translation = np.array(pose[:3])
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    
    if not is_valid_rotation_matrix(rotation):
        raise ValueError("Invalid rotation matrix detected")
    
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform[:3, :]  # Return only the 3x4 part

def process_tum_file(file_path):
    poses, timestamps = read_tum_file(file_path)
    
    valid_poses = []
    valid_timestamps = []
    
    for pose, timestamp in zip(poses, timestamps):
        try:
            pose_matrix = pose_to_matrix(pose)
            valid_poses.append(pose_matrix)
            valid_timestamps.append(timestamp)
        except ValueError as e:
            print(f"Skipping invalid pose at timestamp {timestamp}: {e}")
    
    # Convert to torch tensors
    pose_tensor = torch.tensor(valid_poses, dtype=torch.float32)
    timestamp_tensor = torch.tensor(valid_timestamps, dtype=torch.float32) * 1e9 # to us
    
    return pose_tensor, timestamp_tensor

# Example usage
file_path = 'stamped_traj_estimate.tum'
pose_tensor, timestamp_tensor = process_tum_file(file_path)

print(f"Pose tensor shape: {pose_tensor.shape}")
print(f"Timestamp tensor shape: {timestamp_tensor.shape}")
# print(timestamp_tensor)

poses_hf_dict = {
    'poses_hf': pose_tensor,
    'tss_poses_hf_ns': timestamp_tensor,
    'raw_poses_hf': pose_tensor,
    'raw_tss_poses_hf_ns': timestamp_tensor,
    'comment': 'test devo on partial mocapdesk2.'
}

with open('poses_hf_mocapdesk2_devo_partial.pickle', 'wb') as fout:
    pickle.dump(poses_hf_dict, fout)