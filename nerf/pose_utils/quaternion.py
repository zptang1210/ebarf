import torch

def fast_normalise(quat):
    """Normalise the object to a unit quaternion using a fast approximation method if appropriate.

    Object is guaranteed to be a quaternion of approximately unit length
    after calling this operation UNLESS the object is equivalent to Quaternion(0)
    """
    return torch.nn.functional.normalize(quat, dim=1)

def slerp(q0_, q1_, amount_):
    q0 = fast_normalise(q0_)
    q1 = fast_normalise(q1_)
    amount = torch.clip(amount_, 0, 1)

    dot = (q0 * q1).sum(axis=-1)
    mask_dot_lt_0 = dot < 0.0
    # q0[mask_dot_lt_0] = q0[mask_dot_lt_0] * -1 # * bug: in-place operation, cannot backward
    # dot[mask_dot_lt_0] = dot[mask_dot_lt_0] * -1
    q0 = torch.where(mask_dot_lt_0.reshape(-1, 1), -q0, q0)
    dot = torch.where(mask_dot_lt_0, -dot, dot)

    qr = torch.zeros_like(q0)

    mask_dot_gt_09995 = dot > 0.9995
    if torch.any(mask_dot_gt_09995):
        qr[mask_dot_gt_09995] = fast_normalise(q0[mask_dot_gt_09995] + amount[mask_dot_gt_09995].reshape(-1, 1) * (q1[mask_dot_gt_09995] - q0[mask_dot_gt_09995]))

    mask_r = ~mask_dot_gt_09995
    if torch.any(mask_r):
        theta_0 = torch.arccos(dot[mask_r])
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * amount[mask_r]
        sin_theta = torch.sin(theta)

        s0 = torch.cos(theta) - dot[mask_r] * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        s0, s1 = s0.reshape(-1, 1), s1.reshape(-1, 1)
        qr[mask_r] = fast_normalise((s0 * q0[mask_r]) + (s1 * q1[mask_r]))

    return qr