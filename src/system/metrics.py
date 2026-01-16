'''
use metrics from rignet
'''
import torch
from torch import FloatTensor
from typing import Tuple

def J2J(
    joints_a: FloatTensor,
    joints_b: FloatTensor,
    continuous_range: Tuple[float, float]
) -> FloatTensor:
    '''
    joints_a: (J1, 3) joint

    joints_b: (J2, 3) joint
    '''
    dis1 = torch.cdist(joints_a, joints_b)
    loss1, _ = dis1.min(dim=-1)
    dis2 = torch.cdist(joints_b, joints_a)
    loss2, _ = dis2.min(dim=-1)
    s = continuous_range[1] - continuous_range[0]
    return (loss1.mean() + loss2.mean()) / 2 / s

def sample_bones(bones: FloatTensor, num: int=100) -> FloatTensor:
    sample_coord = []
    for i in range(num):
        l = i / num
        sample_coord.append(bones[:, :3] * l + bones[:, 3:] * (1-l))
    return torch.cat(sample_coord)

def points_to_segments_distance(
    coord: FloatTensor,
    bones_head: FloatTensor,
    bones_tail: FloatTensor,
    keepdims: bool,
    eps: float=1e-6
) -> FloatTensor:
    # (J, 3)
    offset = bones_tail - bones_head
    inv = (1./(offset * offset + eps).sum(dim=-1)).unsqueeze(0)
    # head
    g0 = bones_tail.unsqueeze(0) - coord.unsqueeze(1)
    c0 = (g0 * offset.unsqueeze(0)).sum(dim=-1) * inv
    # tail
    g1 = coord.unsqueeze(1) - bones_head.unsqueeze(0)
    c1 = (g1 * offset.unsqueeze(0)).sum(dim=-1) * inv
    # (N, J)
    scale0 = (c0.clamp(min=0., max=1.) + eps) / (c0.clamp(min=0., max=1.) + c1.clamp(min=0., max=1.) + eps * 2)
    scale1 = -scale0 + 1
    # (N, J, 3)
    nearest = scale0.unsqueeze(2) * bones_head.unsqueeze(0) + scale1.unsqueeze(2) * bones_tail.unsqueeze(0)
    # (N, J)
    dis = (coord.unsqueeze(1) - nearest).norm(dim=-1)
    dis, _ = dis.min(dim=-1)
    if keepdims:
        return dis
    return dis.mean()

def J2B(
    joints_a: FloatTensor,
    joints_b: FloatTensor,
    bones_a: FloatTensor,
    bones_b: FloatTensor,
    continuous_range: Tuple[float, float]
) -> FloatTensor:
    '''
    joints_a: (J1, 3) joint
    
    joints_b: (J2, 3) joint

    bones_a: (J1, 6) (position parent, position)

    bones_b: (J2, 6) (position parent, position)
    '''
    s = continuous_range[1] - continuous_range[0]
    
    def one_way_chamfer_dist(joints_a: FloatTensor, joints_b: FloatTensor) -> FloatTensor:
        # for all points in joints_a, calc distance to the nearest point in joints_b and return average distance
        dist = torch.cdist(joints_a, joints_b)
        min_dist, _ = dist.min(dim=1)
        return min_dist.mean()

    sample_a = sample_bones(bones_a)
    sample_b = sample_bones(bones_b)
    return (one_way_chamfer_dist(joints_a, sample_b) + one_way_chamfer_dist(joints_b, sample_a)) / 2 / s

def B2B(
    bones_a: FloatTensor,
    bones_b: FloatTensor,
    continuous_range: Tuple[float, float]
) -> FloatTensor:
    '''
    bones_a: (J1, 6) (position parent, position)

    bones_b: (J2, 6) (position parent, position)
    '''
    s = continuous_range[1] - continuous_range[0]
    sampled_a = sample_bones(bones=bones_a)
    sampled_b = sample_bones(bones=bones_b)
    return J2J(joints_a=sampled_a, joints_b=sampled_b, continuous_range=continuous_range)


# ==================== Skinning Evaluation Metrics ====================

def skinning_mae(pred: FloatTensor, gt: FloatTensor) -> FloatTensor:
    '''
    Mean Absolute Error between predicted and ground truth skin weights.

    Args:
        pred: Predicted skin weights (N, J)
        gt: Ground truth skin weights (N, J)

    Returns:
        MAE loss scalar
    '''
    return torch.nn.functional.l1_loss(pred, gt)


def skinning_ce(pred: FloatTensor, gt: FloatTensor, eps: float = 1e-8) -> FloatTensor:
    '''
    Cross Entropy loss between predicted and ground truth skin weights.
    CE = -sum(gt * log(pred))

    Args:
        pred: Predicted skin weights (N, J)
        gt: Ground truth skin weights (N, J)
        eps: Small value to avoid log(0)

    Returns:
        Cross entropy loss scalar
    '''
    pred_safe = pred + eps
    gt_safe = gt + eps
    return -(gt_safe * torch.log(pred_safe)).sum(dim=-1).mean()


def skinning_cosine_similarity(pred: FloatTensor, gt: FloatTensor, eps: float = 1e-6) -> float:
    '''
    Cosine similarity between flattened prediction and ground truth.

    Args:
        pred: Predicted skin weights (N, J)
        gt: Ground truth skin weights (N, J)
        eps: Small value for numerical stability

    Returns:
        Cosine similarity value (float)
    '''
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    return torch.nn.functional.cosine_similarity(
        pred_flat.unsqueeze(0), gt_flat.unsqueeze(0), dim=1, eps=eps
    ).item()


def skinning_precision_recall(
    pred: FloatTensor,
    gt: FloatTensor,
    threshold: float = 1e-3,
    relative_threshold: float = 0.15
) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
    '''
    Compute precision, recall, and MAE on filtered (pruned) weights.

    Filters out small weights (< threshold) and weights below relative_threshold
    of the max weight per vertex.

    Args:
        pred: Predicted skin weights (N, J)
        gt: Ground truth skin weights (N, J)
        threshold: Absolute threshold for filtering small weights
        relative_threshold: Relative threshold (fraction of max per vertex)

    Returns:
        Tuple of (precision, recall, mae_filtered)
    '''
    pred_filtered = pred.clone()
    gt_filtered = gt.clone()

    # Filter small weights
    pred_filtered[pred_filtered < threshold] = 0.0
    gt_filtered[gt_filtered < threshold] = 0.0

    # Filter weights below relative_threshold of max per vertex
    pred_max = torch.max(pred_filtered, dim=-1, keepdim=True).values
    gt_max = torch.max(gt_filtered, dim=-1, keepdim=True).values
    pred_filtered[pred_filtered < pred_max * relative_threshold] = 0.0
    gt_filtered[gt_filtered < gt_max * relative_threshold] = 0.0

    # Renormalize
    pred_sum = pred_filtered.sum(dim=-1, keepdim=True)
    gt_sum = gt_filtered.sum(dim=-1, keepdim=True)
    pred_filtered = pred_filtered / (pred_sum + 1e-10)
    gt_filtered = gt_filtered / (gt_sum + 1e-10)

    # Only consider valid rows (sum close to 1)
    valid_rows = torch.abs(gt_sum.squeeze(-1) - 1) < 1e-2
    if valid_rows.sum() == 0:
        device = pred.device
        return (
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device)
        )

    pred_valid = pred_filtered[valid_rows]
    gt_valid = gt_filtered[valid_rows]

    # Compute precision and recall
    pred_nonzero = pred_valid > 0
    gt_nonzero = gt_valid > 0
    true_positive = (pred_nonzero & gt_nonzero).float().sum()

    precision = true_positive / (pred_nonzero.float().sum() + 1e-10)
    recall = true_positive / (gt_nonzero.float().sum() + 1e-10)
    mae_filtered = torch.abs(pred_valid - gt_valid).sum() / (pred_valid.shape[0] + 1e-10)

    return precision, recall, mae_filtered