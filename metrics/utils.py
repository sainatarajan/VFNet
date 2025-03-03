import torch
from typing import Tuple

def unmask(x: torch.Tensor, x_mask: torch.BoolTensor) -> torch.Tensor:
    """Reshape a masked point cloud tensor to remove masked points.

    Args:
        x: Tensor of shape [B, N, C] where B is batch size, N is max points, C is coordinates.
        x_mask: Boolean mask of shape [B, N], True where points are masked.

    Returns:
        Tensor of shape [B, M, C] where M is the number of unmasked points per batch item.

    Raises:
        AssertionError: If the number of unmasked points varies across the batch.
    """
    bsize, max_points = x.shape[0], x.shape[1]
    n_points = (~x_mask).sum(dim=1)
    assert n_points.eq(n_points[0]).all(), "All batch items must have the same number of unmasked points"
    return x[~x_mask].view(bsize, n_points[0], -1)