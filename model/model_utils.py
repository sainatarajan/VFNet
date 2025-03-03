import torch
import torch.nn as nn

import torch


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbors for point clouds.

    Args:
        x: Input point cloud tensor of shape (batch_size, num_dims, num_points)
        k: Number of nearest neighbors to find

    Returns:
        idx: Flattened indices of nearest neighbors
    """
    batch_size, num_dims, num_points = x.shape
    k = min(k, num_points - 1)

    # Ensure correct shape for cdist
    x_transposed = x.transpose(1, 2)  # [batch_size, num_points, num_dims]

    distances = torch.cdist(x_transposed, x_transposed)  # [batch_size, num_points, num_points]

    for i in range(batch_size):
        distances[i].fill_diagonal_(float('inf'))

    idx = distances.topk(k=k, dim=-1, largest=False)[1]  # [batch_size, num_points, k]

    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def local_cov(pts: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Compute local covariance features for point clouds.

    Args:
        pts: Input point cloud tensor of shape (batch_size, num_dims, num_points)
        idx: Indices of nearest neighbors from knn function

    Returns:
        Features tensor of shape (batch_size, num_points, 12)
    """
    batch_size, num_dims, num_points = pts.size()  # Correct unpacking

    x = pts.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    x_flat = x.view(batch_size * num_points, num_dims)  # (batch_size * num_points, num_dims)
    neighbors = x_flat[idx]

    expected_size = batch_size * num_points * 16
    actual_size = idx.size(0)
    if actual_size != expected_size:
        print(f"Warning: Neighbor count mismatch - using available {actual_size} indices")

    k = actual_size // (batch_size * num_points) if batch_size * num_points > 0 else 1
    k = min(k, num_points - 1)
    if k < 2:
        raise ValueError(f"Need at least 2 neighbors, got effective k={k}")
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)

    x0 = neighbors[:, :, 0].unsqueeze(3)
    x1 = neighbors[:, :, 1].unsqueeze(2)
    cov = torch.matmul(x0, x1)

    cov_flat = cov.view(batch_size, num_points, 9)
    x = torch.cat((pts.transpose(2, 1), cov_flat), dim=2)

    return x


def local_maxpool(x, idx):
    """
    Perform local max pooling over k-nearest neighbors.
    
    Args:
        x: Input feature tensor of shape (batch_size, num_points, num_dims)
        idx: Indices of nearest neighbors from knn function
        
    Returns:
        Max-pooled features of shape (batch_size, num_points, num_dims)
    """
    batch_size, num_points, num_dims = x.size()
    
    # Gather features of neighbors
    x = x.reshape(batch_size * num_points, -1)[idx, :]              # (batch_size*num_points*k, num_dims)
    x = x.view(batch_size, num_points, -1, num_dims)                # (batch_size, num_points, k, num_dims)
    
    # Max pooling over neighborhood
    x, _ = torch.max(x, dim=2)                                      # (batch_size, num_points, num_dims)
    
    return x


class ResidualLinearLayer(nn.Module):
    """
    Residual linear layer with a bottleneck structure.
    
    Architecture:
        input -> linear1 -> ReLU -> linear2 -> add input -> output
    
    The bottleneck reduces dimensions to half before expanding back.
    """
    
    def __init__(self, in_channels, out_channels, bias=True):
        """
        Initialize the residual linear layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to use bias in linear layers
        """
        super(ResidualLinearLayer, self).__init__()
        
        bottleneck_channels = int(out_channels * 0.5)
        
        self.linear = nn.Sequential(
            nn.Linear(in_channels, bottleneck_channels, bias=bias),
            nn.ReLU(),
            nn.Linear(bottleneck_channels, out_channels, bias=bias)
        )
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with same shape as input
        """
        return self.linear(x) + x
