import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import knn, local_maxpool, local_cov, ResidualLinearLayer


class FoldNetEncoderLinear(nn.Module):
    """
    FoldNet encoder using linear layers for feature extraction from point clouds.
    
    This encoder processes point cloud data to create a latent representation
    by applying a series of linear transformations and graph-based operations.
    """
    
    def __init__(self, args):
        super(FoldNetEncoderLinear, self).__init__()
        self.k = args.k if args.k is not None else 16
        self.n = 2048  # input point cloud size
        self.point_normals = args.point_normals
        self.in_channels = 15 if args.point_normals else 12
        self.feat_dims = args.feat_dims
        
        # Point-wise feature extraction
        self.mlp1 = self._build_point_mlp()
        
        # Graph layer components
        self.graph_layer1 = self._build_graph_layer1()
        self.graph_layer2 = self._build_graph_layer2()
        
        # Global feature extractor
        self.global_mlp = self._build_global_mlp()
    
    def _build_point_mlp(self):
        """Builds the initial point-wise MLP for feature extraction."""
        return nn.Sequential(
            nn.Linear(self.in_channels, 128),
            nn.ReLU(),
            ResidualLinearLayer(128, 128),
            nn.ReLU(),
            ResidualLinearLayer(128, 128),
            nn.ReLU(),
        )
    
    def _build_graph_layer1(self):
        """Builds the first graph layer component."""
        return nn.Sequential(
            ResidualLinearLayer(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
    
    def _build_graph_layer2(self):
        """Builds the second graph layer component."""
        return nn.Sequential(
            ResidualLinearLayer(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
    
    def _build_global_mlp(self):
        """Builds the global feature extraction MLP."""
        return nn.Sequential(
            nn.Linear(512, 2 * self.feat_dims),
            nn.ReLU(),
            nn.Linear(2 * self.feat_dims, 2 * self.feat_dims)
        )
    
    def graph_layer(self, x, idx):
        """
        Applies graph-based operations to extract hierarchical features.
        
        Args:
            x: Input features of shape (batch_size, num_points, 128)
            idx: K-nearest neighbor indices
            
        Returns:
            tuple: (final_features, intermediate_features)
        """
        # First graph operation
        x = local_maxpool(x, idx)  # (batch_size, num_points, 128)
        intermediate_features = self.graph_layer1(x)  # (batch_size, num_points, 256)
        
        # Second graph operation
        x = local_maxpool(intermediate_features, idx)  # (batch_size, num_points, 256)
        final_features = self.graph_layer2(x)  # (batch_size, num_points, 512)
        
        return final_features, intermediate_features

    def forward(self, pts):
        """
        Forward pass of the encoder.
        
        Args:
            pts: Input point cloud of shape (batch_size, channels, num_points)
            
        Returns:
            tuple: (global_features, local_features)
        """
        # pts = pts.transpose(2, 1)

        if self.point_normals:
            pts_coords = pts[:, :, :3]
            pts_normals = pts[:, :, 3:]
        else:
            pts_coords = pts
            pts_normals = None

        idx = knn(pts_coords.transpose(2, 1), k=self.k)
        covariance_features = local_cov(pts_coords.transpose(2, 1), idx)

        # Prepare input features
        if self.point_normals:
            x = torch.cat([covariance_features, pts_normals], dim=-1)
        else:
            x = covariance_features
        
        # Extract point-wise features
        local_features = self.mlp1(x)  # (batch_size, num_points, 128)
        
        # Extract graph-based features
        graph_features, intermediate_features = self.graph_layer(local_features, idx)
        
        # Global feature pooling
        global_features = torch.max(graph_features, 1, keepdim=True)[0]  # (batch_size, 1, 512)
        
        # Final global feature extraction
        global_features = self.global_mlp(global_features)
        global_features = torch.max(global_features, 1, keepdim=True)[0]
        
        return global_features, local_features
