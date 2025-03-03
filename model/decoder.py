import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import ResidualLinearLayer, local_cov, knn

class DecoderBaseBlock(nn.Module):
    """
    Base decoder block that handles initial grid creation for folding operations.
    
    This class sets up the foundation for generating point clouds from latent representations
    by creating reference grids (plane, sphere) to fold onto.
    """
    def __init__(self, args, num_points):
        super(DecoderBaseBlock, self).__init__()
        # Calculate grid dimensions based on points needed
        self.num_points_sqrt = int(np.sqrt(num_points))
        self.total_points = self.num_points_sqrt * self.num_points_sqrt
        
        # Configuration settings
        self.shape_type = args.fold_orig_shape  # 'plane' or 'sphere'
        self.feature_dims = args.feat_dims
        self.use_point_encoding = True if args.model == "vae" else args.point_encoding

        # Set up initial grid configurations
        if self.shape_type == "plane":
            # Define bounds for 2D grid: [min, max, points_per_dimension]
            self.meshgrid = [[-1, 1, self.num_points_sqrt], [-1, 1, self.num_points_sqrt]]
        elif self.shape_type == "sphere":
            # Pre-sample points on a sphere
            self.sphere = self.sample_spherical(self.total_points)

        # These will be initialized later
        self.grid = None
        self.std = None

    def sample_spherical(self, num_points, ndim=3):
        """
        Generate uniformly distributed points on a unit sphere.
        
        Args:
            num_points: Number of points to generate
            ndim: Dimensionality (typically 3 for 3D)
            
        Returns:
            Array of points on sphere surface
        """
        # Generate random vectors
        vec = np.random.randn(ndim, num_points)
        # Normalize to unit length to place on sphere
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def build_grid(self, batch_size, random_sampling=False, device=None, edge_only=False):
        """
        Constructs grid points that will be deformed by the network.
        
        Args:
            batch_size: Number of samples in batch
            random_sampling: Whether to use random sampling instead of regular grid
            device: Target device for tensor
            edge_only: Whether to only include points on the grid edges
            
        Returns:
            Grid points tensor of shape [batch_size, 2 or 3, num_points]
        """
        if self.shape_type == 'plane':
            if not random_sampling:
                # Create a regular 2D grid
                x = np.linspace(*self.meshgrid[0])  # Evenly spaced points along x
                y = np.linspace(*self.meshgrid[1])  # Evenly spaced points along y
                points = np.array(list(itertools.product(x, y)))  # All combinations
            else:
                if not edge_only:
                    # Random points within the plane bounds
                    points = np.random.uniform(-1, 1, (self.total_points, 2))
                else:
                    # Only include points along the edges of the grid
                    edge_points = []
                    x = np.linspace(*self.meshgrid[0])
                    y = np.linspace(*self.meshgrid[1])
                    
                    # Add points along all four edges
                    for i in range(self.meshgrid[0][-1]):
                        edge_points.append([x[i], y.min()])  # Bottom edge
                        edge_points.append([x[i], y.max()])  # Top edge
                    for i in range(self.meshgrid[1][-1]):
                        edge_points.append([x.min(), y[i]])  # Left edge
                        edge_points.append([x.max(), y[i]])  # Right edge
                    points = np.array(edge_points)

        elif self.shape_type == 'sphere':
            # Use pre-generated or new sphere points
            points = self.sphere if not random_sampling else self.sample_spherical(self.total_points)
            
        elif self.shape_type == 'gaussian':
            # Use pre-generated gaussian distribution (not implemented in this snippet)
            points = self.gaussian
            
        # Replicate the same grid for each sample in the batch
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points).float().to(device)
        return points


class LinearFoldingDecoder(DecoderBaseBlock):
    """
    Decoder that uses linear folding operations to transform a 2D/3D grid into a 3D point cloud.
    
    This decoder can operate in two modes:
    1. Grid-based: Starting from a regular grid and folding twice
    2. Point-encoding: Starting from input points and adapting the grid accordingly
    """
    def __init__(self, args, num_points):
        super(LinearFoldingDecoder, self).__init__(args, num_points)
        
        # First folding network: latent+grid → initial 3D positions
        input_dim = self.feature_dims + 2
        if self.shape_type != 'plane':
            input_dim += 1  # Add one more dimension for sphere
            
        self.folding1 = nn.Sequential(
            nn.Linear(input_dim, args.feat_dims),
            nn.ReLU(),
            ResidualLinearLayer(args.feat_dims, args.feat_dims),
            nn.ReLU(),
            ResidualLinearLayer(args.feat_dims, args.feat_dims),
            nn.ReLU(),
            nn.Linear(args.feat_dims, 3),  # Output 3D coordinates
        )
        
        # Second folding network: latent+3D → refined 3D positions
        self.folding2 = nn.Sequential(
            nn.Linear(args.feat_dims + 3, args.feat_dims),
            nn.ReLU(),
            ResidualLinearLayer(args.feat_dims, args.feat_dims),
            nn.ReLU(),
            ResidualLinearLayer(args.feat_dims, args.feat_dims),
            nn.ReLU(),
            nn.Linear(args.feat_dims, 3),  # Output 3D coordinates
        )

        # Grid mapping network for point-encoding mode
        if self.use_point_encoding:
            # Calculate input dimensions based on point features
            self.original_dims = 3 + 3 * args.point_normals
            
            # Network to predict grid coordinates from input points
            self.grid_map = nn.Sequential(
                nn.Linear(args.feat_dims + 128, int(args.feat_dims / 2)),
                nn.ReLU(),
                nn.Linear(int(args.feat_dims / 2), int(args.feat_dims / 4)),
                nn.ReLU(),
                nn.Linear(int(args.feat_dims / 4), 2),  # Output 2D grid coordinates
                nn.Tanh()  # Scale to [-1, 1]
            )

    def init_std(self, device):
        """
        Initialize the standard deviation prediction network.
        
        Args:
            device: Target device for the network
        """
        self.std = VarianceNetwork(512, device)

    def decode(self, x, grid):
        """
        Apply folding operations to transform grid points using latent features.
        
        Args:
            x: Latent features [batch_size, num_points, feat_dims]
            grid: Grid coordinates [batch_size, num_points, 2 or 3]
            
        Returns:
            Dictionary with reconstruction results and optionally standard deviation
        """
        # First folding operation
        features_with_grid = torch.cat((x, grid), dim=-1)
        folded_points1 = self.folding1(features_with_grid)
        
        # Second folding operation
        features_with_points = torch.cat((x, folded_points1), dim=-1)
        folded_points2 = self.folding2(features_with_points)
        
        # Prepare the result
        result = {"reconstruction": folded_points2.transpose(1, 2)}
        
        # Add standard deviation if available
        if self.std is not None:
            std = self.std(x, folded_points2)
            result["std"] = std
            
        return result

    def forward(self, x, points_orig=None, sample_grid=False, edge_only=False):
        """
        Forward pass through the decoder.
        
        Args:
            x: Latent code [batch_size, 1, feat_dims]
            points_orig: Original points for point encoding mode
            sample_grid: Whether to use random sampling for grid
            edge_only: Whether to only include grid edge points
            
        Returns:
            Dictionary with reconstruction results
        """
        if self.use_point_encoding:
            # Point encoding mode
            assert points_orig is not None, "Original points required for point encoding mode"
            
            # Repeat latent code for each point
            x = x.repeat(1, points_orig.shape[1], 1)  # [batch_size, num_points, feat_dims]
            
            # Concatenate with original point features
            combined = torch.cat((x, points_orig), dim=-1)
            
            # Predict grid coordinates from points and features
            self.grid = self.grid_map(combined)  # [batch_size, num_points, 2]
            
            # Apply folding operations
            return self.decode(x, self.grid)
        else:
            # Regular grid-based mode
            if edge_only:
                num_points = int(self.meshgrid[0][-1]) * 2 + int(self.meshgrid[1][-1]) * 2
            else:
                num_points = self.total_points
                
            # Repeat latent code for each grid point
            x = x.repeat(1, num_points, 1)  # [batch_size, num_points, feat_dims]
            
            # Create or reuse grid
            if self.grid is None or x.shape[0] != self.grid.shape[0]:
                self.grid = self.build_grid(x.shape[0], sample_grid, x.device, edge_only)
                if self.grid.shape[1] < 4:  # Ensure correct shape
                    self.grid = self.grid.transpose(1, 2)  # [batch_size, 2 or 3, num_points]
            
            # Apply folding operations
            output = self.decode(x, self.grid)
            
            # Clear grid if randomly sampled (not reused)
            if sample_grid:
                self.grid = None
                
            return output


class VarianceNetwork(nn.Module):
    """
    Network that predicts point-wise uncertainty in the form of standard deviation.
    
    Uses local geometric features and a three-stage refinement process to estimate
    the uncertainty at each predicted point.
    """
    def __init__(self, feat_dims, device):
        super(VarianceNetwork, self).__init__()
        self.feat_dims = feat_dims
        self.device = device
        
        # First stage: process latent features + point features + local covariance
        self.std_stage1 = nn.Sequential(
            nn.Linear(self.feat_dims + 3 + 9, self.feat_dims),  # +3 for point, +9 for covariance
            nn.ReLU(),
            ResidualLinearLayer(self.feat_dims, self.feat_dims),
            nn.ReLU(),
            nn.Linear(self.feat_dims, 1),  # Output initial std estimate
        ).to(device)
        
        # Second stage: refine using features and initial estimate
        self.std_stage2 = nn.Sequential(
            nn.Linear(self.feat_dims + 1, self.feat_dims),  # +1 for previous std
            nn.ReLU(),
            ResidualLinearLayer(self.feat_dims, self.feat_dims),
            nn.ReLU(),
            nn.Linear(self.feat_dims, 1),  # Output refined std estimate
        ).to(device)
        
        # Third stage: final refinement 
        self.std_stage3 = nn.Sequential(
            nn.Linear(self.feat_dims + 1, self.feat_dims),  # +1 for previous std
            nn.ReLU(),
            ResidualLinearLayer(self.feat_dims, self.feat_dims),
            nn.ReLU(),
            nn.Linear(self.feat_dims, 1),  # Output final std estimate
        ).to(device)

    def forward(self, latent, reconstructed_points):
        """
        Predicts point-wise standard deviation based on local geometry.
        
        Args:
            latent: Latent features [batch_size, num_points, feat_dims]
            reconstructed_points: Predicted 3D points [batch_size, 3, num_points]
            
        Returns:
            Point-wise standard deviation [batch_size, num_points, 1]
        """
        # Ensure correct shape for computing local covariance
        points_transposed = reconstructed_points.transpose(1, 2)
        
        # Find nearest neighbors for each point
        neighbor_indices = knn(points_transposed, k=16)
        
        # Compute local covariance features
        local_features = local_cov(points_transposed, neighbor_indices)  # [batch_size, 12, num_points]
        
        # Transpose back to original orientation
        points = points_transposed.transpose(1, 2)
        
        # Stage 1: Initial prediction using all features
        combined_features = torch.cat((latent, local_features), dim=-1)
        std_initial = self.std_stage1(combined_features)
        
        # Stage 2: Refinement with initial prediction
        combined_features2 = torch.cat((latent, std_initial), dim=-1)
        std_refined = std_initial + self.std_stage2(combined_features2)
        
        # Stage 3: Final refinement
        combined_features3 = torch.cat((latent, std_refined), dim=-1)
        std_final = std_refined + self.std_stage3(combined_features3)
        
        return std_final
