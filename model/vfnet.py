import torch
import torch.distributions as td
import torch.nn as nn
import numpy as np

from .decoder import LinearFoldingDecoder
from .encoder import FoldNetEncoderLinear  # Using the refactored encoder name
from pc_utils import ChamferLoss


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for point cloud generation and manipulation.
    
    This model encodes point clouds into a latent distribution and 
    reconstructs them using a decoder network, following the VAE framework.
    """
    
    def __init__(self, args, num_points, global_std=8.18936):
        """
        Initialize the Variational Autoencoder.
        
        Args:
            args: Configuration arguments containing model parameters
            num_points: Number of points in the output point cloud
            global_std: Global standard deviation parameter
        """
        super(VariationalAutoencoder, self).__init__()
        
        # Model configuration
        self.input_dim = 6 if args.point_normals else 3
        self.feat_dims = args.feat_dims
        self.max_epochs = args.num_epochs
        self.warm_up_epochs = int(args.num_epochs / 4)
        self.global_std = global_std
        self.current_epoch = 0
        
        # Network components
        self.encoder = FoldNetEncoderLinear(args)
        self.decoder = LinearFoldingDecoder(args, num_points)
        
        # Loss function
        self.loss = ChamferLoss()
        
        # Distribution placeholders
        self.prior = None
        self.prior_bs = None
        self.q_zGx = None

    def reparameterize(self, feature):
        """
        Apply the reparameterization trick to sample from the latent distribution.
        
        Args:
            feature: Output feature from the encoder containing mean and log variance
            
        Returns:
            Sampled latent vector
        """
        # Split the feature into mean and log variance
        mu, log_var = torch.chunk(feature, 2, dim=-1)
        
        # Create the posterior distribution
        # Add small epsilon for numerical stability
        std = log_var.mul(0.5).exp() + 1e-10
        self.q_zGx = td.normal.Normal(loc=mu, scale=std)
        
        # Sample using the reparameterization trick
        return self.q_zGx.rsample()

    def forward(self, input_pc, sample_grid=False, edge_only=False, jacobian=False):
        """
        Forward pass of the VAE.
        
        Args:
            input_pc: Input point cloud
            sample_grid: Whether to sample from the grid
            edge_only: Whether to output edge points only
            jacobian: Whether to compute the Jacobian
            
        Returns:
            Reconstructed point cloud
        """
        # Ensure correct input dimensions
        if input_pc.shape[2] != self.input_dim and input_pc.shape[1] == self.input_dim:
            input_pc = input_pc.transpose(1, 2)
        
        # Encode input to latent space
        feature, local_features = self.encoder(input_pc)
        
        # Sample from the latent distribution
        latent_codes = self.reparameterize(feature)
        
        # Decode the latent codes
        if jacobian:
            return self.decoder(latent_codes, local_features, eval, edge_only, jacobian)
        else:
            return self.decoder(latent_codes, local_features, eval, edge_only)
        
    def get_latent(self, pc):
        """
        Get the latent representation of a point cloud.
        
        Args:
            pc: Input point cloud
            
        Returns:
            Latent vector representation
        """
        # Ensure correct input dimensions
        if pc.shape[2] != self.input_dim and pc.shape[1] == self.input_dim:
            pc = pc.transpose(1, 2)
        
        # Encode and sample
        feature, _ = self.encoder(pc)
        return self.reparameterize(feature)
    
    def get_grid(self, pc):
        """
        Get the grid representation for a point cloud.
        
        Args:
            pc: Input point cloud
            
        Returns:
            Grid representation
        """
        # Ensure correct input dimensions
        if pc.shape[2] != self.input_dim and pc.shape[1] == self.input_dim:
            pc = pc.transpose(1, 2)
        
        # Encode and sample
        feature, local_features = self.encoder(pc)
        latent_codes = self.reparameterize(feature)
        
        # Repeat latent codes for each point and concatenate with local features
        latent_codes_repeated = latent_codes.repeat(1, local_features.shape[1], 1)
        cat = torch.cat((latent_codes_repeated, local_features), dim=-1)
        
        # Map to grid
        return self.decoder.grid_map(cat)

    def get_parameter(self):
        """
        Get all trainable parameters of the model.
        
        Returns:
            List of parameters
        """
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def get_loss(self, epoch, ground_truth, model_output, std=1):
        """
        Calculate the loss for training the VAE.
        
        Args:
            epoch: Current training epoch
            ground_truth: Ground truth point cloud
            model_output: Output from the model
            std: Standard deviation scaling factor
            
        Returns:
            Dictionary containing loss components
        """
        reconstruction = model_output["reconstruction"]
        bs, n_dim, n_points = ground_truth.shape
        self.loss.current_epoch = epoch
        self.current_epoch = epoch
        
        # Handle different output formats
        if reconstruction.shape[-1] == 3:
            reconstruction = reconstruction.reshape(bs, -1, 3)
        
        # Ensure correct dimensions for ground truth and reconstruction
        if ground_truth.shape[-1] > 6:
            ground_truth = ground_truth.transpose(1, 2)
        if reconstruction.shape[-1] != 3:
            reconstruction = reconstruction.transpose(1, 2)
        
        # Extract vertex coordinates
        ground_truth_vertex = ground_truth[:, :, :3]
        
        # Initialize prior distribution if not already done
        if self.prior is None or self.prior_bs != ground_truth.shape[0]:
            self.prior_bs = ground_truth.shape[0]
            self.prior = td.normal.Normal(
                loc=torch.zeros_like(self.q_zGx.loc, requires_grad=False),
                scale=torch.ones_like(self.q_zGx.scale, requires_grad=False)
            )
        
        # Determine standard deviation for reconstruction distribution
        if "std" not in model_output.keys():
            scaling = 0.0005
            self.std = torch.ones_like(reconstruction) * np.sqrt(scaling)
        else:
            self.std = (model_output["std"].repeat(1, 1, 3).mul(0.5).exp() * 0.0005) + 1e-10
        
        # Create distribution for reconstruction
        if "std" not in model_output.keys():
            p_xGz = td.studentT.StudentT(df=3, loc=reconstruction, scale=self.std)
        else:
            p_xGz = td.normal.Normal(loc=reconstruction, scale=self.std)
        
        # Calculate KL divergence and reconstruction log probability
        kl = td.kl_divergence(self.q_zGx, self.prior).sum(-1).sum(-1)
        recon_error = p_xGz.log_prob(ground_truth_vertex).sum(-1).sum(-1)
        
        # Apply KL annealing
        kl_coeff = min(1.0, epoch / self.warm_up_epochs)
        
        # Evidence Lower BOund (ELBO)
        ELBO = recon_error - kl * kl_coeff
        
        # Chamfer distance loss
        chamfer_loss = self.loss(ground_truth_vertex * std, reconstruction * std)
        
        return {
            "total_loss": -(ELBO).mean(),
            "chamfer": chamfer_loss['chamfer'],
            "elbo": -ELBO.mean(),
            "kl": kl.mean(),
            "kl_coeff": kl_coeff
        }

    def linear_decrease(self):
        """
        Calculate a linear decrease factor based on current epoch.
        
        Returns:
            Decrease factor
        """
        return 1 - 0.99 * (self.current_epoch / self.max_epochs)
