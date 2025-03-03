import os
import json
import numpy as np
import torch
import torch.utils.data as data
import trimesh
from tqdm import tqdm
import random

class VertebraDataset(data.Dataset):
    """
    Dataset class for vertebra meshes, comparable to ShapeNet15kPointClouds.
    Loads vertebra mesh files, samples points uniformly, and returns standardized data.
    
    Args:
        root (str): Root directory containing the vertebra mesh files.
        sample_size (int): Number of points to sample from each mesh.
        split (str): 'train', 'test', or 'val'. Used to split the dataset.
        train_ratio (float): Proportion of data to use for training.
        test_ratio (float): Proportion of data to use for testing.
        val_ratio (float): Proportion of data to use for validation.
        normalize (bool): Whether to normalize the point clouds.
        standardize_per_shape (bool): Whether to standardize each point cloud by its own mean and std.
        random_subsample (bool): Whether to randomly subsample points during __getitem__.
        seed (int): Random seed for reproducibility.
        
    Attributes:
        root (str): Root directory of the dataset.
        sample_size (int): Number of points to sample.
        split (str): Dataset split ('train', 'test', or 'val').
        normalize (bool): Whether to normalize the point clouds.
        standardize_per_shape (bool): Whether to standardize each point cloud.
        random_subsample (bool): Whether to randomly subsample points.
        all_mesh_files (list): List of all mesh file paths.
        all_points (np.ndarray): All sampled point clouds, shape (N, sample_size, 3).
        all_points_mean (np.ndarray): Mean of all points.
        all_points_std (np.ndarray): Standard deviation of all points.
        global_pc_std (np.ndarray): Global point cloud std (for compatibility).
        k (int): Alias for sample_size.
    """
    
    def __init__(self, root, sample_size=2048, split='train', train_ratio=0.7, 
                 test_ratio=0.15, val_ratio=0.15, normalize=True, 
                 standardize_per_shape=False, random_subsample=True, seed=42):
        self.root = root
        self.sample_size = sample_size
        self.split = split
        assert self.split in ['train', 'test', 'val'], "Invalid split. Must be 'train', 'test', or 'val'."
        self.normalize = normalize
        self.standardize_per_shape = standardize_per_shape
        self.random_subsample = random_subsample
        self.k = sample_size  # For compatibility with other datasets
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Find all mesh files in the root directory
        print(f"Loading mesh files from {root}")
        self.all_mesh_files = []
        for file in os.listdir(root):
            if file.endswith(('.obj', '.ply', '.stl')):
                self.all_mesh_files.append(os.path.join(root, file))
        
        if not self.all_mesh_files:
            raise ValueError(f"No mesh files found in {root}")
        
        print(f"Found {len(self.all_mesh_files)} mesh files")
        
        # Split the dataset
        indices = list(range(len(self.all_mesh_files)))
        random.shuffle(indices)
        
        train_size = int(len(indices) * train_ratio)
        test_size = int(len(indices) * test_ratio)
        
        if split == 'train':
            self.file_indices = indices[:train_size]
        elif split == 'test':
            self.file_indices = indices[train_size:train_size + test_size]
        else:  # val
            self.file_indices = indices[train_size + test_size:]
        
        # Sample points from all meshes in this split
        print(f"Sampling {sample_size} points from each mesh for {split} split")
        self.all_points = []
        self.all_ids = []
        
        for i, idx in enumerate(tqdm(self.file_indices)):
            mesh_file = self.all_mesh_files[idx]
            try:
                # Load mesh and sample points
                mesh = trimesh.load(mesh_file)
                points = self._sample_points_from_mesh(mesh, self.sample_size * 3)  # Oversample and we'll use subsets
                
                # Store points
                self.all_points.append(points)
                
                # Create a unique ID from the filename
                file_id = os.path.splitext(os.path.basename(mesh_file))[0]
                self.all_ids.append(file_id)
            except Exception as e:
                print(f"Error processing {mesh_file}: {e}")
        
        # Convert to numpy array
        self.all_points = np.array(self.all_points)
        
        # Calculate normalization statistics
        if normalize:
            self.all_points_mean = self.all_points.reshape(-1, 3).mean(axis=0).reshape(1, 1, 3)
            self.all_points_std = self.all_points.reshape(-1, 3).std(axis=0).reshape(1, 1, 3)
            
            # Normalize all points
            self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        else:
            self.all_points_mean = np.zeros((1, 1, 3))
            self.all_points_std = np.ones((1, 1, 3))
        
        # For compatibility with Teeth_Dataset
        self.global_pc_std = self.all_points_std
        
        print(f"Processed {len(self.all_points)} meshes for {split} split")
    
    def _sample_points_from_mesh(self, mesh, num_points):
        """
        Sample points uniformly from a mesh surface.
        
        Args:
            mesh (trimesh.Trimesh): The input mesh.
            num_points (int): Number of points to sample.
            
        Returns:
            np.ndarray: Sampled points of shape (num_points, 3).
        """
        # Check if mesh has faces
        if len(mesh.faces) == 0:
            # If no faces, sample from vertices with replacement
            indices = np.random.choice(len(mesh.vertices), num_points, replace=True)
            points = mesh.vertices[indices]
        else:
            # Sample points from mesh surface
            points, _ = trimesh.sample.sample_surface(mesh, num_points)
        
        return points
    
    def get_pc_stats(self, idx):
        """Returns the mean and standard deviation for normalization."""
        if self.standardize_per_shape:
            # Use per-shape statistics
            points = self.all_points[idx]
            m = points.mean(axis=0, keepdims=True)
            s = points.std(axis=0, keepdims=True)
            return m, s
        
        # Use global statistics
        return self.all_points_mean.reshape(1, 3), self.all_points_std.reshape(1, 3)
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.all_points)
    
    def __getitem__(self, idx):
        """
        Retrieves a single sample.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            dict: A dictionary containing:
                - 'ids': Sample index.
                - 'set': The point cloud.
                - 'offset': The mean of the point cloud.
                - 'mean': The normalization mean.
                - 'std': The normalization standard deviation.
                - 'label': The category index (0 for all vertebrae).
                - 'sid': The synset ID (custom for vertebrae).
                - 'mid': The model ID.
        """
        points = self.all_points[idx]
        
        # Randomly subsample if needed
        if self.random_subsample:
            # Randomly select sample_size points
            indices = np.random.choice(points.shape[0], self.sample_size, replace=False)
            out_points = points[indices]
        else:
            # Take the first sample_size points
            out_points = points[:self.sample_size]
        
        # Convert to PyTorch tensor
        out_points = torch.from_numpy(out_points).float()
        
        # Calculate offset (mean of points)
        offset = out_points.mean(0, keepdim=True)
        
        # Standardize if needed
        if self.standardize_per_shape:
            out_points = out_points - offset
        
        # Get normalization statistics
        m, s = self.get_pc_stats(idx)
        m, s = torch.from_numpy(m).float(), torch.from_numpy(s).float()
        
        # Create the return dictionary to match ShapeNet15kPointClouds
        return {
            'ids': idx,
            'set': out_points,
            'offset': offset,
            'mean': m,
            'std': s,
            'label': 0,  # All vertebrae are the same class (0)
            'sid': 'vertebra',  # Custom category
            'mid': self.all_ids[idx]  # Model ID
        }

def collate_fn(batch):
    """
    Collation function for DataLoader. Combines samples into a batch.
    
    Args:
        batch (list): List of dictionaries, each from __getitem__.
        
    Returns:
        dict: A dictionary representing the batch, with batched tensors.
    """
    ret = dict()
    # Collect all keys from the first sample
    for k, v in batch[0].items():
        ret.update({k: [b[k] for b in batch]})
    
    # Stack tensors for these keys
    mean = torch.stack(ret['mean'], dim=0)  # [B, 1, 3]
    std = torch.stack(ret['std'], dim=0)  # [B, 1, 3]
    
    s = torch.stack(ret['set'], dim=0)  # [B, N, 3]  (N is sample_size)
    offset = torch.stack(ret['offset'], dim=0)  # Stack the offset
    mask = torch.zeros(s.size(0), s.size(1)).bool()  # [B, N]  (dummy mask)
    cardinality = torch.ones(s.size(0)) * s.size(1)  # [B,] (number of points)
    
    # Update the dictionary with batched tensors
    ret.update({
        'pc': s,
        'offset': offset,
        'set_mask': mask,
        'cardinality': cardinality,
        'mean': mean,
        'std': std
    })
    return ret


def build_vertebra_dataloaders(args):
    """
    Builds training and validation datasets and data loaders for vertebra meshes.
    
    Args:
        args: Configuration arguments with attributes:
            - vertebra_data_dir: Directory containing vertebra meshes
            - batch_size: Batch size for data loading
            - num_workers: Number of workers for data loading
            - sample_size: Number of points to sample from each mesh
            
    Returns:
        tuple: (train_dataset, val_dataset, data_loaders), where data_loaders
            is a dictionary with keys 'Train' and 'Test'.
    """
    train_dataset = VertebraDataset(
        root=args.vertebra_data_dir,
        sample_size=args.sample_size,
        split='train',
        normalize=True,
        standardize_per_shape=False,
        random_subsample=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        # worker_init_fn=lambda worker_id: np.random.seed(torch.initial_seed() % 4294967296)
    )
    
    val_dataset = VertebraDataset(
        root=args.vertebra_data_dir,
        sample_size=args.sample_size,
        split='val',
        normalize=True,
        standardize_per_shape=False,
        random_subsample=False,
        # Use same normalization statistics as training set
        # Note: This is handled automatically in the class
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        # worker_init_fn=lambda worker_id: np.random.seed(torch.initial_seed() % 4294967296)
    )
    
    data_loaders = {"Train": train_loader, "Test": val_loader}
    
    return train_dataset, val_dataset, data_loaders


# Example usage:
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    # D:\PhD\Disc4All\datasets\vertebra\train\L1
    dataset_path = "D:/PhD/Disc4All/datasets/vertebra/train/L1"
    parser.add_argument('--vertebra_data_dir', type=str, default=dataset_path, help='Path to vertebra mesh files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--sample_size', type=int, default=2048, help='Number of points to sample')
    
    args = parser.parse_args()
    
    train_dataset, val_dataset, data_loaders = build_vertebra_dataloaders(args)
    
    # Print sample batch from train loader
    # for batch in data_loaders['Train']:
    #     print("Batch keys:", batch.keys())
    #     print("Point cloud shape:", batch['pc'].shape)
    #     print("Mean shape:", batch['mean'].shape)
    #     print("Std shape:", batch['std'].shape)
    #     break

    # Print sample batch from train loader
    for batch in data_loaders['Train']:
        print("Batch keys:")
        for key in batch.keys():
            print(f"  - {key}: {batch[key].shape if isinstance(batch[key], torch.Tensor) else type(batch[key])}")
        break
