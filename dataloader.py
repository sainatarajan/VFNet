import json
import os
import platform
import random
from glob import glob
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

import h5py
import numpy as np
import torch
import torch.utils.data as data
from plyfile import PlyData
from torch.utils.data import Dataset
from tqdm import tqdm

import pc_utils  # Assuming this contains point cloud utility functions
from custom_dataset import build_vertebra_dataloaders


def get_few_unnormalized_samples(data_directory, unnormalized_numbers):
    """
    Retrieves information about a few unnormalized point cloud samples.

    Args:
        data_directory: The directory containing the point cloud data.
        unnormalized_numbers: A list of numbers indicating which unnormalized
                              versions to retrieve (e.g., [1, 2, 3]).

    Returns:
        A list of dictionaries, each containing the 'id' and 'ply_file' path
        for a matching unnormalized sample.
    """
    case_directories = os.listdir(data_directory)
    data_info = []
    for directory in case_directories:
        sample_info = {}
        for unn_num in unnormalized_numbers:
            file_path = os.path.join(data_directory, directory, f"unn{unn_num}.ply")
            if os.path.isfile(file_path):
                sample_info["id"] = f"{directory}_{unn_num}"  # Unique identifier
                sample_info["ply_file"] = file_path
                data_info.append(sample_info)
    return data_info


def get_published_data(data_directory):
    """
    Retrieves information about published point cloud samples.  Assumes
    published data has a specific naming convention.

    Args:
        data_directory: The directory containing the point cloud data.

    Returns:
        A list of dictionaries, each containing the 'id' and 'ply_file' path.
    """
    case_files = os.listdir(data_directory)
    # Filter files based on the naming convention "_point_cloud.ply"
    relevant_files = [f for f in case_files if "_point_cloud.ply" in f]

    data_info = []
    for file in relevant_files:
        sample_info = {}
        file_name_parts = file.split("_")
        # Create the ID. Assumes the published data corresponds to "unn3"
        sample_info["id"] = f"{file_name_parts[0]}_3"
        sample_info["ply_file"] = os.path.join(data_directory, file)
        data_info.append(sample_info)
    return data_info


def parse_path(path):
    """
    Parses a file path, handling 'file://' URIs correctly.  This function
    is necessary to account for different operating systems and file path
    conventions, ensuring consistent behavior across platforms.

    Args:
        path: The path string.

    Returns:
        An absolute path string.
    """
    if not path.startswith("file"):
        return path  # Return as is if not a file URI

    parsed_url = urlparse(path)
    # Construct the correct path based on the OS
    host = f"{os.path.sep}{os.path.sep}{parsed_url.netloc}{os.path.sep}"
    return os.path.abspath(os.path.join(host, url2pathname(unquote(parsed_url.path))))


class TeethDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing teeth point cloud data.

    Args:
        unnormalized_numbers: List of unnormalized file numbers to load (e.g., ["3"]).
        folder_path: Path to the data folder.
        is_train: Boolean, whether this is a training dataset.
        args: Configuration arguments.
        global_pc_std: Global point cloud standard deviation (optional, for normalization).
        cluster: Boolean, whether running on a cluster (affects path handling).
        only_worn_teeth: Boolean, whether to include only worn teeth (not used in this version).
        k: Number of points to sample from each point cloud.

    Attributes:
        (See Args above, plus the following)
        data: A list of dictionaries, each holding sample information ('id', 'ply_file').
        global_pc_std: Global point cloud standard deviation.
        point_dist: (If calculated) Point distances.
        scale_dict: (If calculated) Scaling factors.
        translate: A normal distribution for data augmentation (translation).
        random_noise: A normal distribution for adding noise (augmentation).
    """

    def __init__(self, unnormalized_numbers, folder_path, is_train, args,
                 global_pc_std=None, cluster=False, only_worn_teeth=False, k=2048):
        self.is_train = is_train
        self.data_folder_path = folder_path
        self.cluster = cluster
        self.k = k  # Number of points to sample
        self.point_normals = args.point_normals  # Whether to load point normals
        self.global_pc_std = global_pc_std

        if unnormalized_numbers == ["3"]:
            # Load published data if only "unn3" is requested
            self.data = get_published_data(folder_path)
        else:
            # This part was removed to keep the example focused
            raise NotImplementedError("Loading unnormalized numbers other than '3' is not supported.")

        self.data = self.cull_data(self.data, self.k)

        if self.global_pc_std is None:
            print("Calculating global point cloud std for normalization")
            # Calculate normalization statistics if not provided
            self.global_pc_std, self.point_dist, self.scale_dict = pc_utils.calculate_point_cloud_std(
                folder_path, self.data
            )

        if self.is_train:
            # Define data augmentation transformations
            scale = (torch.ones(3) / 2) / self.global_pc_std  # Adjust for normalization
            self.translate = torch.distributions.normal.Normal(
                loc=torch.Tensor([0, 0, 0]), scale=scale
            )
            self.random_noise = torch.distributions.normal.Normal(loc=0, scale=0.0005)

    def __len__(self):
        return len(self.data)

    def cull_data(self, data, min_pc_size):
        """
        Removes samples with fewer points than min_pc_size after edge removal.
        This is important for ensuring consistent input sizes.
        """
        new_data_dict = []
        print("Removing samples not meeting max_pc_n for VAE")
        for data_dict in tqdm(data):
            ply_load = PlyData.read(data_dict['ply_file'])
            point_cloud = torch.from_numpy(
                np.array((ply_load.elements[0]['x'], ply_load.elements[0]['y'], ply_load.elements[0]['z']))
            )

            # Load curvature and edge distance data
            curvature_file = data_dict['ply_file'][:-15] + "curvature_edgedistance.dat"
            with open(curvature_file, "rb") as f:
                curvature_and_edgedistance = np.fromfile(f, np.double)
            curvature_and_edgedistance = curvature_and_edgedistance.reshape(-1, point_cloud.shape[-1]).T

            distance_from_border = curvature_and_edgedistance[:, 1]
            delete_indices = np.where(distance_from_border < 0.5)[0]

            # Keep the sample if enough points remain after edge removal
            if (point_cloud.shape[1] - len(delete_indices)) > min_pc_size:
                data_dict['data'] = point_cloud  # Store the point cloud
                new_data_dict.append(data_dict)

        return new_data_dict

    def pc_normalize(self, pc):
        """Normalizes a single point cloud."""
        return pc / self.global_pc_std

    def pc_unnormalize(self, pc):
        """Unnormalizes a single point cloud."""
        return pc * self.global_pc_std

    def get_specific_sample(self, sample_id):
        """Retrieves a specific sample by its ID."""
        for i, d in enumerate(self.data):
            if d["id"] == sample_id:
                return self.__getitem__(i)
        print("ID not found")
        return None

    def __getitem__(self, item):
        """
        Loads and preprocesses a single point cloud sample.

        Args:
            item: Index of the sample to retrieve.

        Returns:
            A dictionary containing the processed point cloud ('pc') and its ID ('ids').
        """
        ply_path = self.data[item]['ply_file']
        ply_load = PlyData.read(ply_path)

        # Check and potentially preprocess files with face data.  This handles a specific file format issue.
        if 'property list uchar int vertex_indices' in ply_load.header:
            print("Preprocessing files (likely due to face data in PLY).")
            for f in self.data:
                pc_utils.preprocessing(os.path.join(self.data_folder_path, f))  # Assuming a preprocessing function
            ply_load = PlyData.read(ply_path)  # Reload after preprocessing

        # Extract point coordinates
        point_cloud = torch.from_numpy(
            np.array((ply_load.elements[0]['x'], ply_load.elements[0]['y'], ply_load.elements[0]['z']))
        )
        point_cloud = self.pc_normalize(point_cloud).transpose(0, 1)  # Normalize and transpose to (N, 3)

        # Load curvature and edge distance, and remove edge points
        curvature_file = ply_path[:-15] + "curvature_edgedistance.dat"
        with open(curvature_file, "rb") as f:
            curvature_and_edgedistance = np.fromfile(f, np.double)
        curvature_and_edgedistance = curvature_and_edgedistance.reshape(-1, point_cloud.shape[0]).T
        distance_from_border = curvature_and_edgedistance[:, 1]
        delete_indices = np.where(distance_from_border < 0.5)[0]
        keep_indices = np.delete(np.arange(point_cloud.shape[0]), delete_indices)

        # Randomly sample 'k' points, or fewer if not enough points are available after edge removal.
        n_samples = min(self.k, len(keep_indices))
        indices = np.random.choice(keep_indices, n_samples, replace=False)
        point_cloud = point_cloud[indices]

        # Load point normals if requested
        if self.point_normals:
            point_normals = torch.from_numpy(
                np.array((ply_load.elements[0]['nx'], ply_load.elements[0]['ny'], ply_load.elements[0]['nz']))
            )
            point_normals = point_normals.transpose(0, 1)[indices]
        else:
            point_normals = None

        # Data augmentation (only for training)
        if self.is_train and np.random.random() > 0.5:
            # point_cloud = pc_utils.RandomFlip(point_cloud, p=0.33, axis=0) # Flip along x-axis
            point_cloud = pc_utils.RandomScale(point_cloud, scales=[0.8, 1.2])
            # point_cloud = pc_utils.RandomRotate(point_cloud, degrees=180) # Rotate
            point_cloud, point_normals = pc_utils.slight_rotation(point_cloud, point_normals)  # Apply slight rotation
            # point_cloud += self.translate.sample().unsqueeze(0)      # Translate
            # point_cloud += self.random_noise.sample(point_cloud.size()) # Add noise

        # Combine point coordinates and normals (if available)
        if self.point_normals:
            point_cloud = torch.cat((point_cloud, point_normals), dim=1)

        return dict(pc=point_cloud, ids=self.data[item]['id'])
class TeethDatasetUnlimitedPoints(TeethDataset):
    """
    A subclass of Teeth_Dataset that allows for an unlimited number of points
    during loading.  It sets 'k' to a very large value and then culls to
    at least 2048 points.
    """
    def __init__(self, unn, folder_path, is_train, args, global_pc_std=None, cluster=False, only_worn_teeth=False):
        super().__init__(unn, folder_path, is_train, args, global_pc_std, cluster, only_worn_teeth)
        self.k = 1000000  # Effectively unlimited
        self.data = self.cull_data(self.data, 2048) # Ensure at least 2048 points

class BaselineDataset(data.Dataset):
    """
    A dataset class for standard point cloud datasets like ModelNet and ShapeNet.
    This class handles loading data from HDF5 files, applying augmentations,
    and filtering by class.

    Args:
        root (str): Root directory where the dataset files are located.
        dataset_name (str, optional): Name of the dataset ('modelnet40',
            'shapenetcorev2', etc.).  Defaults to 'modelnet40'.
        num_points (int, optional): Number of points to sample from each
            point cloud. Defaults to 2048.
        split (str, optional): Dataset split to load ('train', 'test', 'val',
            'trainval', or 'all'). Defaults to 'train'.
        load_name (bool, optional): If True, load object names from JSON files.
            Defaults to False.
        random_rotate (bool, optional): If True, apply random rotation
            augmentation. Defaults to False.
        random_jitter (bool, optional): If True, apply random jitter
            augmentation. Defaults to True.
        random_translate (bool, optional): If True, apply random translation
            augmentation. Defaults to False.

    Attributes:
        root (str): Root directory of the dataset.
        dataset_name (str): Name of the dataset.
        num_points (int): Number of points per sample.
        split (str): Dataset split.
        load_name (bool): Whether object names are loaded.
        random_rotate (bool): Whether to apply random rotation.
        random_jitter (bool): Whether to apply random jitter.
        random_translate (bool): Whether to apply random translation.
        k (int):  Alias for num_points.
        path_h5py_all (list): List of paths to all HDF5 files.
        path_json_all (list): List of paths to all JSON files (if load_name).
        data (np.ndarray):  Point cloud data of shape (N, num_points, 3).
        label (np.ndarray): Class labels of shape (N,).
        name (list): Object names (if load_name).  Filtered by class.
    """

    def __init__(self, root, dataset_name='modelnet40', num_points=2048, split='train',
                 load_name=False, random_rotate=False, random_jitter=True, random_translate=False):

        # --- Input Validation and Setup ---
        dataset_name = dataset_name.lower()
        assert dataset_name in ['shapenetcorev2', 'shapenetpart', 'modelnet10', 'modelnet40'], \
            "Invalid dataset name. Choose from: 'shapenetcorev2', 'shapenetpart', 'modelnet10', 'modelnet40'"
        assert num_points <= 2048, "num_points must be <= 2048"

        if dataset_name in ['shapenetpart', 'shapenetcorev2']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all'], \
                "Invalid split for ShapeNet. Choose from: 'train', 'test', 'val', 'trainval', 'all'"
        else:
            assert split.lower() in ['train', 'test', 'all'], \
                "Invalid split for ModelNet. Choose from: 'train', 'test', 'all'"

        self.root = os.path.join(root, dataset_name + '*hdf5_2048')  # Path to HDF5 files
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split.lower()
        self.load_name = load_name
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        self.k = num_points  # For consistency with other dataset classes

        self.path_h5py_all = []
        self.path_json_all = []

        # --- Collect File Paths ---
        self._get_paths()  # Populate path_h5py_all and path_json_all

        # --- Load Data ---
        self._load_data()


    def _get_paths(self):
        """Collects paths to HDF5 and JSON files based on the split."""

        if self.split in ['train', 'trainval', 'all']:
            self._append_paths('train')
        if self.dataset_name in ['shapenetpart', 'shapenetcorev2']:
            if self.split in ['val', 'trainval', 'all']:
                self._append_paths('val')
        if self.split in ['test', 'all']:
            self._append_paths('test')

    def _append_paths(self, split_type):
        """Appends paths for a given split type."""
        h5_pattern = os.path.join(self.root, f'*{split_type}*.h5')
        self.path_h5py_all.extend(glob(h5_pattern))
        if self.load_name:
            json_pattern = os.path.join(self.root, f'{split_type}*_id2name.json')
            self.path_json_all.extend(glob(json_pattern))

    def _load_data(self):
        """Loads data from HDF5 files and optionally JSON files."""
        self.path_h5py_all.sort()
        data, label = self._load_h5py(self.path_h5py_all)

        if self.load_name:
            self.path_json_all.sort()
            self.name_all = self._load_json(self.path_json_all)  # Load all names
            # Example: Filter for the "airplane" class.  Modify this for other classes.
            self.class_matches = np.array(self.name_all) == "airplane"
            self.name = np.array(self.name_all)[self.class_matches].tolist()  # Filtered names
            self.data = np.concatenate(data, axis=0)[self.class_matches]
            self.label = np.concatenate(label, axis=0)[self.class_matches]
        else:
            self.data = np.concatenate(data, axis=0)  # Concatenate all data
            self.label = np.concatenate(label, axis=0)  # Concatenate all labels

    def _load_h5py(self, paths):
        """Loads data and labels from a list of HDF5 files.

        Args:
            paths (list): List of paths to HDF5 files.

        Returns:
            tuple: (all_data, all_label) as numpy arrays.
        """
        all_data = []
        all_label = []
        for h5_name in paths:
            with h5py.File(h5_name, 'r') as f:
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
            all_data.append(data)
            all_label.append(label)
        return all_data, all_label

    def _load_json(self, paths):
        """Loads data from a list of JSON files.

        Args:
            paths (list): List of paths to JSON files.

        Returns:
            list: Concatenated list of data from all JSON files.
        """
        all_data = []
        for json_name in paths:
            with open(json_name, 'r') as j:
                data = json.load(j)
                all_data.extend(data)
        return all_data

    def __getitem__(self, item):
        """Retrieves a single sample from the dataset.

        Args:
            item (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the point cloud ('pc'), label ('ids'),
            and optionally the object name ('name').
        """
        point_set = self.data[item][:self.num_points]  # Sample points
        label = self.label[item]

        if self.load_name:
            name = self.name[item]  # Get object name

        # --- Data Augmentation ---
        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        # Convert to PyTorch tensors
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64)).squeeze(0)

        if self.load_name:
            return {"pc": point_set, "ids": label, "name": name}
        else:
            return {"pc": point_set, "ids": label}

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.data.shape[0]


def translate_pointcloud(pointcloud):
    """Applies random translation to a point cloud.

    Args:
        pointcloud (np.ndarray): Point cloud data of shape (N, 3).

    Returns:
        np.ndarray: Translated point cloud.
    """
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """Adds random jitter to a point cloud.

    Args:
        pointcloud (np.ndarray): Point cloud data of shape (N, 3).
        sigma (float, optional): Standard deviation of the jitter.
            Defaults to 0.01.
        clip (float, optional): Clipping value for the jitter. Defaults to 0.02.

    Returns:
        np.ndarray: Jittered point cloud.
    """
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    """Applies random rotation around the Y-axis (up-axis) to a point cloud.

    Args:
        pointcloud (np.ndarray): Point cloud data of shape (N, 3).

    Returns:
        np.ndarray: Rotated point cloud.
    """
    theta = np.pi * 2 * np.random.choice(24) / 24  # Random angle
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    # Rotate only the X and Z coordinates
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)
    return pointcloud

# --- Mapping from synset ID to category name (ShapeNet) ---
# From: https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat',  # Not in dataset, merged into vessels
    # '02834778': 'bicycle', # Not in taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


def init_np_seed(worker_id):
    """Initializes the NumPy random seed for a worker."""
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)  # Wrap around to 32-bit unsigned int


class Uniform15KPC(torch.utils.data.Dataset):
    """
    Dataset for uniformly sampled 15K point clouds (typically from ShapeNet).
    Loads pre-sampled point clouds from .npy files.

    Args:
        root (str): Root directory of the dataset.
        subdirs (list): List of subdirectories (synset IDs) to load.  e.g. ['03001627'] for chairs.
        tr_sample_size (int): Number of points to sample for training.
        te_sample_size (int): Number of points to sample for testing/validation.
        split (str): Dataset split ('train', 'test', or 'val').
        scale (float): Unused scaling factor (legacy parameter).
        standardize_per_shape (bool): Whether to standardize each point cloud by its own mean and std.
        normalize_per_shape (bool):  Whether to normalize each point cloud to be zero-mean.
        random_offset (bool) : Whether to randomly translate the pointcloud
        random_subsample (bool): Whether to randomly subsample to the specified number of points.
        normalize_std_per_axis (bool): Whether to normalize the standard deviation per axis.
        all_points_mean (np.ndarray, optional): Pre-calculated mean of all point clouds.  If None, it's calculated.
        all_points_std (np.ndarray, optional): Pre-calculated standard deviation of all point clouds. If None, it's calculated.
        input_dim (int): Dimensionality of input points (usually 3).

    Attributes:
        root, split, in_tr_sample_size, in_te_sample_size, subdirs, scale,
        random_offset, random_subsample, input_dim, max,
        normalize_per_shape, normalize_std_per_axis,
        standardize_per_shape, k (tr_sample_size),
        train_points, test_points, tr_sample_size, te_sample_size:
            (as described in Args)
        all_cate_mids (list): List of (category, model_id) tuples, e.g., [('03001627', 'train/1234'), ...].
        cate_idx_lst (list): List of category indices corresponding to each point cloud.
        all_points (np.ndarray):  All loaded point clouds, concatenated, shape (N, 15000, 3).
        shuffle_idx (list):  Indices for shuffling the data.
        all_points_mean (np.ndarray): Mean of all points (or per shape if normalize_per_shape).
        all_points_std (np.ndarray): Standard deviation of all points (or per shape, or per axis).
        global_pc_std (np.array): A placeholder for the global point cloud standard deviation (set to 1).

    """

    def __init__(self, root, subdirs, tr_sample_size=10000, te_sample_size=10000, split='train', scale=1.,
                 standardize_per_shape=False,
                 normalize_per_shape=False, random_offset=False, random_subsample=False, normalize_std_per_axis=False,
                 all_points_mean=None, all_points_std=None, input_dim=3):
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val'], "Invalid split.  Must be 'train', 'test', or 'val'."
        self.in_tr_sample_size = tr_sample_size  # Input sample size (before subsampling)
        self.in_te_sample_size = te_sample_size  # Input sample size (before subsampling)
        self.subdirs = subdirs
        self.scale = scale  # NOTE: This parameter is not actually used.
        self.random_offset = random_offset
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        if split == 'train':
            self.max = tr_sample_size
        elif split == 'val':
            self.max = te_sample_size
        else:
            self.max = max((tr_sample_size, te_sample_size))

        self.all_cate_mids = []  # (category_id, model_id) tuples
        self.cate_idx_lst = []   # Category indices
        self.all_points = []     # All point clouds

        # --- Load Point Clouds ---
        self._load_point_clouds()

        # --- Normalization/Standardization ---
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.standardize_per_shape = standardize_per_shape

        if all_points_mean is not None and all_points_std is not None:
            # Use precomputed statistics
            self.all_points_mean = all_points_mean
            self.all_points_std = all_points_std
        elif self.normalize_per_shape:
            # Normalize each point cloud to have zero mean.  Deprecated.
            raise NotImplementedError("normalize_per_shape==True is deprecated")
            # B, N = self.all_points.shape[:2]
            # self.all_points_mean = self.all_points.mean(axis=1).reshape(B, 1, input_dim)
            # if normalize_std_per_axis:
            #     self.all_points_std = self.all_points.reshape(B, N, -1).std(axis=1).reshape(B, 1, input_dim)
            # else:
            #     self.all_points_std = self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        else:
            # Normalize across the entire dataset
            self.all_points_mean = self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(1, 1, input_dim)
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(-1, input_dim).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)

        #  modified - just subtract the mean, set std to 1
        self.all_points = (self.all_points - self.all_points_mean)
        self.all_points_std = np.array([[[1]]])  # Set std to 1
        self.global_pc_std = np.array([[[1]]])   # For compatibility with other dataset classes

        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

        self.tr_sample_size = min(10000, tr_sample_size) # can't sample more than what exists
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d" % len(self.train_points))
        print("Min number of points: (train)%d (test)%d" % (self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"
        self.k = self.tr_sample_size # for the dataloader

    def _load_point_clouds(self):
        """Loads point clouds from .npy files and populates data structures."""
        for cate_idx, subd in enumerate(self.subdirs):
            # NOTE: [subd] here is the synset id (e.g., '03001627')
            sub_path = os.path.join(self.root, subd, self.split)
            if not os.path.isdir(sub_path):
                print("Directory missing: %s" % sub_path)
                continue

            all_mids = []  # Model IDs within this category
            for x in os.listdir(sub_path):
                if not x.endswith('.npy'):
                    continue
                all_mids.append(os.path.join(self.split, x[:-len('.npy')]))  # e.g., 'train/1a2b3c...'

            # NOTE: [mid] contains the split: i.e. "train/<mid>" or "val/<mid>" or "test/<mid>"
            for mid in all_mids:
                obj_fname = os.path.join(self.root, subd, mid + ".npy")
                try:
                    point_cloud = np.load(obj_fname)  # (15k, 3)
                except Exception as e:
                    print(f"Error loading {obj_fname}: {e}")
                    continue

                assert point_cloud.shape[0] == 15000, f"Point cloud {obj_fname} has incorrect shape: {point_cloud.shape}"
                self.all_points.append(point_cloud[np.newaxis, ...])  # Add a batch dimension
                self.cate_idx_lst.append(cate_idx)  # Category index
                self.all_cate_mids.append((subd, mid))  # (category_id, model_id)

        # Shuffle the data deterministically
        self.shuffle_idx = list(range(len(self.all_points)))
        random.Random(38383).shuffle(self.shuffle_idx)  # Consistent shuffling
        self.cate_idx_lst = [self.cate_idx_lst[i] for i in self.shuffle_idx]
        self.all_points = [self.all_points[i] for i in self.shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in self.shuffle_idx]

        self.all_points = np.concatenate(self.all_points)  # (N, 15000, 3)


    def get_pc_stats(self, idx):
        """Returns the mean and standard deviation for normalization."""
        if self.normalize_per_shape:
            m = self.all_points_mean[idx].reshape(1, self.input_dim)
            s = self.all_points_std[idx].reshape(1, -1)
            return m, s

        return self.all_points_mean.reshape(1, -1), self.all_points_std.reshape(1, -1)

    def renormalize(self, mean, std):
        """Renormalizes the dataset with new mean and standard deviation."""
        self.all_points = self.all_points * self.all_points_std + self.all_points_mean
        self.all_points_mean = mean
        self.all_points_std = std
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]

    def save_statistics(self, save_dir):
        """Saves the dataset statistics (mean, std, shuffle indices) to files."""
        np.save(os.path.join(save_dir, f"{self.split}_set_mean.npy"), self.all_points_mean)
        np.save(os.path.join(save_dir, f"{self.split}_set_std.npy"), self.all_points_std)
        np.save(os.path.join(save_dir, f"{self.split}_set_idx.npy"), np.array(self.shuffle_idx))

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.train_points)

    def __getitem__(self, idx):
        """Retrieves a single sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing:
                - 'ids': Sample index.
                - 'set':  The point cloud (either train or test, depending on self.split).
                - 'offset': The mean of the point cloud (if standardize_per_shape).
                - 'mean': The normalization mean.
                - 'std': The normalization standard deviation.
                - 'label': The category index.
                - 'sid': The synset ID (category).
                - 'mid': The model ID.
        """
        if self.split == 'train':
            out_points = self.train_points[idx]
            sample_size = self.tr_sample_size
        else:  # 'test' or 'val'
            out_points = self.test_points[idx]
            sample_size = self.te_sample_size

        if self.random_subsample:
            tr_idxs = np.random.choice(out_points.shape[0], sample_size, replace=False)
        else:
            tr_idxs = np.arange(sample_size)
        out_points = torch.from_numpy(out_points[tr_idxs, :]).float()

        offset = out_points.mean(0, keepdim=True)  # Center of the point cloud

        if self.standardize_per_shape:
            # Center the point cloud
            out_points -= offset

        if self.random_offset:
            # scale data offset
            if random.uniform(0., 1.) < 0.2:
                scale = random.uniform(1., 1.5)
                out_points -= offset
                offset *= scale
                out_points += offset


        m, s = self.get_pc_stats(idx)  # Get normalization stats
        m, s = torch.from_numpy(np.asarray(m)), torch.from_numpy(np.asarray(s))
        cate_idx = self.cate_idx_lst[idx]  # Category index
        sid, mid = self.all_cate_mids[idx]  # (synset_id, model_id)

        return {
            'ids': idx,
            'set': out_points,
            'offset': offset,
            'mean': m, 'std': s, 'label': cate_idx,
            'sid': sid, 'mid': mid
        }

class ShapeNet15kPointClouds(Uniform15KPC):
    """
    Dataset class for ShapeNet with 15K uniformly sampled points.
    This is a specialized version of Uniform15KPC for ShapeNet data,
    handling category mapping and dataset-specific settings.

    Args:
        root (str): Root directory of the ShapeNetCore.v2.PC15k dataset.  Should contain
            subdirectories named by synset IDs (e.g., '03001627').
        categories (list): List of ShapeNet category names to load (e.g.,
            ['airplane', 'chair']).  Use ['all'] for all categories.
        tr_sample_size (int): Number of points for training samples.
        te_sample_size (int): Number of points for testing/validation samples.
        split (str): 'train', 'test', or 'val'.
        scale (float): Unused scaling factor (legacy parameter).
        normalize_per_shape (bool): Normalize each point cloud to zero-mean.
        standardize_per_shape (bool): Standardize each point cloud by its own mean and std dev.
        normalize_std_per_axis (bool): Normalize the standard deviation per axis.
        random_offset (bool): Whether to randomly offset
        random_subsample (bool): Randomly subsample to the specified number of points.
        all_points_mean (np.ndarray, optional): Pre-calculated mean.
        all_points_std (np.ndarray, optional): Pre-calculated std.

    Attributes:
        Inherits attributes from Uniform15KPC.
        cates (list): List of category names (e.g., ['airplane', 'chair']).
        synset_ids (list): List of synset IDs (e.g., ['03001627', '04379243']).
        gravity_axis (int): The gravity axis (1 for ShapeNet v2, which is the Y-axis).
        display_axis_order (list): Order of axes for display (0, 2, 1 for ShapeNet v2,
            which corresponds to X, Z, Y).
    """

    def __init__(self, root="/data/shapenet/ShapeNetCore.v2.PC15k",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 standardize_per_shape=False,
                 normalize_std_per_axis=False,
                 random_offset=False,
                 random_subsample=False,
                 all_points_mean=None, all_points_std=None):

        self.k = tr_sample_size  # Keep track of training sample size
        self.root = root
        self.split = split
        assert self.split in ['train', 'test', 'val'], "Invalid split. Must be 'train', 'test', or 'val'."
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        if 'all' in categories:
            self.synset_ids = list(cate_to_synsetid.values())  # All categories
        else:
            self.synset_ids = [cate_to_synsetid[c] for c in self.cates]  # Specific categories

        assert 'v2' in root, "Only supporting ShapeNetCore v2 right now."
        self.gravity_axis = 1  # Y-axis is up in ShapeNet v2
        self.display_axis_order = [0, 2, 1]  # X, Z, Y for visualization

        super().__init__(
            root, self.synset_ids,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split,
            scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            standardize_per_shape=standardize_per_shape,
            random_offset=random_offset,
            random_subsample=random_subsample,
            all_points_mean=all_points_mean,
            all_points_std=all_points_std,
            input_dim=3)

        print(f"ShapeNet15kPointClouds initialized for categories: {categories}, split: {split}")


def collate_fn(batch):
    """
    Collation function for DataLoader.  Combines samples into a batch.

    Args:
        batch (list): List of dictionaries, each from `__getitem__`.

    Returns:
        dict: A dictionary representing the batch, with batched tensors.
    """
    ret = dict()
    # Collect all keys from the first sample
    for k, v in batch[0].items():
        ret.update({k: [b[k] for b in batch]})

    # Stack tensors for these keys
    mean = torch.stack(ret['mean'], dim=0)  # [B, 1, 3]
    std = torch.stack(ret['std'], dim=0)  # [B, 1, 1]

    s = torch.stack(ret['set'], dim=0)  # [B, N, 3]  (N varies)
    offset = torch.stack(ret['offset'], dim=0) # Stack the offset
    mask = torch.zeros(s.size(0), s.size(1)).bool()  # [B, N]  (dummy mask)
    cardinality = torch.ones(s.size(0)) * s.size(1)  # [B,] (number of points)

    # Update the dictionary with batched tensors
    ret.update({'pc': s, 'offset': offset, 'set_mask': mask, 'cardinality': cardinality,
                'mean': mean, 'std': std})
    return ret

def build(args):
    """
    Builds training and validation datasets and data loaders for ShapeNet.

    Args:
        args: Configuration arguments (should have attributes like batch_size,
              num_workers, shapenet_data_dir, etc.).

    Returns:
        tuple: (train_dataset, val_dataset, data_loaders), where data_loaders
            is a dictionary with keys 'Train' and 'Test'.
    """
    # train_dataset = ShapeNet15kPointClouds(
    #     categories=["airplane"],  # Example: Only airplanes
    #     split='train',
    #     tr_sample_size=2048,  # Or: args.tr_max_sample_points
    #     te_sample_size=2048,  # Or: args.te_max_sample_points
    #     scale=1.,  # Or: args.dataset_scale
    #     root="/train/ShapeNetCore.v2.PC15k/",  # Or: args.shapenet_data_dir
    #     standardize_per_shape=False,  # Or: args.standardize_per_shape
    #     normalize_per_shape=False,   # Or: args.normalize_per_shape
    #     normalize_std_per_axis=False,  # Or: args.normalize_std_per_axis
    #     random_subsample=True  # Important for training with random input
    # )
    import argparse

    parser = argparse.ArgumentParser()

    dataset_path = "Your/Folder/To/Meshes"
    parser.add_argument('--vertebra_data_dir', type=str, default=dataset_path, help='Path to vertebra mesh files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--sample_size', type=int, default=2048, help='Number of points to sample')

    args = parser.parse_args()
    train_dataset, val_dataset, data_loaders = build_vertebra_dataloaders(args)

    # train_sampler = None  # Could add a custom sampler if needed

    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=(train_sampler is None),  # Shuffle if no custom sampler
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     sampler=train_sampler,
    #     drop_last=True,  # Important for consistent batch size
    #     collate_fn=collate_fn,  # Use the custom collate function
    #     worker_init_fn=init_np_seed  # Ensure reproducibility
    # )
    #
    # val_dataset = ShapeNet15kPointClouds(
    #     categories=["airplane"],  # Use the same categories as training
    #     split='val',
    #     tr_sample_size=2048,
    #     te_sample_size=2048,
    #     scale=1.,
    #     root="/train/ShapeNetCore.v2.PC15k/",
    #     standardize_per_shape=False,
    #     normalize_per_shape=False,
    #     normalize_std_per_axis=False,
    #     all_points_mean=train_dataset.all_points_mean,  # Use training set stats
    #     all_points_std=train_dataset.all_points_std,  # Use training set stats
    #     random_subsample=False # Usually no random subsampling for validation
    # )
    #
    # val_loader = torch.utils.data.DataLoader(
    #     dataset=val_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,  # No need to shuffle validation data
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=False,  # Keep all validation samples
    #     collate_fn=collate_fn,
    #     worker_init_fn=init_np_seed
    # )

    # data_loaders = {"Train": train_loader, "Test": val_loader}  # Consistent names

    return train_dataset, val_dataset, data_loaders