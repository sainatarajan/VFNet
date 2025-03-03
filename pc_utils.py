import math
import random
from typing import Tuple, Optional, Dict, List
import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData
from pyntcloud import PyntCloud
import pandas as pd
from typing import Union, Tuple
from tqdm import tqdm


# --- Point Cloud Augmentations ---
def random_scale(data: torch.Tensor, scales: Tuple[float, float]) -> torch.Tensor:
    """Scale point cloud by a random factor within given range."""
    assert isinstance(scales, (tuple, list)) and len(scales) == 2
    scale = random.uniform(*scales)
    return data * scale


def random_flip(data: torch.Tensor, p: float = 0.5, axis: int = -1) -> torch.Tensor:
    """Flip point cloud along an axis with probability p."""
    assert data.dim() == 2, "Input must be [N, 3]"
    if axis == -1:
        flipped = False
        flip_dim = max(data.shape)
        for ax in range(0, flip_dim, 2):
            p_adj = p * (0.5 if flipped else 1)  # Reduce chance of double flip
            if random.random() < p_adj:
                data = data.clone()
                data[:, ax] *= -1
                flipped = True
    elif random.random() < p:
        data = data.clone()
        data[:, axis] *= -1
    return data


def random_rotate(data: torch.Tensor, degrees: Union[Tuple[float, float], float], axis: int = -1) -> torch.Tensor:
    """Rotate point cloud around an axis by a random angle in degrees."""
    degrees = (-abs(degrees), abs(degrees)) if isinstance(degrees, (int, float)) else degrees
    assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
    angle = math.radians(random.uniform(*degrees))
    sin, cos = math.sin(angle), math.cos(angle)
    matrices = [
        [[1, 0, 0], [0, cos, sin], [0, -sin, cos]],  # X-axis
        [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]],  # Y-axis
        [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]   # Z-axis
    ]
    if axis == -1:
        for ax in range(3):
            data = data @ torch.tensor(matrices[ax], dtype=data.dtype).T
    else:
        data = data @ torch.tensor(matrices[axis], dtype=data.dtype).T
    return data


def uniform_random_rotation(data: torch.Tensor) -> torch.Tensor:
    """Apply uniform random 3D rotation (Avro 1992)."""
    R = torch.eye(3)
    x1 = random.random()
    R[0, 0] = R[1, 1] = math.cos(2 * math.pi * x1)
    R[0, 1] = -math.sin(2 * math.pi * x1)
    R[1, 0] = math.sin(2 * math.pi * x1)

    x2, x3 = 2 * math.pi * random.random(), random.random()
    v = torch.tensor([
        math.cos(x2) * math.sqrt(x3),
        math.sin(x2) * math.sqrt(x3),
        math.sqrt(1 - x3)
    ])
    H = torch.eye(3) - 2 * torch.outer(v, v)
    M = -(H @ R)

    data = data.reshape(-1, 3)
    mean = data.mean(dim=0)
    return (data - mean) @ M + mean @ M


def slight_rotation(point_cloud: torch.Tensor, point_normals: Optional[torch.Tensor] = None) -> Tuple[
    torch.Tensor, Optional[torch.Tensor]]:
    """Apply slight random rotation and tilt to point cloud."""
    # Y-axis rotation
    theta_y = torch.rand(1) * math.pi / 5 - math.pi / 10
    R = torch.eye(3)
    R[0, 0] = R[2, 2] = torch.cos(theta_y)
    R[0, 2] = torch.sin(theta_y)
    R[2, 0] = -R[0, 2]
    transform = torch.eye(4)
    transform[:3, :3] = R

    # Tilt
    theta = torch.rand(1) * 2 * math.pi
    tilt_axis = torch.tensor([math.cos(theta), 0, math.sin(theta)])
    tilt_angle = torch.rand(1) * math.pi / 5 - math.pi / 10
    tilt_rot = angle_axis_to_rotation_matrix(tilt_angle * tilt_axis)[0]
    transform = transform @ tilt_rot

    # Apply to points
    pc_homo = torch.ones(point_cloud.shape[0], 4)
    pc_homo[:, :3] = point_cloud
    pc_aug = (transform @ pc_homo.T).T[:, :3]

    # Apply to normals if provided
    normals_aug = (transform[:3, :3] @ point_normals.T).T if point_normals is not None else None
    return pc_aug, normals_aug


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert angle-axis to 4x4 rotation matrix (from PyTorch Geometry)."""
    theta2 = angle_axis.pow(2).sum(dim=1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-6)
    wxyz = angle_axis / theta
    wx, wy, wz = wxyz[:, 0], wxyz[:, 1], wxyz[:, 2]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    k_one = torch.ones_like(wx)

    r = torch.stack([
        cos_t + wx * wx * (k_one - cos_t),
        wx * wy * (k_one - cos_t) - wz * sin_t,
        wy * sin_t + wx * wz * (k_one - cos_t),
        wz * sin_t + wx * wy * (k_one - cos_t),
        cos_t + wy * wy * (k_one - cos_t),
        -wx * sin_t + wy * wz * (k_one - cos_t),
        -wy * sin_t + wx * wz * (k_one - cos_t),
        wx * sin_t + wy * wz * (k_one - cos_t),
        cos_t + wz * wz * (k_one - cos_t)
    ], dim=1).view(-1, 3, 3)

    mask = (theta2 > 1e-6).float()
    taylor = torch.stack([k_one, -wz, wy, wz, k_one, -wx, -wy, wx, k_one], dim=1).view(-1, 3, 3)
    rot = mask * r + (1 - mask) * taylor

    out = torch.eye(4).repeat(angle_axis.shape[0], 1, 1)
    out[:, :3, :3] = rot
    return out


# --- Utilities ---
def calculate_point_cloud_stats(folder: str, data_files: List[Dict]) -> Tuple[float, List[int], Dict]:
    """Compute std, point counts, and scales for point clouds."""
    stds, point_counts, scales = [], [], {}
    for file in tqdm(data_files, desc="Processing point clouds"):
        ply = PlyData.read(file["ply_file"])
        pc = np.vstack([ply.elements[0]["x"], ply.elements[0]["y"], ply.elements[0]["z"]]).T
        point_counts.append(pc.shape[0])
        scales[file["id"]] = np.max(np.linalg.norm(pc, axis=1))
        stds.append(np.linalg.norm(pc, axis=1).std())
    return np.mean(stds), point_counts, scales


def sample_facets(point_cloud: torch.Tensor, facet_areas: np.ndarray, k: int, ply_load: PlyData) -> torch.Tensor:
    """Sample points from facets proportional to area."""
    facet_idx = np.vstack([ply_load.elements[1]["v1"], ply_load.elements[1]["v2"], ply_load.elements[1]["v3"]]).T
    facets = point_cloud[facet_idx]
    sampled = random.choices(range(facet_areas.shape[0]), weights=facet_areas, k=k)
    s, t = torch.sort(torch.rand(k, 2), dim=1)[0].split(1, dim=1)
    return (s * facets[sampled, 0] + (t - s) * facets[sampled, 1] + (1 - t) * facets[sampled, 2])


# --- Loss ---
class ChamferLoss(nn.Module):
    """Compute Chamfer distance between two point clouds."""

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.backprop_num = 0  # Restored counter for backpropagation tracking

    def batch_pairwise_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xx = x.pow(2).sum(dim=2)
        yy = y.pow(2).sum(dim=2)
        xy = torch.bmm(x, y.transpose(2, 1))
        return xx[:, :, None] + yy[:, None, :] - 2 * xy

    def forward(self, ground_truth: torch.Tensor, reconstruction: torch.Tensor, surface_normals_tuple=None) -> Dict[str, torch.Tensor]:
        P = self.batch_pairwise_dist(ground_truth, reconstruction)
        loss_1 = torch.mean(torch.clamp(P.min(dim=1)[0], min=1e-10))
        loss_2 = torch.mean(torch.clamp(P.min(dim=2)[0], min=1e-10))
        total = (loss_1 + loss_2) * 1000
        return {"total_loss": total, "chamfer": total}


# --- I/O ---
def save_pointcloud(point_cloud: torch.Tensor, save_path: str, point_normals: Optional[torch.Tensor] = None) -> None:
    """Save point cloud to PLY file."""
    if point_cloud.dim() == 3:
        point_cloud = point_cloud[0].T
    data = {"x": point_cloud[:, 0], "y": point_cloud[:, 1], "z": point_cloud[:, 2]}
    if point_normals is not None:
        if point_normals.dim() == 3:
            point_normals = point_normals[0].T
        data.update({"nx": point_normals[:, 0], "ny": point_normals[:, 1], "nz": point_normals[:, 2]})
    PyntCloud(pd.DataFrame(data)).to_file(save_path)


def save_ply_manual(verts: torch.Tensor, filename: str, faces: np.ndarray) -> None:
    """Manually write PLY file with vertices and faces."""
    if verts.shape[1] != 3:
        verts = verts.T
    with open(filename, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {verts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def facets_from_grid(num_points: int, reverse_facets: bool = False) -> np.ndarray:
    """Generate facets for a grid of points."""
    facets = []
    n = num_points * num_points
    for i in range(1, n - num_points):
        if i % num_points != 0:
            if not reverse_facets:
                facets.extend([[i - 1, i, i + num_points - 1], [i, i + num_points, i + num_points - 1]])
            else:
                facets.extend([[i - 1, i + num_points - 1, i], [i, i + num_points - 1, i + num_points]])
    return np.array(facets)


def save_cloud_rgb(cloud: torch.Tensor, red: np.ndarray, green: np.ndarray, blue: np.ndarray, filename: str) -> None:
    """Save point cloud with RGB colors."""
    cloud = cloud.cpu()
    data = pd.DataFrame({
        "x": cloud[0], "y": cloud[1], "z": cloud[2],
        "red": red, "green": green, "blue": blue
    }).astype({"red": np.uint8, "green": np.uint8, "blue": np.uint8})
    PyntCloud(data).to_file(filename)