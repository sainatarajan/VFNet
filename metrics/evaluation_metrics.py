import torch
import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

from .StructuralLosses.match_cost import match_cost
from .StructuralLosses.nn_distance import nn_distance
from .utils import unmask


# --- Distance Metrics ---
def chamfer_distance(sample: torch.Tensor, ref: torch.Tensor, accelerated: bool = True) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """Compute Chamfer Distance (CD) between two point clouds."""
    if accelerated and sample.is_cuda:
        return nn_distance(sample, ref)
    # CPU fallback (batched)
    xx = torch.bmm(sample, sample.transpose(2, 1))
    yy = torch.bmm(ref, ref.transpose(2, 1))
    zz = torch.bmm(sample, ref.transpose(2, 1))
    diag_ind = torch.arange(sample.size(1), device=sample.device)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


def emd_approx(sample: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Approximate Earth Mover's Distance (EMD) using CUDA."""
    assert sample.shape[2] == 3 and sample.shape[1] == ref.shape[
        1], "Expected 3D point clouds with matching point counts"
    emd = match_cost(sample, ref) / sample.shape[1]
    return emd


def batch_emd_cd(sample: torch.Tensor, ref: torch.Tensor, batch_size: int, accelerated_cd: bool = True,
                 reduced: bool = True) -> Dict[str, torch.Tensor]:
    """Compute batched CD and EMD between sample and reference point clouds."""
    assert sample.shape[0] == ref.shape[0], f"Sample ({sample.shape[0]}) and ref ({ref.shape[0]}) sizes must match"
    cd, emd = [], []
    for b_start in tqdm(range(0, sample.shape[0], batch_size), desc="Batch EMD/CD"):
        b_end = min(sample.shape[0], b_start + batch_size)
        s_batch, r_batch = sample[b_start:b_end], ref[b_start:b_end]
        dl, dr = chamfer_distance(s_batch, r_batch, accelerated_cd)
        cd.append(dl.mean(dim=1) + dr.mean(dim=1))
        emd.append(emd_approx(s_batch, r_batch))
    cd, emd = torch.cat(cd), torch.cat(emd)
    return {"CD": cd.mean() if reduced else cd, "EMD": emd.mean() if reduced else emd}


def batch_emd_cd_masked(sample: torch.Tensor, sample_mask: torch.Tensor, ref: torch.Tensor, ref_mask: torch.Tensor,
                        batch_size: int, accelerated_cd: bool = True, reduced: bool = True) -> Dict[str, torch.Tensor]:
    """Compute EMD/CD for masked point clouds."""
    return batch_emd_cd(unmask(sample, sample_mask), unmask(ref, ref_mask), batch_size, accelerated_cd, reduced)


# --- Pairwise Distances ---
def pairwise_emd_cd(sample: torch.Tensor, ref: torch.Tensor, batch_size: int = 32, accelerated_cd: bool = True) -> \
Tuple[torch.Tensor, torch.Tensor]:
    """Compute pairwise CD and EMD between all sample and ref point clouds."""
    cd, emd = [], []
    for s in tqdm(sample, desc="Pairwise EMD/CD"):
        cd_batch, emd_batch = [], []
        for r_start in range(0, ref.shape[0], batch_size):
            r_end = min(ref.shape[0], r_start + batch_size)
            r_batch = ref[r_start:r_end]
            s_exp = s.unsqueeze(0).expand(r_batch.size(0), -1, -1)
            dl, dr = chamfer_distance(s_exp, r_batch, accelerated_cd)
            cd_batch.append((dl.mean(dim=1) + dr.mean(dim=1)))
            emd_batch.append(emd_approx(s_exp, r_batch))
        cd.append(torch.cat(cd_batch))
        emd.append(torch.cat(emd_batch))
    return torch.stack(cd), torch.stack(emd)


# --- KNN Accuracy ---
def knn_accuracy(Mxx: torch.Tensor, Mxy: torch.Tensor, Myy: torch.Tensor, k: int = 1) -> Dict[str, float]:
    """Compute K-NN accuracy metrics."""
    n0, n1 = Mxx.size(0), Myy.size(0)
    labels = torch.cat([torch.ones(n0), torch.zeros(n1)]).to(Mxx.device)
    M = torch.cat([torch.cat([Mxx, Mxy], dim=1), torch.cat([Mxy.t(), Myy], dim=1)], dim=0)
    val, idx = (M + torch.diag(torch.full((n0 + n1,), float("inf"), device=M.device))).topk(k, dim=0, largest=False)

    count = labels.index_select(0, idx.t()).sum(dim=0)
    pred = (count >= k / 2).float()

    tp = (pred * labels).sum()
    fp = (pred * (1 - labels)).sum()
    fn = ((1 - pred) * labels).sum()
    tn = ((1 - pred) * (1 - labels)).sum()

    return {
        "acc": (labels == pred).float().mean(),
        "acc_t": tp / (tp + fn + 1e-10),
        "acc_f": tn / (tn + fp + 1e-10)
    }


# --- MMD and Coverage ---
def mmd_coverage(dist: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute Minimum Matching Distance (MMD) and Coverage."""
    min_ref, min_idx = dist.min(dim=0)
    min_sample = dist.min(dim=1)[0]
    return {
        "lgan_mmd": min_ref.mean(),
        "lgan_mmd_smp": min_sample.mean(),
        "lgan_cov": torch.tensor(min_idx.unique().size(0) / dist.size(1), device=dist.device)
    }


# --- Main Metric Computation ---
def compute_all_metrics(sample: torch.Tensor, ref: torch.Tensor, batch_size: int = 32, accelerated_cd: bool = True) -> \
Dict[str, float]:
    """Compute all metrics: MMD, COV, 1-NNA for CD and EMD."""
    with torch.no_grad():
        results = {}
        # Pairwise distances
        cd_rs, emd_rs = pairwise_emd_cd(ref, sample, batch_size, accelerated_cd)
        cd_rr, emd_rr = pairwise_emd_cd(ref, ref, batch_size, accelerated_cd)
        cd_ss, emd_ss = pairwise_emd_cd(sample, sample, batch_size, accelerated_cd)

        # MMD and Coverage
        results.update({f"{k}-CD": v for k, v in mmd_coverage(cd_rs.t()).items()})
        results.update({f"{k}-EMD": v for k, v in mmd_coverage(emd_rs.t()).items()})

        # 1-NN Accuracy
        results.update({f"1-NN-CD-{k}": v for k, v in knn_accuracy(cd_rr, cd_rs, cd_ss).items()})
        results.update({f"1-NN-EMD-{k}": v for k, v in knn_accuracy(emd_rr, emd_rs, emd_ss).items()})

        return results


def compute_all_metrics_masked(sample: torch.Tensor, sample_mask: torch.Tensor, ref: torch.Tensor,
                               ref_mask: torch.Tensor,
                               batch_size: int = 32, accelerated_cd: bool = True) -> Dict[str, float]:
    """Compute all metrics for masked point clouds."""
    return compute_all_metrics(unmask(sample, sample_mask), unmask(ref, ref_mask), batch_size, accelerated_cd)


# --- JSD (Occupancy Grid Entropy) ---
def unit_cube_grid(resolution: int, clip_sphere: bool = False) -> Tuple[np.ndarray, float]:
    """Generate a 3D grid in unit cube."""
    spacing = 1.0 / (resolution - 1)
    grid = np.stack(np.meshgrid(*[np.linspace(-0.5, 0.5, resolution)] * 3), axis=-1)
    if clip_sphere:
        grid = grid.reshape(-1, 3)[norm(grid, axis=1) <= 0.5]
    return grid, spacing


def occupancy_entropy(pclouds: np.ndarray, resolution: int, in_sphere: bool = False) -> Tuple[float, np.ndarray]:
    """Estimate entropy of occupancy grid activation."""
    grid, _ = unit_cube_grid(resolution, in_sphere)
    nn = NearestNeighbors(n_neighbors=1).fit(grid.reshape(-1, 3))
    counters = np.zeros(grid.size // 3)

    for pc in pclouds:
        indices = nn.kneighbors(pc, return_distance=False).squeeze()
        counters[indices] += 1

    probs = counters[counters > 0] / len(pclouds)
    ent = sum(entropy([p, 1 - p]) for p in probs) / len(counters)
    return ent, counters


def jensen_shannon_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute Jensen-Shannon Divergence between two distributions."""
    P, Q = P / P.sum(), Q / Q.sum()
    M = (P + Q) / 2
    return entropy(M, base=2) - (entropy(P, base=2) + entropy(Q, base=2)) / 2


def jsd_between_point_clouds(sample: np.ndarray, ref: np.ndarray, resolution: int = 28) -> float:
    """Compute JSD between two point cloud sets."""
    sample_ent, sample_grid = occupancy_entropy(sample, resolution, in_sphere=True)
    ref_ent, ref_grid = occupancy_entropy(ref, resolution, in_sphere=True)
    return jensen_shannon_divergence(sample_grid, ref_grid)


if __name__ == "__main__":
    x, y = torch.rand(2, 10, 3).cuda(), torch.rand(2, 10, 3).cuda()
    dl, dr = chamfer_distance(x, y)
    print(f"CD Left: {dl.mean().item():.4f}, CD Right: {dr.mean().item():.4f}")