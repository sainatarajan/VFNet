import torch
from typing import Tuple
from .utils import unmask


def chamfer_loss_v1(output: torch.Tensor, output_mask: torch.BoolTensor,
                    target: torch.Tensor, target_mask: torch.BoolTensor) -> torch.Tensor:
    """Compute Chamfer Distance (CD) loss with masking, v1 implementation."""
    sizes = (~output_mask).sum(dim=1).tolist()
    out = output.flatten(0, 1)[~output_mask.flatten()]
    tgt = target.flatten(0, 1)[~target_mask.flatten()]
    out_split, tgt_split = out.split(sizes), tgt.split(sizes)

    cd = []
    for o, t in zip(out_split, tgt_split):
        o_exp = o[:, None, :].expand(-1, t.size(0), -1)  # [M, N, C]
        t_exp = t[None, :, :].expand(o.size(0), -1, -1)  # [M, N, C]
        dist = (o_exp - t_exp).pow(2).sum(dim=-1)  # [M, N]
        cd.append(dist.min(dim=0)[0].sum() + dist.min(dim=1)[0].sum())

    return torch.tensor(cd).mean()


def chamfer_loss(output: torch.Tensor, output_mask: torch.BoolTensor,
                 target: torch.Tensor, target_mask: torch.BoolTensor, accelerate: bool = False) -> torch.Tensor:
    """Compute Chamfer Distance loss with optional CUDA acceleration."""
    assert output.shape[-1] == 3, "Expected 3D point clouds"
    assert ((~output_mask).sum(-1) == (~target_mask).sum(-1)).all(), "Masked point counts must match"

    if accelerate and output.is_cuda:
        from metrics.StructuralLosses.nn_distance import nn_distance
        out_unmasked = unmask(output, output_mask).clone().contiguous()
        tgt_unmasked = unmask(target, target_mask).clone().contiguous()
        dl, dr = nn_distance(out_unmasked, tgt_unmasked)
        return (dl.sum(dim=1) + dr.sum(dim=1)).mean()
    return chamfer_loss_v1(output, output_mask, target, target_mask)


def emd_loss(output: torch.Tensor, output_mask: torch.BoolTensor,
             target: torch.Tensor, target_mask: torch.BoolTensor) -> torch.Tensor:
    """Compute Earth Mover's Distance loss with masking."""
    from metrics.StructuralLosses.match_cost import match_cost
    assert output.shape[-1] == 3, "Expected 3D point clouds"
    assert ((~output_mask).sum(-1) == (~target_mask).sum(-1)).all(), "Masked point counts must match"

    out_unmasked = unmask(output, output_mask).clone().contiguous()
    tgt_unmasked = unmask(target, target_mask).clone().contiguous()
    return match_cost(out_unmasked, tgt_unmasked).mean()


if __name__ == "__main__":
    # Quick test
    B, N, C = 2, 5, 3
    output = torch.rand(B, N, C)
    target = torch.rand(B, N, C)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :3] = True  # Mask out last 2 points
    cd = chamfer_loss(output, mask, target, mask, accelerate=False)
    emd = emd_loss(output, mask, target, mask)
    print(f"CD: {cd.item():.4f}, EMD: {emd.item():.4f}")