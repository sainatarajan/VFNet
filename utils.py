import os
import shutil
from typing import Optional, Tuple
import torch
import torch.nn as nn
from argparse import ArgumentParser

def add_standard_args_to_parser(parser: ArgumentParser) -> ArgumentParser:
    """Add common training args to parser."""
    parser.add_argument("--k", type=int, default=None, help="Number of nearest neighbors for KNN")
    parser.add_argument("--feat_dims", type=int, default=512, help="Feature dimensionality")
    parser.add_argument("--fold_orig_shape", default="plane", help="Original shape for folding")
    parser.add_argument("--num_epochs", type=int, default=15000, help="Number of training epochs")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--patience", type=int, default=50, help="Scheduler patience")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    return parser

def str_to_bool(s: str) -> bool:
    """Convert string 'True'/'False' to boolean."""
    return s == "True"

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class PrintLayer(nn.Module):
    """Debug layer to print input shape."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape)
        return x

class Identity(nn.Module):
    """Pass-through layer."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def save_checkpoint(file_path: str, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    description: Optional[str] = None) -> None:
    """Save training checkpoint."""
    state = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    if scheduler:
        state["scheduler"] = scheduler.state_dict()
    if description:
        state["description"] = description
    torch.save(state, file_path)

def load_checkpoint(ckpt_path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Tuple[int, nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """Load training checkpoint."""
    ckpt = torch.load(ckpt_path)
    print(f"Model description: {ckpt.get('description', 'None')}")
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt and scheduler:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], model, optimizer, scheduler

def save_pretraining(file_path: str, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, best_loss: float) -> None:
    """Save pretraining state for crash recovery."""
    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "best_loss": best_loss}, file_path)

def load_previous_training_params(file_path: str, optimizer: torch.optim.Optimizer) -> Tuple[int, dict, float]:
    """Load pretraining state."""
    state = torch.load(file_path)
    return state["epoch"], state["optimizer"], state["best_loss"]

def load_pretrained_model(model: nn.Module, load_path: Optional[str] = None) -> nn.Module:
    """Load pretrained weights into model, matching available keys."""
    if not load_path:
        return model
    loaded = torch.load(load_path)["state_dict"]
    model_dict = model.state_dict()
    for k in loaded:
        if k in model_dict:
            model_dict[k] = loaded[k]
            print(f"    Loaded weight: {k}")
        else:
            print(f"WARNING: Weight {k} not in model")
    model.load_state_dict(model_dict)
    return model