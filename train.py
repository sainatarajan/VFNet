import json
from datetime import datetime
from pathlib import Path
import torch
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from pc_utils import loss_std_correlation
from utils import save_pretraining


def log_epoch_loss(epoch: int, phase: str, loss_dict: dict, std: float) -> None:
    """Print epoch losses with standardized formatting."""
    print(f"Epoch {epoch} [{phase}]", end="")
    for key, value in loss_dict.items():
        if key == "num_samples":
            continue
        if "coeff" in key:
            print(f", {key.capitalize()}: {value[0]:.2f}", end="")
        elif "chamfer" in key:
            mean = torch.tensor(value).mean()
            print(f", {key.capitalize()}: {mean:.2f} ({mean * std:.2f})", end="")
        elif isinstance(value, list):
            mean = torch.tensor(value).mean()
            print(f", {key.capitalize()}: {mean:.2f}", end="")
    print()


def update_epoch_loss(epoch_dict: dict, batch_dict: dict) -> dict:
    """Append batch losses to epoch dictionary."""
    if len(epoch_dict) == 1:  # First batch
        epoch_dict.update({k: [float(v)] for k, v in batch_dict.items() if "coeff" not in k})
    else:
        for k, v in batch_dict.items():
            if "coeff" not in k:
                epoch_dict[k].append(float(v))
    return epoch_dict


# def log_to_tensorboard(writer: SummaryWriter, phase: str, epoch: int, loss_dict: dict, std: float) -> None:
#     """Log metrics to TensorBoard."""
#     for key, value in loss_dict.items():
#         if key == "num_samples":
#             continue
#         if key == "total_loss":
#             total = torch.tensor(value).sum() / loss_dict["num_samples"]
#             writer.add_scalar(f"foldingnet/{phase}_epoch", total, epoch)
#         elif "chamfer" in key:
#             mean = torch.tensor(value).mean() * std
#             writer.add_scalar(f"{key.capitalize()}/{phase}", mean, epoch)
#         else:
#             mean = torch.tensor(value).mean() if isinstance(value, list) else value
#             writer.add_scalar(f"{key.capitalize()}/{phase}", mean, epoch)


def train(data_loaders: dict, model, optimizer, scheduler, device: torch.device, args,
          preloaded_epoch: int = None) -> None:
    """Train the model with progress tracking via tqdm and checkpointing."""
    # Setup (stripped of logging, using a static dir)
    log_dir = Path("runs", "temp")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    start_epoch = preloaded_epoch or 0
    best_loss = float("inf")

    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        for phase in ["Train", "Test"]:
            # Skip early test phases
            if epoch < args.num_epochs // 4 and epoch % 10 != 0 and phase == "Test":
                continue

            model.train(phase == "Train")
            epoch_loss_dict = {"num_samples": 0}

            # Use tqdm to track batches and display loss
            with tqdm(total=len(data_loaders[phase]), desc=f"{phase} Epoch {epoch}") as pbar:
                for batch in data_loaders[phase]:
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "Train"):
                        pc = batch["pc"].to(device)
                        output = model(pc, jacobian=(args.decoder.lower() == "stochman" and args.point_normals))
                        loss_dict = model.get_loss(epoch, pc, output)

                        if phase == "Train":
                            loss_dict["total_loss"].backward()
                            model.loss.backprop_num += 1
                            optimizer.step()

                        # if "std" in output:
                        #     loss_dict["std_corr"] = loss_std_correlation(pc, output).mean()

                        epoch_loss_dict = update_epoch_loss(epoch_loss_dict, loss_dict)
                        epoch_loss_dict["num_samples"] += pc.shape[0]

                    # Update tqdm with current loss
                    pbar.set_postfix({"loss": f"{torch.tensor(epoch_loss_dict['total_loss']).mean().item():.4f}"})
                    pbar.update(1)

            # Save checkpoint (no additional logging)
            epoch_loss = torch.tensor(epoch_loss_dict["total_loss"]).mean()
            if phase == "Test":
                save_interval = 1000 if args.num_epochs > 10000 else 100
                if epoch % save_interval == 0:
                    save_pretraining(log_dir / f"epoch_{epoch}.pth.tar", epoch, model, optimizer, best_loss)
                scheduler.step(epoch_loss)


if __name__ == "__main__":
    # Placeholder for args and execution logic
    pass