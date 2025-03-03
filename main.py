import argparse
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import dataloader
import utils
from model import vfnet
from train import train

parser = argparse.ArgumentParser(description="Training loop for point cloud models")
parser.add_argument("--dataset", default="vertebra", choices=["vertebra", "teeth", "modelnet40", "shapenetcore"], help="Dataset to use")
parser.add_argument("--x_train", help="Path to training data folder")
parser.add_argument("--x_val", help="Path to validation data folder")
parser.add_argument("--point_normals", action="store_true", help="Include point normals in data")
parser.add_argument("--std_training", action="store_true", help="Train standard deviation only")
parser.add_argument("--unn", default="3", help="UNNs to include, comma-separated (no spaces) or 'all'")
parser.add_argument("--num_workers", type=int, default=16, help="Number of data loader workers")
parser.add_argument("--cpu_only", action="store_true", help="Force CPU mode for debugging")
parser.add_argument("--model", default="vae", choices=["vae"], help="Model type")
parser.add_argument("--resume_from", default="", help="Path to state dict for resuming training")
parser.add_argument("--encoder", default="foldnet", help="Encoder type")
parser.add_argument("--decoder", default="stochman", help="Decoder type")
parser.add_argument("--point_encoding", action="store_true", help="Map to grid instead of using full grid")
parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (defaults to datetime)")
parser.add_argument("--commit_name", type=str, default=None, help="Commit name")
parser.add_argument("--commit_text", type=str, default=None, help="Commit message")
parser.add_argument("--rundatetime", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Run timestamp")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    utils.add_standard_args_to_parser(parser)
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu_only else "cuda" if torch.cuda.is_available() else "cpu")

    # Dataset setup
    if args.dataset in ["modelnet40", "shapenetcore", "vertebra"]:
        train_set, val_set, data_loaders = dataloader.build(args)
        collate_fn = None
    elif args.dataset == "teeth":
        unn_list = args.unn.split(",") if args.unn != "all" else "all"
        teeth_std = 9.8186 if unn_list == ["3"] else 11.75121
        args.teeth_std = teeth_std
        train_set = dataloader.Teeth_Dataset(
            unn=unn_list, folder_path=args.x_train, is_train=True, global_pc_std=teeth_std, args=args
        )
        val_set = dataloader.Teeth_Dataset(
            unn=unn_list, folder_path=args.x_val, is_train=False, global_pc_std=teeth_std, args=args
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Use 'teeth', 'modelnet40', or 'shapenetcore'.")

    # Model and training setup
    model = vfnet.VariationalAutoencoder(args, num_points=train_set.k, global_std=train_set.global_pc_std).to(device)
    optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.85, verbose=True)

    # Resume training if specified
    epoch = 0
    if args.resume_from:
        model = utils.load_pretrained_model(model, args.resume_from)
        epoch, opt_state, _ = utils.load_previous_training_params(args.resume_from, optimizer)
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        if args.std_training:
            model.decoder.init_std(device)
            optimizer = optim.Adam(model.decoder.std.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            epoch = 0
        else:
            optimizer.load_state_dict(opt_state)

    # Data loaders
    if not data_loaders:
        data_loaders = {
            "Train": DataLoader(
                train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                pin_memory=True, collate_fn=collate_fn, drop_last=True
            ),
            "Test": DataLoader(
                val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                pin_memory=True, collate_fn=collate_fn, drop_last=True
            ),
        }

    # Kick off training
    train(data_loaders=data_loaders, model=model, optimizer=optimizer, scheduler=scheduler, device=device, args=args, preloaded_epoch=epoch)