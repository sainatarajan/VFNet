import argparse
import json
import os
from pathlib import Path
import shutil
from typing import List, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools

import dataloader
import utils
import pc_utils
from model import vfnet
from model.model_utils import Residual_Linear_Layer
from nflows import transforms, distributions, flows
import torch.nn as nn
import metrics

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Training and sampling for point cloud VAE")
parser.add_argument("--dataset", default="teeth", choices=["teeth", "modelnet40", "shapenetcore"])
parser.add_argument("--x_train", help="Path to training data")
parser.add_argument("--x_val", help="Path to validation data")
parser.add_argument("--x_test", help="Path to test data")
parser.add_argument("--model_path", default="", help="Path to pretrained VAE model")
parser.add_argument("--flow_num_epochs", type=int, default=150)
parser.add_argument("--pe_num_epochs", type=int, default=300)
parser.add_argument("--flow_viz_epochs", type=int, default=20)
parser.add_argument("--flow_n_layers", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--one_fold", action="store_true")
parser.add_argument("--test_name", type=str, required=True)


# --- Models ---
class OneFold(nn.Module):
    def __init__(self, feat_dims: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(514, feat_dims), nn.Dropout(0.2), nn.ReLU(),
            Residual_Linear_Layer(feat_dims, feat_dims), nn.ReLU(),
            Residual_Linear_Layer(feat_dims, feat_dims), nn.ReLU(),
            nn.Linear(feat_dims, 2), nn.Tanh()
        )
        self.grid = torch.from_numpy(np.array(list(itertools.product(
            np.linspace(-1, 1, int(np.sqrt(2048))),
            np.linspace(-1, 1, int(np.sqrt(2048)))
        )))).float()

    def forward(self, x: torch.Tensor, add_n_points: int = 0) -> torch.Tensor:
        grid = self.grid.to(x.device)
        if add_n_points > 0:
            extra = torch.rand(add_n_points, 2, device=x.device) * 2 - 1
            grid = torch.cat([grid, extra], dim=0)
        x = x.repeat(1, grid.shape[0], 1)
        return self.net(torch.cat([x, grid.repeat(x.shape[0], 1, 1)], dim=-1))


class TwoFold(OneFold):
    def __init__(self, feat_dims: int = 256):
        super().__init__(feat_dims)
        self.fold2 = nn.Sequential(
            nn.Linear(514, feat_dims), nn.Dropout(0.2), nn.ReLU(),
            Residual_Linear_Layer(feat_dims, feat_dims), nn.ReLU(),
            Residual_Linear_Layer(feat_dims, feat_dims), nn.ReLU(),
            nn.Linear(feat_dims, feat_dims), nn.ReLU(),
            nn.Linear(feat_dims, 2), nn.Tanh()
        )

    def forward(self, x: torch.Tensor, add_n_points: int = 0) -> torch.Tensor:
        out = super().forward(x, add_n_points)
        return self.fold2(torch.cat([out, x.repeat(1, out.shape[1], 1)], dim=-1))


# --- Utilities ---
def collate_fn(batch: List[dict]) -> dict:
    min_points = min(b["pc"].shape[0] for b in batch)
    pc = [b["pc"] if b["pc"].shape[0] == min_points else b["pc"][torch.randperm(b["pc"].shape[0])[:min_points]] for b in
          batch]
    return {"pc": torch.stack(pc).transpose(1, 2), "ids": [b["ids"] for b in batch]}


def batch_pairwise_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xx = x.pow(2).sum(dim=2)
    yy = y.pow(2).sum(dim=2)
    xy = torch.bmm(x, y.transpose(2, 1))
    return xx[:, :, None] + yy[:, None, :] - 2 * xy


def load_dataset(args, path: str, is_train: bool) -> dataloader.Teeth_Dataset:
    unn_list = args.unn.split(",") if args.unn != "all" else "all"
    std = 9.8186 if unn_list == ["3"] else 11.75121
    return dataloader.Teeth_Dataset(unn=unn_list, folder_path=path, is_train=is_train, global_pc_std=std, args=args)


def save_latents(latents: dict, phase: str, model_dir: str) -> None:
    np.savez(os.path.join(model_dir, f"{phase}_latents.npz"), mu=latents["mu"], std=latents["std"])
    np.save(os.path.join(model_dir, f"{phase}_pe.npy"), latents["pe"])


# --- Latent Extraction ---
def extract_latents(args, model_dir: str) -> None:
    datasets = {
        "train": load_dataset(args, args.x_train, False),
        "val": load_dataset(args, args.x_val, False),
        "test": load_dataset(args, args.x_test, False)
    }
    loaders = {phase: DataLoader(ds, batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn,
                                 drop_last=True)
               for phase, ds in datasets.items()}

    vae = vfnet.Variational_autoencoder(args, num_points=datasets["train"].k).to(args.device).eval()
    vae = utils.load_pretrained_model(vae, args.model_path)

    latents = {phase: {"mu": [], "std": [], "pe": []} for phase in loaders}
    for phase, loader in loaders.items():
        for batch in tqdm(loader, desc=f"Extracting {phase} latents"):
            with torch.no_grad():
                pc = batch["pc"].to(args.device)
                feature, feat1 = vae.encoder(pc.transpose(1, 2))
                mu, lv = feature.chunk(2, dim=-1)
                z = vae.reparameterize(feature)
                grid = vae.decoder.grid_map(torch.cat([z.repeat(1, feat1.shape[1], 1), feat1], dim=-1))
                latents[phase]["mu"].append(mu.cpu().numpy())
                latents[phase]["std"].append(lv.cpu().numpy())
                latents[phase]["pe"].append(grid.cpu().numpy())

    for phase in latents:
        latents[phase] = {k: np.array(v) for k, v in latents[phase].items()}
        save_latents(latents[phase], phase, model_dir)


# --- Flow Training ---
def train_flow_prior(args, model_dir: str) -> flows.Flow:
    train_loader = DataLoader(load_dataset(args, args.x_train, True), batch_size=64, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(load_dataset(args, args.x_val, False), batch_size=64, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)

    vae = vfnet.Variational_autoencoder(args, num_points=2048).to(args.device).eval()
    vae = utils.load_pretrained_model(vae, args.model_path)
    for param in vae.parameters():
        param.requires_grad = False

    grid = torch.from_numpy(np.array(list(itertools.product(
        np.linspace(-0.9, 0.9, 75), np.linspace(-0.9, 0.9, 75)
    )))).unsqueeze(0).float()

    # Flow model
    transforms_list = []
    for _ in range(args.flow_n_layers):
        transforms_list.extend([
            transforms.MaskedAffineAutoregressiveTransform(features=512, hidden_features=512),
            transforms.ReversePermutation(features=512)
        ])
    flow = flows.Flow(transforms.CompositeTransform(transforms_list), distributions.StandardNormal([512])).to(
        args.device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=2e-4)

    for epoch in range(1, args.flow_num_epochs + 1):
        flow.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Flow Train Epoch {epoch}"):
            pc = batch["pc"].to(args.device).transpose(1, 2)
            z = vae.reparameterize(vae.encoder(pc)[0])
            loss = -flow.log_prob(z.squeeze()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch} | Train Loss: {train_loss / len(train_loader):.5f}")

        flow.eval()
        test_loss = 0
        for batch in test_loader:
            with torch.no_grad():
                pc = batch["pc"].to(args.device).transpose(1, 2)
                z = vae.reparameterize(vae.encoder(pc)[0])
                test_loss += -flow.log_prob(z.squeeze()).mean().item()
        print(f"Epoch {epoch} | Test Loss: {test_loss / len(test_loader):.4f}")

        if epoch % args.flow_viz_epochs == 0:
            plot_flow_latents(epoch, flow, model_dir)
            sample_dir = Path(model_dir) / "samples" / f"epoch_{epoch}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            samples = flow.sample(10).to(args.device)
            facets = pc_utils.facets_from_grid(75)
            for i, latent in enumerate(samples):
                recon = vae.decoder.decode(latent.unsqueeze(0).repeat(1, grid.shape[1], 1), grid.to(args.device))[
                    "reconstruction"]
                pc_utils.save_ply_manual(recon.squeeze().cpu().numpy() * 9.8186,
                                         str(sample_dir / f"flow_sample_{i}.ply"), facets)

    torch.save(flow.state_dict(), os.path.join(model_dir, "flow_prior.pth.tar"))
    return flow


def plot_flow_latents(epoch: str, flow: flows.Flow, model_dir: str) -> None:
    phases = ["train", "val", "test"]
    latents = {p: np.load(os.path.join(model_dir, f"{p}_latents.npz")) for p in phases}
    samples = {p: torch.distributions.Normal(torch.tensor(latents[p]["mu"]),
                                             torch.tensor(latents[p]["std"]).mul(0.5).exp()).sample().squeeze().numpy()
               for p in phases}

    flow_samples = np.concatenate([flow.sample(1000 if p == "train" else 500).squeeze().cpu().numpy() for _ in
                                   range(5 if p == "train" else 3)]).real
    flow_samples = flow_samples[~np.isnan(flow_samples).any(axis=1)]

    pca_dir = Path(model_dir) / "pca"
    pca_dir.mkdir(exist_ok=True)
    for phase in phases:
        pca = PCA(n_components=2).fit(samples[phase])
        plt.figure(figsize=(10, 10))
        plt.scatter(pca.transform(samples[phase])[:, 0], pca.transform(samples[phase])[:, 1], label=phase)
        plt.scatter(pca.transform(flow_samples)[:, 0], pca.transform(flow_samples)[:, 1], label="sample")
        plt.legend()
        plt.savefig(pca_dir / f"{phase}_pca_{epoch}.png")
        plt.close()

        if phase in ["val", "test"]:
            dists = metrics.pairwise_l2_distances(flow_samples, samples[phase])
            coverage = np.unique(dists.argmin(axis=1)).shape[0] / len(flow_samples)
            print(f"{phase.capitalize()} coverage: {coverage:.4f}")


# --- Point Encoding Training ---
def train_point_encoding(args, model_dir: str) -> nn.Module:
    device = args.device
    model = OneFold().to(device) if args.one_fold else TwoFold().to(device)
    chamfer = pc_utils.ChamferLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loaders = {
        "Train": DataLoader(load_dataset(args, args.x_train, True), batch_size=32, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True),
        "Test": DataLoader(load_dataset(args, args.x_val, False), batch_size=32, shuffle=True,
                           num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    }
    vae = vfnet.Variational_autoencoder(args, num_points=loaders["Train"].dataset.k).to(device).eval()
    vae = utils.load_pretrained_model(vae, args.model_path)

    sample_dir = Path(model_dir) / "pe_samples"
    sample_dir.mkdir(exist_ok=True)
    for epoch in range(1, args.pe_num_epochs + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(loaders["Train"], desc=f"PE Train Epoch {epoch}"):
            pc = batch["pc"].to(device).transpose(1, 2)
            z, feat1 = vae.encoder(pc)
            z = vae.reparameterize(z)
            pe = vae.decoder.grid_map(torch.cat([z.repeat(1, feat1.shape[1], 1), feat1], dim=-1))
            recon = model(z)
            loss = chamfer(pe, recon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch} | Train Loss: {train_loss / len(loaders['Train']):.4f}")

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in loaders["Test"]:
                pc = batch["pc"].to(device).transpose(1, 2)
                z, feat1 = vae.encoder(pc)
                z = vae.reparameterize(z)
                pe = vae.decoder.grid_map(torch.cat([z.repeat(1, feat1.shape[1], 1), feat1], dim=-1))
                recon = model(z)
                test_loss += chamfer(pe, recon).item()
            pe, recon = pe.cpu().numpy(), recon.cpu().numpy()
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1);
            plt.scatter(pe[0, :, 0], pe[0, :, 1]);
            plt.title("GT")
            plt.subplot(1, 2, 2);
            plt.scatter(recon[0, :, 0], recon[0, :, 1]);
            plt.title("Recon")
            plt.savefig(sample_dir / f"pe_{epoch}_{args.pe_model_string}.png")
            plt.close()
        print(f"Epoch {epoch} | Test Loss: {test_loss / len(loaders['Test']):.4f}")

    torch.save(model.state_dict(), os.path.join(model_dir, f"pe_{args.pe_model_string}.pth.tar"))
    return model


# --- Sampling and Evaluation ---
def get_reference(args) -> torch.Tensor:
    loader = DataLoader(load_dataset(args, args.x_test, False), batch_size=1, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    return torch.cat([batch["pc"] for batch in tqdm(loader, desc="Loading references")], dim=0).to(args.device) * (
        9.8186 if args.unn == "3" else 11.75121)


def generate_samples(args, num_samples: int, flow: flows.Flow, pe: nn.Module) -> torch.Tensor:
    vae = vfnet.Variational_autoencoder(args, num_points=2048).to(args.device).eval()
    vae = utils.load_pretrained_model(vae, args.model_path)
    std = 9.8186 if args.unn == "3" else 11.75121
    samples = []
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        with torch.no_grad():
            z = flow.sample(1)
            point_encoding = pe(z, add_n_points=23)
            sample = vae.decoder.decode(z.repeat(1, point_encoding.shape[1], 1), point_encoding)["reconstruction"]
            samples.append(sample)
    return torch.cat(samples, dim=0) * std


# --- Main ---
if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.dirname(args.model_path)
    torch.manual_seed(args.seed);
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load or update args from file
    with open(os.path.join(model_dir, "commandline_input.json"), "r") as f:
        for k, v in json.load(f).items():
            if k not in vars(args):
                setattr(args, k, v)

    # Setup test directory
    test_dir = Path("test")
    test_dir.mkdir(exist_ok=True)
    shutil.copy(__file__, test_dir / f"{args.test_name}.py")

    # Latents
    latent_files = ["train_latents.npz", "val_latents.npz", "test_latents.npz", "train_pe.npy", "val_pe.npy",
                    "test_pe.npy"]
    if not all(os.path.exists(os.path.join(model_dir, f)) for f in latent_files):
        print("Generating latents...")
        extract_latents(args, model_dir)

    # Flow prior
    flow_path = os.path.join(model_dir, "flow_prior.pth.tar")
    if not os.path.exists(flow_path):
        print("Training flow prior...")
        flow = train_flow_prior(args, model_dir)
    else:
        flow = flows.Flow(
            transforms.CompositeTransform(
                [transforms.MaskedAffineAutoregressiveTransform(features=512, hidden_features=512),
                 transforms.ReversePermutation(features=512)] * args.flow_n_layers),
            distributions.StandardNormal([512])
        ).to(args.device)
        flow.load_state_dict(torch.load(flow_path))
        flow.eval()
        plot_flow_latents("final", flow, model_dir)
        print("Loaded existing flow prior")

    # Point encoding
    args.pe_model_string = "oneFold" if args.one_fold else "twoFold"
    pe_path = os.path.join(model_dir, f"pe_{args.pe_model_string}.pth.tar")
    pe = OneFold().to(args.device) if args.one_fold else TwoFold().to(args.device)
    if not os.path.exists(pe_path):
        print("Training point encoding predictor...")
        pe = train_point_encoding(args, model_dir)
    else:
        pe.load_state_dict(torch.load(pe_path))
        print("Loaded existing point encoding predictor")

    # Generate and evaluate
    ref = get_reference(args).transpose(1, 2)
    samples = generate_samples(args, ref.shape[0], flow, pe).transpose(1, 2)
    results = metrics.compute_all_metrics(samples, ref, batch_size=256, accelerated_cd=True)
    print(f"\n{args.test_name} Results:")
    for k, v in results.items():
        print(f"{k}: {v * 100:.4f}")
    results["args"] = vars(args)
    torch.save(results, os.path.join(model_dir, "results.pt"))