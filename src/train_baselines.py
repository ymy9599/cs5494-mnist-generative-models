"""Train the required FVSBN, MADE, and DCGAN baselines on MNIST."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


@dataclass
class Config:
    model: str
    data_dir: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    num_workers: int
    train_limit: int | None
    sample_count: int
    hidden: int = 512
    layers: int = 2
    latent_dim: int = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def loader_kwargs(cfg: Config) -> dict:
    return {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": True,
        "persistent_workers": cfg.num_workers > 0,
    }


def get_loaders(cfg: Config, binary: bool):
    transform = transforms.ToTensor()
    full = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=transform)
    gen = torch.Generator().manual_seed(cfg.seed)
    train, val = random_split(full, [55_000, 5_000], generator=gen)
    if cfg.train_limit is not None:
        if cfg.train_limit > len(train):
            raise ValueError("train_limit exceeds training split")
        indices = torch.randperm(len(train), generator=gen)[: cfg.train_limit].tolist()
        train = Subset(train, indices)

    def collate(batch):
        x = torch.stack([item[0] for item in batch])
        y = torch.tensor([item[1] for item in batch], dtype=torch.long)
        if binary:
            x = (x >= 0.5).float()
        return x, y

    kw = loader_kwargs(cfg)
    return (
        DataLoader(train, shuffle=True, collate_fn=collate, **kw),
        DataLoader(val, shuffle=False, collate_fn=collate, **kw),
        DataLoader(test, shuffle=False, collate_fn=collate, **kw),
    )


class FVSBN(nn.Module):
    """p(x)=prod_i Bernoulli(x_i; sigmoid(w_i^T x_<i + b_i))."""

    def __init__(self, dim: int = 784):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("mask", torch.tril(torch.ones(dim, dim), diagonal=-1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        x = torch.zeros(n, self.dim, device=device)
        for i in range(self.dim):
            logits = self.bias[i].expand(n)
            if i:
                logits = logits + x[:, :i] @ self.weight[i, :i]
            x[:, i] = torch.bernoulli(torch.sigmoid(logits))
        return x.view(n, 1, 28, 28)


class MaskedLinear(nn.Linear):
    """Linear layer whose connectivity is fixed by an autoregressive mask."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: np.ndarray) -> None:
        self.mask.copy_(torch.from_numpy(mask.astype(np.float32)).t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """Masked autoencoder implementing an exact Bernoulli autoregressive model."""

    def __init__(
        self, dim: int = 784, hidden: int = 512, layers: int = 2, seed: int = 5489
    ):
        super().__init__()
        self.dim = dim
        sizes = [dim] + [hidden] * layers + [dim]
        modules: list[nn.Module] = []
        self.masked_layers: list[MaskedLinear] = []
        for index, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = MaskedLinear(in_size, out_size)
            modules.append(layer)
            self.masked_layers.append(layer)
            if index < len(sizes) - 2:
                modules.append(nn.ReLU())
        self.net = nn.Sequential(*modules)
        self._create_masks(hidden, layers, seed)

    def _create_masks(self, hidden: int, layers: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        degrees = [np.arange(1, self.dim + 1)]
        for _ in range(layers):
            degrees.append(rng.integers(1, self.dim, size=hidden))
        degrees.append(np.arange(1, self.dim + 1))
        for index, layer in enumerate(self.masked_layers):
            if index == len(self.masked_layers) - 1:
                mask = degrees[index][:, None] < degrees[index + 1][None, :]
            else:
                mask = degrees[index][:, None] <= degrees[index + 1][None, :]
            layer.set_mask(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        x = torch.zeros(n, self.dim, device=device)
        for i in range(self.dim):
            probability = torch.sigmoid(self(x)[:, i])
            x[:, i] = torch.bernoulli(probability)
        return x.view(n, 1, 28, 28)


class Generator(nn.Module):
    """DCGAN generator for 28 x 28 grayscale images."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7, bias=False),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.project(z).view(-1, 256, 7, 7))


class Discriminator(nn.Module):
    """DCGAN discriminator returning one real/fake logit per image."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1)).squeeze(1)


def save_grid(
    samples: torch.Tensor,
    path: Path,
    nrow: int = 8,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        samples.detach().cpu(), path, nrow=nrow, normalize=True, value_range=value_range
    )


def save_history(history: list[dict], output: Path, title: str) -> None:
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (output / "history.json").write_text(json.dumps(history, indent=2))

    keys = [k for k in history[0] if k not in {"epoch", "seconds"}]
    fig, axes = plt.subplots(len(keys), 1, figsize=(6.8, 2.5 * len(keys)), sharex=True)
    axes = np.atleast_1d(axes)
    epochs = [row["epoch"] for row in history]
    for ax, key in zip(axes, keys):
        ax.plot(epochs, [row[key] for row in history], marker="o", linewidth=2)
        ax.set_ylabel(key)
        ax.grid(alpha=0.25)
    axes[0].set_title(title, loc="left", fontweight="bold")
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)


def evaluate_nll(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.inference_mode():
        for x, _ in loader:
            flat = x.to(device, non_blocking=True).flatten(1)
            loss = F.binary_cross_entropy_with_logits(
                model(flat), flat, reduction="sum"
            )
            total_loss += loss.item()
            count += flat.size(0)
    nll = total_loss / count
    return nll, nll / (784 * math.log(2))


def train_autoregressive(cfg: Config, device: torch.device) -> None:
    train_loader, val_loader, test_loader = get_loaders(cfg, binary=True)
    if cfg.model == "fvsbn":
        model: nn.Module = FVSBN()
    else:
        model = MADE(hidden=cfg.hidden, layers=cfg.layers, seed=cfg.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    start = time.perf_counter()
    sample_epochs = sorted(
        set([1, cfg.epochs // 4, cfg.epochs // 2, 3 * cfg.epochs // 4, cfg.epochs])
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        count = 0
        epoch_start = time.perf_counter()
        for x, _ in train_loader:
            flat = x.to(device, non_blocking=True).flatten(1)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(flat), flat, reduction="sum"
            ) / flat.size(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * flat.size(0)
            count += flat.size(0)
        val_nll, val_bpd = evaluate_nll(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_nll": train_loss / count,
            "val_nll": val_nll,
            "val_bpd": val_bpd,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_nll < best:
            best = val_nll
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_nll": val_nll,
                },
                output / "best.pt",
            )
        if epoch in sample_epochs:
            samples = model.sample(cfg.sample_count, device)
            save_grid(samples, output / "samples" / f"epoch_{epoch:03d}.png")

    fit_seconds = time.perf_counter() - start
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_nll, test_bpd = evaluate_nll(model, test_loader, device)
    inference_start = time.perf_counter()
    samples = model.sample(cfg.sample_count, device)
    inference_seconds = time.perf_counter() - inference_start
    save_grid(samples, output / "samples_final.png")
    metrics = {
        "best_epoch": checkpoint["epoch"],
        "best_val_nll": checkpoint["val_nll"],
        "test_nll": test_nll,
        "test_bpd": test_bpd,
        "training_seconds": fit_seconds,
        "sample_count": cfg.sample_count,
        "inference_seconds": inference_seconds,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_history(history, output, f"{cfg.model.upper()} training")
    print(json.dumps(metrics, indent=2), flush=True)


def weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def train_gan(cfg: Config, device: torch.device) -> None:
    train_loader, _, _ = get_loaders(cfg, binary=False)
    generator = Generator(cfg.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fixed_z = torch.randn(cfg.sample_count, cfg.latent_dim, device=device)
    history = []
    start = time.perf_counter()
    sample_epochs = sorted(
        set([1, cfg.epochs // 4, cfg.epochs // 2, 3 * cfg.epochs // 4, cfg.epochs])
    )

    for epoch in range(1, cfg.epochs + 1):
        generator.train()
        discriminator.train()
        d_sum = g_sum = 0.0
        steps = 0
        epoch_start = time.perf_counter()
        for real, _ in train_loader:
            real = real.to(device, non_blocking=True).mul(2).sub(1)
            batch = real.size(0)
            real_target = torch.full((batch,), 0.9, device=device)
            fake_target = torch.zeros(batch, device=device)

            opt_d.zero_grad(set_to_none=True)
            d_real = discriminator(real)
            with torch.no_grad():
                fake_detached = generator(
                    torch.randn(batch, cfg.latent_dim, device=device)
                )
            d_fake = discriminator(fake_detached)
            d_loss = criterion(d_real, real_target) + criterion(d_fake, fake_target)
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake = generator(torch.randn(batch, cfg.latent_dim, device=device))
            g_loss = criterion(discriminator(fake), torch.ones(batch, device=device))
            g_loss.backward()
            opt_g.step()
            d_sum += d_loss.item()
            g_sum += g_loss.item()
            steps += 1

        row = {
            "epoch": epoch,
            "d_loss": d_sum / steps,
            "g_loss": g_sum / steps,
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if epoch in sample_epochs:
            generator.eval()
            with torch.inference_mode():
                samples = generator(fixed_z)
            save_grid(
                samples,
                output / "samples" / f"epoch_{epoch:03d}.png",
                value_range=(-1, 1),
            )
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch,
            },
            output / "latest.pt",
        )

    fit_seconds = time.perf_counter() - start
    generator.eval()
    inference_start = time.perf_counter()
    with torch.inference_mode():
        samples = generator(fixed_z)
    inference_seconds = time.perf_counter() - inference_start
    save_grid(samples, output / "samples_final.png", value_range=(-1, 1))
    metrics = {
        "training_seconds": fit_seconds,
        "sample_count": cfg.sample_count,
        "inference_seconds": inference_seconds,
        "generator_parameters": sum(p.numel() for p in generator.parameters()),
        "discriminator_parameters": sum(p.numel() for p in discriminator.parameters()),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_history(history, output, "DCGAN training")
    print(json.dumps(metrics, indent=2), flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fvsbn", "made", "gan"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=128)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official experiment run")
    device = torch.device("cuda")
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(
        json.dumps({"device": torch.cuda.get_device_name(0), **asdict(cfg)}, indent=2),
        flush=True,
    )
    if cfg.model in {"fvsbn", "made"}:
        train_autoregressive(cfg, device)
    else:
        train_gan(cfg, device)


if __name__ == "__main__":
    main()
