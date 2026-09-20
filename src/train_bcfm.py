"""Train Balanced Conditional Flow Matching on MNIST."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image

from train_flows import ResidualBlock, groups, sinusoidal_embedding, update_ema


@dataclass
class Config:
    data_dir: str
    output_dir: str
    epochs: int = 80
    batch_size: int = 256
    lr: float = 6e-4
    seed: int = 5489
    num_workers: int = 4
    train_limit: int | None = None
    base_channels: int = 64
    ema_decay: float = 0.999
    time_scale: float = 100.0
    sample_count: int = 100


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


class BalancedConditionalFlow(nn.Module):
    """Class-conditional velocity field used by BCFM."""

    def __init__(self, base: int = 64, time_scale: float = 100.0):
        super().__init__()
        condition_dim = base * 4
        self.base = base
        self.time_scale = time_scale
        self.time_mlp = nn.Sequential(
            nn.Linear(128, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.class_embedding = nn.Embedding(10, condition_dim)
        nn.init.normal_(self.class_embedding.weight, std=0.02)
        self.input = nn.Conv2d(1, base, 3, padding=1)
        self.enc1 = ResidualBlock(base, base, condition_dim)
        self.down1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.enc2 = ResidualBlock(base * 2, base * 2, condition_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)
        self.mid1 = ResidualBlock(base * 4, base * 4, condition_dim)
        self.mid2 = ResidualBlock(base * 4, base * 4, condition_dim)
        self.up2 = nn.Conv2d(base * 4, base * 2, 3, padding=1)
        self.dec2 = ResidualBlock(base * 4, base * 2, condition_dim)
        self.up1 = nn.Conv2d(base * 2, base, 3, padding=1)
        self.dec1 = ResidualBlock(base * 2, base, condition_dim)
        self.output_norm = nn.GroupNorm(groups(base), base)
        self.output = nn.Conv2d(base, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        zeros = torch.zeros_like(t)
        condition = self.time_mlp(
            torch.cat(
                [
                    sinusoidal_embedding(t, scale=self.time_scale),
                    sinusoidal_embedding(zeros, scale=self.time_scale),
                ],
                dim=1,
            )
        ) + self.class_embedding(labels)
        h1 = self.enc1(self.input(x), condition)
        h2 = self.enc2(self.down1(h1), condition)
        h = self.mid2(self.mid1(self.down2(h2), condition), condition)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up2(h)
        h = self.dec2(torch.cat([h, h2], dim=1), condition)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up1(h)
        h = self.dec1(torch.cat([h, h1], dim=1), condition)
        return self.output(F.silu(self.output_norm(h)))


def make_loaders(cfg: Config):
    data = datasets.MNIST(
        cfg.data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    train, val = random_split(data, [55_000, 5_000], generator=generator)
    if cfg.train_limit is not None:
        train = Subset(
            train,
            torch.randperm(len(train), generator=generator)[: cfg.train_limit].tolist(),
        )

    def collate(batch):
        return torch.stack([item[0] for item in batch]).mul(2).sub(1), torch.tensor(
            [item[1] for item in batch]
        )

    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=collate,
    )
    return (
        DataLoader(train, shuffle=True, drop_last=True, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
    )


def flow_loss(
    model: BalancedConditionalFlow, x: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Return the conditional flow-matching regression loss."""
    batch = x.size(0)
    noise = torch.randn_like(x)
    t = torch.rand(batch, device=x.device)
    z = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * noise
    return F.mse_loss(model(z, t, labels), noise - x)


@torch.inference_mode()
def sample(
    model: BalancedConditionalFlow,
    count: int,
    device: torch.device,
    steps: int = 25,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a class-balanced batch with explicit Euler integration."""
    if labels is None:
        labels = torch.arange(count, device=device) % 10
    else:
        labels = labels.to(device)

    z = torch.randn(count, 1, 28, 28, device=device)
    times = torch.linspace(1, 0, steps + 1, device=device)
    for t, r in zip(times[:-1], times[1:]):
        velocity = model(z, t.expand(count), labels)
        z = z + (r - t) * velocity
    return z.add(1).div(2).clamp(0, 1)


def validation_loss(
    model: BalancedConditionalFlow,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    batches: int = 10,
) -> float:
    model.eval()
    values = []
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(cfg.seed + 10_000)
    torch.cuda.manual_seed(cfg.seed + 10_000)
    for index, (x, labels) in enumerate(loader):
        if index >= batches:
            break
        values.append(flow_loss(model, x.to(device), labels.to(device)).item())
    torch.random.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, device)
    return float(np.mean(values))


def train(cfg: Config) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda")
    train_loader, val_loader = make_loaders(cfg)
    model = BalancedConditionalFlow(cfg.base_channels, cfg.time_scale).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0
    )
    warmup = min(1000, max(1, len(train_loader) * 5))
    total = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup, 1.0)
        * (
            0.5
            + 0.5 * math.cos(math.pi * max(0, step - warmup) / max(1, total - warmup))
        ),
    )
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    history = []
    best = float("inf")
    start = time.perf_counter()
    sample_epochs = sorted(
        set([1, cfg.epochs // 4, cfg.epochs // 2, 3 * cfg.epochs // 4, cfg.epochs])
    )
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_sum = 0.0
        epoch_start = time.perf_counter()
        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = flow_loss(model, x, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            update_ema(ema, model, cfg.ema_decay)
            loss_sum += loss.item()
        val = validation_loss(ema, val_loader, cfg, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader),
            "val_loss": val,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        state = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
            "val_loss": val,
        }
        torch.save(state, output / "latest.pt")
        if val < best:
            best = val
            torch.save(state, output / "best.pt")
        if epoch in sample_epochs:
            save_image(
                sample(ema, cfg.sample_count, device, 25),
                output / "samples" / f"epoch_{epoch:03d}.png",
                nrow=10,
            )
    elapsed = time.perf_counter() - start
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    ema.load_state_dict(checkpoint["ema"])
    ema.eval()
    generation = {}
    for steps in [10, 25, 50]:
        tic = time.perf_counter()
        images = sample(ema, cfg.sample_count, device, steps)
        seconds = time.perf_counter() - tic
        generation[str(steps)] = {
            "seconds": seconds,
            "milliseconds_per_image": seconds * 1000 / cfg.sample_count,
        }
        save_image(images, output / f"samples_{steps:03d}_nfe.png", nrow=10)
    metrics = {
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "training_seconds": elapsed,
        "parameters": sum(p.numel() for p in ema.parameters()),
        "generation": generation,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "history.json").write_text(json.dumps(history, indent=2))
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(
        [r["epoch"] for r in history], [r["train_loss"] for r in history], label="train"
    )
    ax.plot(
        [r["epoch"] for r in history],
        [r["val_loss"] for r in history],
        label="validation",
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Flow-matching MSE",
        title="Balanced Conditional Flow Matching",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--sample-count", type=int, default=100)
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    train(parse_args())
