"""Train standard Flow Matching or Improved MeanFlow on MNIST."""

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
from torch.func import jvp
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image


@dataclass
class Config:
    method: str
    data_dir: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    num_workers: int
    train_limit: int | None
    base_channels: int
    ema_decay: float
    nonzero_interval_ratio: float
    time_mu: float
    time_sigma: float
    adaptive_power: float
    sample_count: int
    time_scale: float
    auxiliary_head: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def sinusoidal_embedding(
    t: torch.Tensor, dim: int = 64, max_period: int = 10_000, scale: float = 1000.0
) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device) / half
    )
    angles = t[:, None] * frequencies[None] * scale
    return torch.cat([angles.cos(), angles.sin()], dim=1)


def groups(channels: int) -> int:
    return min(32, max(1, channels // 4))


class ResidualBlock(nn.Module):
    """FiLM-conditioned residual block used throughout the velocity U-Net."""

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.condition = nn.Sequential(
            nn.SiLU(), nn.Linear(condition_dim, out_channels * 2)
        )
        self.norm2 = nn.GroupNorm(groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class VelocityUNet(nn.Module):
    """Pixel-space U-Net conditioned on (t, t-r), shared by FM and iMF."""

    def __init__(
        self, base: int = 64, time_scale: float = 1000.0, auxiliary_head: bool = False
    ):
        super().__init__()
        condition_dim = base * 4
        self.time_scale = time_scale
        self.auxiliary_head = auxiliary_head
        self.time_mlp = nn.Sequential(
            nn.Linear(128, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
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
        if auxiliary_head:
            self.velocity_output = nn.Conv2d(base, 1, 3, padding=1)
            nn.init.zeros_(self.velocity_output.weight)
            nn.init.zeros_(self.velocity_output.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        interval: torch.Tensor,
        head: str = "average",
    ) -> torch.Tensor:
        condition = self.time_mlp(
            torch.cat(
                [
                    sinusoidal_embedding(t, scale=self.time_scale),
                    sinusoidal_embedding(interval, scale=self.time_scale),
                ],
                dim=1,
            )
        )
        h1 = self.enc1(self.input(x), condition)
        h2 = self.enc2(self.down1(h1), condition)
        h = self.mid2(self.mid1(self.down2(h2), condition), condition)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up2(h)
        h = self.dec2(torch.cat([h, h2], dim=1), condition)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up1(h)
        h = self.dec1(torch.cat([h, h1], dim=1), condition)
        h = F.silu(self.output_norm(h))
        if head == "velocity":
            if not self.auxiliary_head:
                raise RuntimeError("velocity head is disabled")
            return self.velocity_output(h)
        return self.output(h)


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
        x = torch.stack([item[0] for item in batch]).mul(2).sub(1)
        return x

    kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        collate_fn=collate,
    )
    return DataLoader(train, shuffle=True, drop_last=True, **kwargs), DataLoader(
        val, shuffle=False, **kwargs
    )


def sample_times(
    batch: int, cfg: Config, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    pair = torch.sigmoid(
        torch.randn(batch, 2, device=device) * cfg.time_sigma + cfg.time_mu
    )
    r, t = pair.min(dim=1).values, pair.max(dim=1).values
    keep_interval = torch.rand(batch, device=device) < cfg.nonzero_interval_ratio
    r = torch.where(keep_interval, r, t)
    return r, t


def adaptive_loss(
    error: torch.Tensor, power: float, constant: float = 1e-3
) -> torch.Tensor:
    per_sample = error.square().flatten(1).mean(1)
    weight = (per_sample.detach() + constant).pow(-power)
    return (weight * per_sample).mean()


def prediction_and_loss(
    model: VelocityUNet, x: torch.Tensor, cfg: Config
) -> tuple[torch.Tensor, dict]:
    batch = x.size(0)
    noise = torch.randn_like(x)
    if cfg.method == "flow":
        t = torch.rand(batch, device=x.device)
        interval = torch.zeros_like(t)
        z = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * noise
        prediction = model(z, t, interval)
        target = noise - x
        error = prediction - target
        return adaptive_loss(error, 0.0), {
            "nonzero_loss": float("nan"),
            "instant_loss": error.square().mean().item(),
            "auxiliary_loss": float("nan"),
        }

    r, t = sample_times(batch, cfg, x.device)
    z = (1 - t[:, None, None, None]) * x + t[:, None, None, None] * noise
    zeros = torch.zeros_like(t)
    ones = torch.ones_like(t)
    if cfg.auxiliary_head:
        velocity = model(z, t, zeros, head="velocity")
        auxiliary_error = velocity - (noise - x)
        auxiliary_loss = adaptive_loss(auxiliary_error, cfg.adaptive_power)
    else:
        velocity = model(z, t, zeros)
        auxiliary_loss = torch.zeros((), device=x.device)

    def average_velocity(z_arg, r_arg, t_arg):
        return model(z_arg, t_arg, t_arg - r_arg)

    # The JVP enforces the MeanFlow identity over the sampled interval.
    average, total_derivative = jvp(
        average_velocity, (z, r, t), (velocity, zeros, ones)
    )
    compound_velocity = (
        average + (t - r)[:, None, None, None] * total_derivative.detach()
    )
    error = compound_velocity - (noise - x)
    loss = adaptive_loss(error, cfg.adaptive_power) + auxiliary_loss
    nonzero = r != t
    return loss, {
        "nonzero_loss": (
            error[nonzero].square().mean().item() if nonzero.any() else float("nan")
        ),
        "instant_loss": (
            error[~nonzero].square().mean().item() if (~nonzero).any() else float("nan")
        ),
        "auxiliary_loss": (
            auxiliary_error.square().mean().item()
            if cfg.auxiliary_head
            else float("nan")
        ),
    }


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter, 1 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


@torch.inference_mode()
def sample(
    model: VelocityUNet, count: int, device: torch.device, method: str, steps: int = 1
) -> torch.Tensor:
    z = torch.randn(count, 1, 28, 28, device=device)
    if method == "flow":
        times = torch.linspace(1, 0, steps + 1, device=device)
        for t, r in zip(times[:-1], times[1:]):
            t_batch = t.expand(count)
            interval = torch.zeros_like(t_batch)
            z = z + (r - t) * model(z, t_batch, interval)
    else:
        times = torch.linspace(1, 0, steps + 1, device=device)
        for t, r in zip(times[:-1], times[1:]):
            t_batch = t.expand(count)
            interval = (t - r).expand(count)
            z = z - (t - r) * model(z, t_batch, interval)
    return z.add(1).div(2).clamp(0, 1)


def validation_loss(
    model: VelocityUNet,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    batches: int = 10,
) -> float:
    model.eval()
    values = []
    # Keep the validation Monte Carlo draw fixed across epochs.
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    torch.manual_seed(cfg.seed + 10_000)
    torch.cuda.manual_seed(cfg.seed + 10_000)
    for index, x in enumerate(loader):
        if index >= batches:
            break
        loss, _ = prediction_and_loss(model, x.to(device, non_blocking=True), cfg)
        values.append(loss.detach().item())
    torch.random.set_rng_state(state)
    torch.cuda.set_rng_state(cuda_state, device)
    return float(np.mean(values))


def train(cfg: Config) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda")
    train_loader, val_loader = make_loaders(cfg)
    model = VelocityUNet(cfg.base_channels, cfg.time_scale, cfg.auxiliary_head).to(
        device
    )
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0
    )
    warmup_steps = min(1000, max(1, len(train_loader) * 5))
    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, 1.0)
        * (
            0.5
            + 0.5
            * math.cos(
                math.pi
                * max(0, step - warmup_steps)
                / max(1, total_steps - warmup_steps)
            )
        ),
    )
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    history = []
    best = float("inf")
    start = time.perf_counter()
    step = 0
    sample_epochs = sorted(
        set([1, cfg.epochs // 4, cfg.epochs // 2, 3 * cfg.epochs // 4, cfg.epochs])
    )
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sum_loss = sum_nonzero = sum_instant = sum_auxiliary = 0.0
        count_nonzero = count_instant = count_auxiliary = 0
        epoch_start = time.perf_counter()
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, details = prediction_and_loss(model, x, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            update_ema(ema, model, cfg.ema_decay)
            step += 1
            sum_loss += loss.item()
            if math.isfinite(details["nonzero_loss"]):
                sum_nonzero += details["nonzero_loss"]
                count_nonzero += 1
            if math.isfinite(details["instant_loss"]):
                sum_instant += details["instant_loss"]
                count_instant += 1
            if math.isfinite(details["auxiliary_loss"]):
                sum_auxiliary += details["auxiliary_loss"]
                count_auxiliary += 1
        val = validation_loss(ema, val_loader, cfg, device)
        row = {
            "epoch": epoch,
            "train_loss": sum_loss / len(train_loader),
            "val_loss": val,
            "nonzero_mse": sum_nonzero / max(1, count_nonzero),
            "instant_mse": sum_instant / max(1, count_instant),
            "auxiliary_mse": sum_auxiliary / max(1, count_auxiliary),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val < best:
            best = val
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "config": asdict(cfg),
                    "epoch": epoch,
                    "val_loss": val,
                },
                output / "best.pt",
            )
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "config": asdict(cfg),
                "epoch": epoch,
                "val_loss": val,
            },
            output / "latest.pt",
        )
        if epoch in sample_epochs:
            generated = sample(
                ema,
                cfg.sample_count,
                device,
                cfg.method,
                50 if cfg.method == "flow" else 1,
            )
            save_image(generated, output / "samples" / f"epoch_{epoch:03d}.png", nrow=8)

    fit_seconds = time.perf_counter() - start
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    ema.load_state_dict(checkpoint["ema"])
    ema.eval()
    generation = {}
    for steps in [1, 2, 4] if cfg.method == "imf" else [10, 25, 50]:
        sample_start = time.perf_counter()
        generated = sample(ema, cfg.sample_count, device, cfg.method, steps)
        seconds = time.perf_counter() - sample_start
        generation[str(steps)] = {
            "seconds": seconds,
            "milliseconds_per_image": seconds * 1000 / cfg.sample_count,
        }
        save_image(generated, output / f"samples_{steps:03d}_nfe.png", nrow=8)
    metrics = {
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "training_seconds": fit_seconds,
        "parameters": sum(p.numel() for p in ema.parameters()),
        "generation": generation,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (output / "history.json").write_text(json.dumps(history, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(
        [r["epoch"] for r in history], [r["train_loss"] for r in history], label="train"
    )
    axes[0].plot(
        [r["epoch"] for r in history],
        [r["val_loss"] for r in history],
        label="validation",
    )
    axes[0].set(
        xlabel="Epoch", ylabel="Adaptive loss", title=f"{cfg.method.upper()} objective"
    )
    axes[0].legend()
    axes[1].plot(
        [r["epoch"] for r in history], [r["instant_mse"] for r in history], label="r=t"
    )
    if cfg.method == "imf":
        axes[1].plot(
            [r["epoch"] for r in history],
            [r["nonzero_mse"] for r in history],
            label="r<t",
        )
    axes[1].set(xlabel="Epoch", ylabel="Unweighted MSE", title="Interval diagnostics")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2), flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["flow", "imf"], required=True)
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
    parser.add_argument("--nonzero-interval-ratio", type=float, default=0.5)
    parser.add_argument("--time-mu", type=float, default=-0.4)
    parser.add_argument("--time-sigma", type=float, default=1.0)
    parser.add_argument("--adaptive-power", type=float, default=0.75)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--time-scale", type=float, default=1000.0)
    parser.add_argument("--auxiliary-head", action="store_true")
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    train(parse_args())
