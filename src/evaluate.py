"""Train the frozen MNIST evaluator and evaluate the baseline generators."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baselines import Discriminator, FVSBN, Generator, MADE  # noqa: E402
from train_flows import VelocityUNet, sample as sample_flow_model  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


class MNISTClassifier(nn.Module):
    """Compact evaluator with an explicit 128-D perceptual feature layer."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.features(x).flatten(1)
        logits = self.head(features)
        return (logits, features) if return_features else logits


def loaders(data_dir: str, batch_size: int):
    dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transforms.ToTensor()
    )
    test = datasets.MNIST(
        data_dir, train=False, download=True, transform=transforms.ToTensor()
    )
    train, val = random_split(
        dataset, [55_000, 5_000], generator=torch.Generator().manual_seed(5489)
    )
    kwargs = dict(
        batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True
    )
    return (
        DataLoader(train, shuffle=True, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
    )


@torch.inference_mode()
def classifier_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device, non_blocking=True)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def train_classifier(args, device: torch.device) -> None:
    train, val, test = loaders(args.data_dir, args.batch_size)
    model = MNISTClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = total = 0
        for x, y in train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.02)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        val_acc = classifier_accuracy(model, val, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_acc": correct / total,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        scheduler.step()
        if val_acc > best:
            best = val_acc
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc},
                output / "classifier.pt",
            )
    checkpoint = torch.load(
        output / "classifier.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    metrics = {
        "best_epoch": checkpoint["epoch"],
        "validation_accuracy": checkpoint["val_acc"],
        "test_accuracy": classifier_accuracy(model, test, device),
        "training_seconds": time.perf_counter() - start,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


def load_generator(kind: str, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    if kind == "fvsbn":
        model = FVSBN()
    elif kind == "made":
        model = MADE(hidden=cfg["hidden"], layers=cfg["layers"], seed=cfg["seed"])
    elif kind == "gan":
        model = Generator(cfg["latent_dim"])
    elif kind in {"flow", "imf"}:
        model = VelocityUNet(
            cfg["base_channels"],
            cfg.get("time_scale", 1000.0),
            cfg.get("auxiliary_head", False),
        )
    else:
        raise ValueError(kind)
    key = (
        "generator"
        if kind == "gan"
        else ("ema" if kind in {"flow", "imf"} else "model")
    )
    model.load_state_dict(checkpoint[key])
    model.to(device).eval()
    return model, cfg


@torch.inference_mode()
def generate(
    kind: str,
    model: nn.Module,
    cfg: dict,
    count: int,
    batch: int,
    device: torch.device,
    steps: int,
) -> torch.Tensor:
    result = []
    for start in range(0, count, batch):
        n = min(batch, count - start)
        if kind == "gan":
            x = (
                model(torch.randn(n, cfg["latent_dim"], device=device))
                .add(1)
                .div(2)
                .clamp(0, 1)
            )
        elif kind in {"flow", "imf"}:
            x = sample_flow_model(model, n, device, kind, steps)
        else:
            x = model.sample(n, device)
        result.append(x.cpu())
    return torch.cat(result)


@torch.inference_mode()
def extract(
    model: MNISTClassifier, images: torch.Tensor, batch: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    probs, features = [], []
    for x in images.split(batch):
        logits, f = model(x.to(device), return_features=True)
        probs.append(logits.softmax(1).cpu())
        features.append(f.cpu())
    return torch.cat(probs), torch.cat(features)


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Frechet distance between two feature distributions."""
    mu_x, mu_y = x.mean(0), y.mean(0)
    cov_x, cov_y = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    product_root = sqrtm(cov_x @ cov_y)
    if np.iscomplexobj(product_root):
        product_root = product_root.real
    return float(
        np.sum((mu_x - mu_y) ** 2) + np.trace(cov_x + cov_y - 2 * product_root)
    )


def manifold_radii(x: torch.Tensor, k: int = 3, chunk: int = 256) -> torch.Tensor:
    """Return each feature vector's k-nearest-neighbor radius."""
    radii = []
    for q in x.split(chunk):
        distance = torch.cdist(q, x)
        radii.append(distance.kthvalue(k + 1, dim=1).values)
    return torch.cat(radii)


def fraction_in_manifold(
    query: torch.Tensor, reference: torch.Tensor, radii: torch.Tensor, chunk: int = 256
) -> float:
    hits = []
    for q in query.split(chunk):
        distance = torch.cdist(q, reference)
        hits.append((distance <= radii.unsqueeze(0)).any(1))
    return torch.cat(hits).float().mean().item()


def evaluate(args, device: torch.device) -> None:
    classifier_checkpoint = torch.load(
        args.classifier, map_location=device, weights_only=False
    )
    classifier = MNISTClassifier().to(device)
    classifier.load_state_dict(classifier_checkpoint["model"])
    classifier.eval()
    generator, cfg = load_generator(args.kind, args.checkpoint, device)
    # Exclude one-off CUDA/cuDNN initialization from the reported generation
    # runtime. Resetting the seed after warm-up preserves the exact evaluation
    # samples used by the un-warmed benchmark.
    warmup_count = min(args.generation_batch, args.samples)
    generate(args.kind, generator, cfg, warmup_count, warmup_count, device, args.steps)
    seed_everything(5489)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    samples = generate(
        args.kind,
        generator,
        cfg,
        args.samples,
        args.generation_batch,
        device,
        args.steps,
    )
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - start
    _, _, test_loader = loaders(args.data_dir, args.feature_batch)
    real = torch.cat([x for x, _ in test_loader])[: args.samples]
    fake_probs, fake_features = extract(classifier, samples, args.feature_batch, device)
    _, real_features = extract(classifier, real, args.feature_batch, device)
    marginal = fake_probs.mean(0)
    kl = (fake_probs * ((fake_probs + 1e-8).log() - (marginal + 1e-8).log())).sum(1)
    label_counts = fake_probs.argmax(1).bincount(minlength=10).float()
    class_distribution = label_counts / label_counts.sum()
    normalized_entropy = float(
        -(class_distribution * (class_distribution + 1e-8).log()).sum() / math.log(10)
    )
    n_pr = min(args.pr_samples, args.samples)
    real_pr = F.normalize(real_features[:n_pr].float(), dim=1)
    fake_pr = F.normalize(fake_features[:n_pr].float(), dim=1)
    real_radii = manifold_radii(real_pr)
    fake_radii = manifold_radii(fake_pr)
    metrics = {
        "kind": args.kind,
        "nfe": args.steps if args.kind in {"flow", "imf"} else None,
        "samples": args.samples,
        "generation_seconds": generation_seconds,
        "milliseconds_per_image": generation_seconds * 1000 / args.samples,
        "mnist_feature_distance": frechet_distance(
            real_features.numpy(), fake_features.numpy()
        ),
        "precision": fraction_in_manifold(fake_pr, real_pr, real_radii),
        "recall": fraction_in_manifold(real_pr, fake_pr, fake_radii),
        "mean_classifier_confidence": fake_probs.max(1).values.mean().item(),
        "high_confidence_fraction": (fake_probs.max(1).values >= 0.9)
        .float()
        .mean()
        .item(),
        "mnist_classifier_score": kl.mean().exp().item(),
        "normalized_class_entropy": normalized_entropy,
        "classes_over_one_percent": int((class_distribution >= 0.01).sum()),
        "class_distribution": class_distribution.tolist(),
        "evaluator_validation_accuracy": classifier_checkpoint["val_acc"],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_image(samples[:100], output / "samples_10x10.png", nrow=10, padding=2)
    torch.save(samples, output / "samples.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train-classifier")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=2e-3)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--kind", choices=["fvsbn", "made", "gan", "flow", "imf"], required=True
    )
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--classifier", required=True)
    evaluate_parser.add_argument("--data-dir", default="data")
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--samples", type=int, default=5000)
    evaluate_parser.add_argument("--pr-samples", type=int, default=3000)
    evaluate_parser.add_argument("--generation-batch", type=int, default=256)
    evaluate_parser.add_argument("--feature-batch", type=int, default=512)
    evaluate_parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    seed_everything(5489)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if args.command == "train-classifier":
        train_classifier(args, device)
    else:
        evaluate(args, device)


if __name__ == "__main__":
    main()
