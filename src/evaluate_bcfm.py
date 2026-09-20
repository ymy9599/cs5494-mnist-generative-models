"""Evaluate the optimized BCFM sampler and write all reported metrics."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from train_bcfm import BalancedConditionalFlow, seed_everything
from evaluate import (
    MNISTClassifier,
    extract,
    fraction_in_manifold,
    frechet_distance,
    loaders,
    manifold_radii,
)


@torch.inference_mode()
def integrate(model, z, labels, steps: int, use_bf16: bool = True):
    precision = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )
    with precision:
        times = torch.linspace(1, 0, steps + 1, device=z.device)
        for t, r in zip(times[:-1], times[1:]):
            z = z + (r - t) * model(z, t.expand(z.size(0)), labels)
    return z.add(1).div(2).clamp(0, 1)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--pr-samples", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--feature-batch", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--reference-fp32",
        action="store_true",
        help="Disable compilation and mixed precision for controlled analyses.",
    )
    args = parser.parse_args()
    seed_everything(5489)
    device = torch.device("cuda")
    classifier_checkpoint = torch.load(
        args.classifier, map_location=device, weights_only=False
    )
    classifier = MNISTClassifier().to(device)
    classifier.load_state_dict(classifier_checkpoint["model"])
    classifier.eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["config"]
    model = BalancedConditionalFlow(cfg["base_channels"], cfg["time_scale"]).to(
        device, memory_format=torch.channels_last
    )
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    if not args.reference_fp32:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    warm_noise = torch.randn(args.batch, 1, 28, 28, device=device).to(
        memory_format=torch.channels_last
    )
    warm_labels = torch.arange(args.batch, device=device) % 10
    warmup_iterations = 1 if args.reference_fp32 else 3
    for _ in range(warmup_iterations):
        integrate(
            model,
            warm_noise,
            warm_labels,
            args.steps,
            use_bf16=not args.reference_fp32,
        )
    seed_everything(5489)
    torch.cuda.synchronize(device)
    tic = time.perf_counter()
    generated = []
    requested = []
    for start in range(0, args.samples, args.batch):
        n = min(args.batch, args.samples - start)
        work_batch = n if args.reference_fp32 else args.batch
        labels = torch.arange(start, start + work_batch, device=device) % 10
        noise = torch.randn(work_batch, 1, 28, 28, device=device).to(
            memory_format=torch.channels_last
        )
        generated.append(
            integrate(
                model,
                noise,
                labels,
                args.steps,
                use_bf16=not args.reference_fp32,
            )[:n]
        )
        requested.append(labels[:n])
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - tic
    samples = torch.cat(generated).float().cpu()
    labels = torch.cat(requested).cpu()
    _, _, test_loader = loaders(args.data_dir, args.feature_batch)
    real = torch.cat([x for x, _ in test_loader])[: args.samples]
    fake_probs, fake_features = extract(classifier, samples, args.feature_batch, device)
    _, real_features = extract(classifier, real, args.feature_batch, device)
    marginal = fake_probs.mean(0)
    kl = (fake_probs * ((fake_probs + 1e-8).log() - (marginal + 1e-8).log())).sum(1)
    distribution = fake_probs.argmax(1).bincount(minlength=10).float()
    distribution /= distribution.sum()
    n = min(args.pr_samples, args.samples)
    real_pr = F.normalize(real_features[:n].float(), dim=1)
    fake_pr = F.normalize(fake_features[:n].float(), dim=1)
    metrics = {
        "kind": (
            "bcfm_fp32_reference"
            if args.reference_fp32
            else "bcfm_compile_bf16_channels_last"
        ),
        "nfe": args.steps,
        "samples": args.samples,
        "timing_boundary": (
            "warm-start batched FP32 sampling on GPU; excludes model load, D2H transfer, and metrics"
            if args.reference_fp32
            else "warm-start batched sampling on GPU; excludes model load, compilation, D2H transfer, and metrics"
        ),
        "generation_seconds": generation_seconds,
        "milliseconds_per_image": generation_seconds * 1000 / args.samples,
        "mnist_feature_distance": frechet_distance(
            real_features.numpy(), fake_features.numpy()
        ),
        "precision": fraction_in_manifold(fake_pr, real_pr, manifold_radii(real_pr)),
        "recall": fraction_in_manifold(real_pr, fake_pr, manifold_radii(fake_pr)),
        "mean_classifier_confidence": fake_probs.max(1).values.mean().item(),
        "high_confidence_fraction": (fake_probs.max(1).values >= 0.9)
        .float()
        .mean()
        .item(),
        "mnist_classifier_score": kl.mean().exp().item(),
        "normalized_class_entropy": float(
            -(distribution * (distribution + 1e-8).log()).sum() / math.log(10)
        ),
        "class_distribution": distribution.tolist(),
        "requested_label_accuracy": (fake_probs.argmax(1) == labels)
        .float()
        .mean()
        .item(),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_image(samples[:100], output / "samples_10x10.png", nrow=10, padding=2)
    torch.save(samples, output / "samples.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
