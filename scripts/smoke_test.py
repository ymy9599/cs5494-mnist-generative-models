#!/usr/bin/env python3
"""Run fast shape and autoregressive-causality checks on CPU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_baselines import Discriminator, FVSBN, Generator, MADE  # noqa: E402
from train_bcfm import BalancedConditionalFlow  # noqa: E402
from train_flows import VelocityUNet  # noqa: E402


def check_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, received {actual}")


def main() -> None:
    batch = 2
    pixels = torch.rand(batch, 784)

    fvsbn = FVSBN()
    check_shape("FVSBN", tuple(fvsbn(pixels).shape), (batch, 784))

    made = MADE(hidden=32, layers=2)
    made_input = pixels.clone().requires_grad_(True)
    made(made_input)[0, 100].backward()
    future_gradient = made_input.grad[0, 100:].abs().max().item()
    if future_gradient != 0.0:
        raise AssertionError(f"MADE causality violation: {future_gradient}")

    generator = Generator(latent_dim=16)
    generated = generator(torch.randn(batch, 16))
    check_shape("DCGAN generator", tuple(generated.shape), (batch, 1, 28, 28))
    check_shape(
        "DCGAN discriminator", tuple(Discriminator()(generated).shape), (batch,)
    )

    images = torch.randn(batch, 1, 28, 28)
    times = torch.rand(batch)
    intervals = torch.zeros(batch)
    flow = VelocityUNet(base=16, time_scale=100.0)
    check_shape(
        "Flow Matching",
        tuple(flow(images, times, intervals).shape),
        tuple(images.shape),
    )

    bcfm = BalancedConditionalFlow(base=16, time_scale=100.0)
    labels = torch.tensor([0, 1])
    check_shape("BCFM", tuple(bcfm(images, times, labels).shape), tuple(images.shape))

    print("All smoke tests passed.")


if __name__ == "__main__":
    main()
