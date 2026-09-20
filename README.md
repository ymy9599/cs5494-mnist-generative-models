# MNIST Generative Models

Code for CS5494 Assignment 1. The repository contains the three required models—FVSBN, MADE, and DCGAN—plus Flow Matching, Improved MeanFlow (iMF), and Balanced Conditional Flow Matching (BCFM).

## Setup

Python 3.12 and a CUDA GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

MNIST is downloaded automatically by `torchvision` into `data/`.

## Reproduce the experiments

Run the complete training and evaluation pipeline:

```bash
bash scripts/reproduce.sh
```

The full pipeline trains every model with seed `5489` and writes checkpoints, learning curves, samples, and JSON metrics to `outputs/`. On an NVIDIA RTX 6000-class GPU, the training stages take about one hour in total.

For a quick code check without downloading data or training models:

```bash
python scripts/smoke_test.py
```

## Main files

- `src/train_baselines.py`: FVSBN, MADE, and DCGAN.
- `src/train_flows.py`: standard Flow Matching and iMF.
- `src/train_bcfm.py`: the proposed BCFM model.
- `src/evaluate.py`: frozen MNIST evaluator and common metrics.
- `src/evaluate_bcfm.py`: compiled BF16 evaluation for BCFM.

Use `python <file> --help` to inspect individual options. The report contains the model definitions, experimental protocol, and analysis.
