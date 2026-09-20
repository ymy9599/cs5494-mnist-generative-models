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

The robustness, learning-rate sensitivity, and conditioning ablations reported
in the appendix can be reproduced after the main pipeline has produced the
seed-5489 BCFM checkpoint and frozen evaluator:

```bash
bash scripts/reproduce_appendix.sh
```

## Main files

- `src/train_baselines.py`: FVSBN, MADE, and DCGAN.
- `src/train_flows.py`: standard Flow Matching and iMF.
- `src/train_bcfm.py`: the proposed BCFM model.
- `src/evaluate.py`: frozen MNIST evaluator and common metrics.
- `src/evaluate_bcfm.py`: compiled BF16 evaluation for BCFM.
- `scripts/reproduce_appendix.sh`: optional controlled analyses from the report appendix.

Use `python <file> --help` to inspect individual options. The report contains the model definitions, experimental protocol, and analysis.
