#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/appendix}"
CLASSIFIER="${CLASSIFIER:-outputs/evaluator/classifier.pt}"
MAIN_BCFM="${MAIN_BCFM:-outputs/bcfm/best.pt}"

mkdir -p "$OUTPUT_DIR"

# The main pipeline supplies seed 5489. Two additional complete runs isolate
# variation from initialization, data order, and the fixed-size data split.
"$PYTHON_BIN" src/evaluate_bcfm.py \
  --checkpoint "$MAIN_BCFM" --classifier "$CLASSIFIER" \
  --data-dir "$DATA_DIR" --output "$OUTPUT_DIR/eval_seed_5489" --steps 10 \
  --reference-fp32

for seed in 5490 5491; do
  "$PYTHON_BIN" src/train_bcfm.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/seed_${seed}" \
    --epochs 80 --batch-size 256 --lr 6e-4 --seed "$seed" \
    --time-scale 100 --label-dropout 0 --sample-guidance 1
  "$PYTHON_BIN" src/evaluate_bcfm.py \
    --checkpoint "$OUTPUT_DIR/seed_${seed}/best.pt" \
    --classifier "$CLASSIFIER" --data-dir "$DATA_DIR" \
    --output "$OUTPUT_DIR/eval_seed_${seed}" --steps 10 --reference-fp32
done

# Learning rates are screened using validation loss after the same 20-epoch budget.
for lr in 1.5e-4 3e-4 6e-4 1.2e-3 2.4e-3 4.8e-3; do
  tag="${lr//./p}"
  "$PYTHON_BIN" src/train_bcfm.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/lr_${tag}" \
    --epochs 20 --batch-size 256 --lr "$lr" --seed 5489 \
    --time-scale 100 --label-dropout 0 --sample-guidance 1
done

# Check whether the two strongest short-budget rates remain preferable at the
# full 80-epoch horizon used by the final model.
for lr in 1.2e-3 2.4e-3; do
  tag="${lr//./p}"
  "$PYTHON_BIN" src/train_bcfm.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR/lr_${tag}_full" \
    --epochs 80 --batch-size 256 --lr "$lr" --seed 5489 \
    --time-scale 100 --label-dropout 0 --sample-guidance 1
done

# Label dropout provides the controlled classifier-free training ablation.
"$PYTHON_BIN" src/train_bcfm.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR/label_dropout_0p15" \
  --epochs 80 --batch-size 256 --lr 6e-4 --seed 5489 \
  --time-scale 100 --label-dropout 0.15 --sample-guidance 1
"$PYTHON_BIN" src/evaluate_bcfm.py \
  --checkpoint "$OUTPUT_DIR/label_dropout_0p15/best.pt" \
  --classifier "$CLASSIFIER" --data-dir "$DATA_DIR" \
  --output "$OUTPUT_DIR/eval_label_dropout_0p15" --steps 10 --reference-fp32
