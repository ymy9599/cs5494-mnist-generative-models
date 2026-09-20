#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

mkdir -p "$OUTPUT_DIR"

# Required models.
"$PYTHON_BIN" src/train_baselines.py --model fvsbn --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR/fvsbn" --epochs 40 --batch-size 256 --lr 1e-3
"$PYTHON_BIN" src/train_baselines.py --model made --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR/made" --epochs 50 --batch-size 256 --lr 1e-3
"$PYTHON_BIN" src/train_baselines.py --model gan --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR/dcgan" --epochs 50 --batch-size 256 --lr 2e-4

# Flow-based models.
"$PYTHON_BIN" src/train_flows.py --method flow --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR/flow" --epochs 80 --batch-size 256 --lr 6e-4 --time-scale 100
"$PYTHON_BIN" src/train_flows.py --method imf --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR/imf" --epochs 80 --batch-size 256 --lr 3e-4 --time-scale 100 --adaptive-power 0 --auxiliary-head
"$PYTHON_BIN" src/train_bcfm.py --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR/bcfm" --epochs 80 --batch-size 256 --lr 6e-4 --time-scale 100 --label-dropout 0 --sample-guidance 1

# Frozen evaluator used by all reported sample metrics.
"$PYTHON_BIN" src/evaluate.py train-classifier --data-dir "$DATA_DIR" --output "$OUTPUT_DIR/evaluator" --epochs 12 --batch-size 256 --lr 2e-3

"$PYTHON_BIN" src/evaluate.py evaluate --kind fvsbn --checkpoint "$OUTPUT_DIR/fvsbn/best.pt" --classifier "$OUTPUT_DIR/evaluator/classifier.pt" --data-dir "$DATA_DIR" --output "$OUTPUT_DIR/eval_fvsbn"
"$PYTHON_BIN" src/evaluate.py evaluate --kind made --checkpoint "$OUTPUT_DIR/made/best.pt" --classifier "$OUTPUT_DIR/evaluator/classifier.pt" --data-dir "$DATA_DIR" --output "$OUTPUT_DIR/eval_made"
"$PYTHON_BIN" src/evaluate.py evaluate --kind gan --checkpoint "$OUTPUT_DIR/dcgan/best.pt" --classifier "$OUTPUT_DIR/evaluator/classifier.pt" --data-dir "$DATA_DIR" --output "$OUTPUT_DIR/eval_dcgan"

for steps in 4 10 25 50; do
  "$PYTHON_BIN" src/evaluate.py evaluate --kind flow --checkpoint "$OUTPUT_DIR/flow/best.pt" --classifier "$OUTPUT_DIR/evaluator/classifier.pt" --data-dir "$DATA_DIR" --output "$OUTPUT_DIR/eval_flow_${steps}" --steps "$steps"
  "$PYTHON_BIN" src/evaluate.py evaluate --kind imf --checkpoint "$OUTPUT_DIR/imf/best.pt" --classifier "$OUTPUT_DIR/evaluator/classifier.pt" --data-dir "$DATA_DIR" --output "$OUTPUT_DIR/eval_imf_${steps}" --steps "$steps"
  "$PYTHON_BIN" src/evaluate_bcfm.py --checkpoint "$OUTPUT_DIR/bcfm/best.pt" --classifier "$OUTPUT_DIR/evaluator/classifier.pt" --data-dir "$DATA_DIR" --output "$OUTPUT_DIR/eval_bcfm_${steps}" --steps "$steps"
done
