#!/usr/bin/env bash
set -euo pipefail

RUN="./runs/ofa_kernel_transform_10h"
DATA="./data"
PY="python"
SCRIPT="ofa_cifar10_options.py"

mkdir -p "$RUN/logs"
echo "=== START OFA 10H RUN ===" | tee "$RUN/logs/run_all.log"

run_step () {
  STEP="$1"
  shift

  echo
  echo "=== RUNNING $STEP ===" | tee -a "$RUN/logs/run_all.log"

  if "$@" > "$RUN/logs/${STEP}.log" 2>&1; then
    echo "DONE $STEP" | tee -a "$RUN/logs/run_all.log"
    return 0
  else
    echo "FAILED $STEP" | tee -a "$RUN/logs/run_all.log"

    if [[ "$STEP" == search_* ]]; then
      echo "Skipping failed search step: $STEP" | tee -a "$RUN/logs/run_all.log"
      return 1
    fi

    exit 1
  fi
}

run_step train_supernet \
  "$PY" "$SCRIPT" train_supernet \
  --data-dir "$DATA" \
  --save-dir "$RUN" \
  --epochs-largest 180 \
  --epochs-kernel 60 \
  --epochs-depth1 30 \
  --epochs-depth2 60 \
  --epochs-width1 30 \
  --epochs-width2 60 \
  --channel-sort \
  --shrink-lr 0.003 \
  --batch-size 256 \
  --num-workers 2 \
  --log-interval 50

run_step sample_acc_dataset \
  "$PY" "$SCRIPT" sample_acc_dataset \
  --data-dir "$DATA" \
  --save-dir "$RUN" \
  --checkpoint "$RUN/checkpoints/supernet_last.pt" \
  --num-arch-samples 30000 \
  --output "$RUN/acc_dataset_30000.jsonl" \
  --bn-calib-batches 20 \
  --val-batches 50 \
  --batch-size 256 \
  --num-workers 2 \
  --eval-preset accurate

run_step train_acc_predictor \
  "$PY" "$SCRIPT" train_acc_predictor \
  --save-dir "$RUN" \
  --dataset "$RUN/acc_dataset_30000.jsonl" \
  --predictor-epochs 600 \
  --predictor-batch-size 256 \
  --predictor-lr 0.001 \
  --predictor-weight-decay 0.0001

for M in 20 25 30 40 60 80 120 160; do
  if run_step "search_${M}m" \
    "$PY" "$SCRIPT" search \
    --predictor "$RUN/accuracy_predictor.pt" \
    --save-dir "$RUN/search_${M}m" \
    --macs-limit "${M}e6" \
    --population-size 300 \
    --parent-size 75 \
    --generations 200 \
    --mutation-prob 0.1
  then
    run_step "eval_${M}m" \
      "$PY" "$SCRIPT" eval_subnet \
      --data-dir "$DATA" \
      --checkpoint "$RUN/checkpoints/supernet_last.pt" \
      --spec-json "$RUN/search_${M}m/search/best_spec.json" \
      --bn-calib-batches 50 \
      --batch-size 256 \
      --num-workers 2 \
      --eval-preset accurate
  else
    echo "No feasible subnet for ${M}M MACs, skipping eval_${M}m" | tee -a "$RUN/logs/run_all.log"
  fi
done

run_step eval_80m_no_bn \
  "$PY" "$SCRIPT" eval_subnet \
  --data-dir "$DATA" \
  --checkpoint "$RUN/checkpoints/supernet_last.pt" \
  --spec-json "$RUN/search_80m/search/best_spec.json" \
  --bn-calib-batches 0 \
  --batch-size 256 \
  --num-workers 2 \
  --eval-preset accurate

echo "=== ALL DONE ===" | tee -a "$RUN/logs/run_all.log"
