#!/usr/bin/env bash

set -e

VENV_PATH="${VENV_PATH:-/workspace/douzj/AutoTSA-master/douzj_AutoTSA}"
if [ -f "${VENV_PATH}/bin/activate" ]; then
  source "${VENV_PATH}/bin/activate"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/douzj/AutoTSA-master/dataset}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints}"
TEST_RESULT_DIR="${TEST_RESULT_DIR:-test_results}"

datasets=(
  "ArticularyWordRecognition"
)

downstreams=(
  "ROCKET"
)

tsas=(
  "AutoDA-Timeseries"
)

BATCH_SIZE=16
LR=0.005
TRAIN_EPOCHS=35 # 50
PATIENCE=17

for dataset in "${datasets[@]}"; do
  for downstream in "${downstreams[@]}"; do
    for tsa in "${tsas[@]}"; do

      ts="$(date +%Y%m%d-%H%M%S)"
      log_file="log/${ts}.${tsa}.${dataset}.${downstream}.classification.txt"
      mkdir -p log

      echo "================================================="
      echo "[RUN] task=classification dataset=${dataset} model=${downstream} tsa=${tsa}"
      echo "log_file=${log_file}"
      echo "================================================="

      python -u run.py \
        --task "classification" \
        --tsa "$tsa" \
        --downstream "$downstream" \
        --feature_extractor Catch22 \
        --tsa_config_path "examples/AutoDA-Timeseries.classification.json" \
        --dataset_root "$DATASET_ROOT" \
        --dataset "$dataset" \
        --dataset_type "default" \
        --log_file "$log_file" \
        --checkpoints "$CHECKPOINTS_DIR" \
        --test_result_dir "$TEST_RESULT_DIR" \
        --use_gpu 1 \
        --batch_size $BATCH_SIZE \
        --is_training 1 \
        --skip_exist 1 \
        --itr 1 \
        --learning_rate $LR \
        --train_epochs $TRAIN_EPOCHS \
        --patience $PATIENCE

      sleep 2  # Avoid duplicate timestamps.
    done
  done
done
