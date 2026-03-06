#!/bin/bash
# ==============================================================================
# GBDT Training and Evaluation Pipeline
# ==============================================================================
#
# Trains Gradient-Boosted Decision Tree classifiers for PSD tasks
# (DCR, high AvsE, low AvsE, LQ) using hand-crafted geometric features.
#
# This provides a classical ML baseline for comparison with the transformer.
#
# Steps:
#   1. Extract features from all waveforms (parallelized)
#   2. Train one GBDT per PSD task (with 90/10 train/val split, seed=42)
#   3. Evaluate on the held-out test set (MJD_Test_*)
#
# Usage:
#   bash gbdt/run_gbdt.sh
# ==============================================================================

set -e

DATA_ROOT="data/waveform_npz"
PREPARED_DIR="gbdt/prepared"
MODEL_DIR="gbdt/models"
RESULTS_DIR="gbdt/results"
NUM_WORKERS=16

echo "=============================================="
echo "GBDT Pipeline for Majorana PSD Classification"
echo "=============================================="
echo ""

# Step 1: Feature extraction
echo "[Step 1/3] Extracting features from waveforms..."
echo "  Data root:    ${DATA_ROOT}"
echo "  Output:       ${PREPARED_DIR}"
echo "  Workers:      ${NUM_WORKERS}"
echo ""

PYTHONPATH=. python gbdt/prepare_data.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${PREPARED_DIR}" \
    --num_workers "${NUM_WORKERS}"

echo ""
echo "[Step 2/3] Training GBDT models..."
echo ""

# Step 2: Train GBDT models
PYTHONPATH=. python gbdt/train_gbdt.py \
    --prepared_dir "${PREPARED_DIR}" \
    --output_dir "${MODEL_DIR}" \
    --n_estimators 500 \
    --max_depth 5 \
    --learning_rate 0.1 \
    --subsample 0.8 \
    --min_samples_split 100 \
    --min_samples_leaf 50 \
    --max_features sqrt \
    --train_val_split 0.9 \
    --seed 42

echo ""
echo "[Step 3/3] Evaluating on test set..."
echo ""

# Step 3: Evaluate on test set
PYTHONPATH=. python gbdt/evaluate_gbdt.py \
    --prepared_dir "${PREPARED_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --output_dir "${RESULTS_DIR}"

echo ""
echo "=============================================="
echo "GBDT Pipeline Complete!"
echo "  Models:  ${MODEL_DIR}/"
echo "  Results: ${RESULTS_DIR}/"
echo "=============================================="
