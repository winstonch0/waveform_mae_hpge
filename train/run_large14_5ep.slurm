#!/bin/bash

#SBATCH -A m2676_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J mae_large14
#SBATCH -o logs/mae_large14-%j.out
#SBATCH -e logs/mae_large14-%j.err

set -euo pipefail

cd "$HOME/waveform_mae_hpge"

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate waveform-mae

export OMP_NUM_THREADS=1

TRAIN_ROOT="$SCRATCH/waveform_mae_hpge_data/large_npz/train"
CHECKPOINT_DIR="$SCRATCH/waveform_mae_hpge_data/checkpoints/large14_5ep"
LOG_DIR="$SCRATCH/waveform_mae_hpge_data/logs"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "LARGE-DATA MAE TRAINING"
echo "========================================"
echo "Node:       $(hostname)"
echo "Date:       $(date)"
echo "Branch:     $(git branch --show-current)"
echo "Commit:     $(git rev-parse HEAD)"
echo "Train root: $TRAIN_ROOT"
echo
echo "Configuration:"
echo "  shards:        MJD_Train_0 through 13"
echo "  external test: MJD_Train_14 (NOT USED HERE)"
echo "  window_size:   10"
echo "  mask_ratio:    0.50"
echo "  epochs:        5"
echo "  batch_size:    128"
echo "  shift_aug:     0"
echo "========================================"

echo
echo "NPZ count:"
find "$TRAIN_ROOT" -type f -name '*.npz' | wc -l

echo
nvidia-smi

echo
echo "Starting training..."
echo

srun python train/pretrain_mae.py \
    --train_root "$TRAIN_ROOT" \
    --no_npml \
    --train_val_split 0.98 \
    --shift_augmentation 0 \
    --d_model 64 \
    --nhead 8 \
    --num_layers 6 \
    --decoder_d_model 64 \
    --decoder_num_layers 2 \
    --dropout 0.0 \
    --mask_ratio 0.5 \
    --window_size 10 \
    --batch_size 128 \
    --lr 1.5e-4 \
    --weight_decay 0.05 \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup_epochs 1 \
    --max_epochs 5 \
    --patience 5 \
    --num_workers 8 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_dir "$LOG_DIR" \
    --experiment_name large14_5ep \
    --save_encoder_path "$CHECKPOINT_DIR/mae_encoder_final.pt" \
    --save_encoder_every_n_epochs 1 \
    --precision 16-mixed \
    --seed 42 \
    --devices 1 \
    --accelerator gpu \
    --strategy auto \
    --use_detector

echo
echo "========================================"
echo "TRAINING FINISHED"
echo "Date: $(date)"
echo "========================================"
