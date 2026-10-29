#!/bin/bash

#SBATCH -A m2676_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH -J mae_shard0_m25
#SBATCH -o logs/mae_shard0_m25-%j.out
#SBATCH -e logs/mae_shard0_m25-%j.err

set -e

cd $HOME/waveform_mae_hpge

module load python
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate waveform-mae

mkdir -p logs
mkdir -p "$SCRATCH/waveform_mae_hpge_data/checkpoints/shard0_1k_mask25"
mkdir -p "$SCRATCH/waveform_mae_hpge_data/logs"

echo "Node: $(hostname)"
echo "Commit: $(git rev-parse HEAD)"
echo "Branch: $(git branch --show-current)"
echo "Python: $(python --version)"
echo "Experiment: window_size=10, mask_ratio=0.25"

nvidia-smi

srun python train/pretrain_mae.py \
    --train_root data/waveform_npz \
    --no_npml \
    --train_val_split 0.9 \
    --shift_augmentation 0 \
    --d_model 64 \
    --nhead 8 \
    --num_layers 6 \
    --decoder_d_model 64 \
    --decoder_num_layers 2 \
    --dropout 0.0 \
    --mask_ratio 0.25 \
    --window_size 10 \
    --batch_size 128 \
    --lr 1.5e-4 \
    --weight_decay 0.05 \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup_epochs 1 \
    --max_epochs 30 \
    --patience 30 \
    --num_workers 8 \
    --checkpoint_dir "$SCRATCH/waveform_mae_hpge_data/checkpoints/shard0_1k_mask25" \
    --log_dir "$SCRATCH/waveform_mae_hpge_data/logs" \
    --experiment_name shard0_1k_mask25 \
    --save_encoder_every_n_epochs 5 \
    --precision 16-mixed \
    --seed 42 \
    --devices 1 \
    --accelerator gpu \
    --strategy auto \
    --use_detector