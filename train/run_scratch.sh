#!/bin/bash

cd "$(dirname "$0")"/.. || exit

# Optimized configuration to prevent overfitting
PYTHONPATH=. python train/train_lightning.py \
    --train_root data/waveform_npz \
    --train_val_split 0.9 \
    --shift_augmentation 5 \
    --shift_pad_mode edge \
    --d_model 64 \
    --nhead 8 \
    --num_layers 6 \
    --dropout 0.0 \
    --drop_path 0.1 \
    --pooling gap \
    --predict_energy \
    --batch_size 256 \
    --num_workers 16 \
    --window_size 10 \
    --normalize_energy \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --beta1 0.9 \
    --beta2 0.999 \
    --warmup_epochs 5 \
    --max_epochs 40 \
    --psd_loss_weight 1.0 \
    --energy_loss_weight 0.5 \
    --label_smoothing 0.05 \
    --checkpoint_dir /scratch2/salonso/majorana/checkpoints/scratch_mse_detector2 \
    --log_dir lightning_logs \
    --experiment_name waveform_scratch_mse_detector2 \
    --patience 10 \
    --gradient_clip_val 0.5 \
    --precision 16-mixed \
    --seed 42 \
    --gpu_ids "0" \
    --accelerator gpu \
    --strategy auto \
    --compile \
    --use_detector
