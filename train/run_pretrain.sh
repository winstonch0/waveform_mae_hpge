#!/bin/bash
# Run MAE pre-training with default settings

python train/pretrain_mae.py \
    --train_root data/waveform_npz \
    --include_npml \
    --train_val_split 0.9 \
    --shift_augmentation 5 \
    --shift_pad_mode edge \
    --d_model 64 \
    --nhead 8 \
    --num_layers 6 \
    --decoder_d_model 64 \
    --decoder_num_layers 2 \
    --dropout 0.0 \
    --mask_ratio 0.5 \
    --window_size 10 \
    --batch_size 512 \
    --lr 1.5e-4 \
    --weight_decay 0.05 \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup_epochs 20 \
    --max_epochs 200 \
    --patience 200 \
    --num_workers 16 \
    --checkpoint_dir /scratch2/salonso/majorana/checkpoints/mae_pretrain_detector \
    --experiment_name mae_pretrain_detector \
    --save_encoder_path /scratch2/salonso/majorana/checkpoints/mae_encoder_detector.pt \
    --save_encoder_every_n_epochs 25 \
    --precision 16-mixed \
    --seed 42 \
    --gpu_ids "0" \
    --accelerator gpu \
    --strategy auto \
    --compile \
    --use_detector
