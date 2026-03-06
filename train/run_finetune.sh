#!/bin/bash
# Fine-tune with pre-trained encoder using layer-wise LR decay

python train/train_lightning.py \
    --pretrained_encoder /scratch2/salonso/majorana/checkpoints/mae_pretrain_detector/encoder_checkpoints/mae_encoder_final_epoch0200.pt \
    --layer_decay 0.75 \
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
    --window_size 10 \
    --batch_size 256 \
    --lr 1e-3 \
    --weight_decay 0.05 \
    --beta1 0.9 \
    --beta2 0.999 \
    --warmup_epochs 5 \
    --max_epochs 20 \
    --psd_loss_weight 1.0 \
    --energy_loss_weight 0.5 \
    --label_smoothing 0.05 \
    --patience 15 \
    --num_workers 16 \
    --checkpoint_dir /scratch2/salonso/majorana/checkpoints/finetune_mse_detector \
    --experiment_name finetune_mse_detector \
    --precision 16-mixed \
    --seed 42 \
    --gpu_ids "1" \
    --accelerator gpu \
    --strategy auto \
    --compile \
    --use_detector
