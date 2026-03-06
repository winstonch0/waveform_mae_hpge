#!/bin/bash
# Sweep over different amounts of training data and epochs for fine-tuning
# This script fine-tunes pre-trained models with varying data sizes and epoch counts
# to analyze data efficiency vs training iterations (comparing against scratch training)

cd "$(dirname "$0")"/.. || exit

# Define arrays of data fractions and epoch counts to sweep over
DATA_SIZES=(1 2 4 8 16)  # Number of MJD_Train_* directories to use
EPOCHS=(2 5 10 20)      # Different epoch counts

# Pre-trained encoder path (update this to your best pre-trained checkpoint)
PRETRAINED_ENCODER="/scratch2/salonso/majorana/checkpoints/mae_pretrain_detector/encoder_checkpoints/mae_encoder_final_epoch0200.pt"

# Verify pre-trained encoder exists
if [ ! -f "$PRETRAINED_ENCODER" ]; then
    echo "ERROR: Pre-trained encoder not found at: $PRETRAINED_ENCODER"
    echo "Please update PRETRAINED_ENCODER path in this script"
    exit 1
fi

# Base checkpoint and log directories
BASE_CHECKPOINT_DIR="/scratch2/salonso/majorana/checkpoints/sweep_finetune_detector"
BASE_LOG_DIR="lightning_logs/sweep_finetune_detector"

# Create necessary directories
mkdir -p "$BASE_CHECKPOINT_DIR"
mkdir -p "$BASE_LOG_DIR"

# Common hyperparameters (matching run_finetune.sh)
COMMON_ARGS=(
    --pretrained_encoder "$PRETRAINED_ENCODER"
    --layer_decay 0.75
    --train_val_split 0.9
    --shift_augmentation 5
    --shift_pad_mode edge
    --d_model 64
    --nhead 8
    --num_layers 6
    --dropout 0.0
    --drop_path 0.1
    --pooling gap
    --predict_energy
    --batch_size 256
    --num_workers 16
    --window_size 10
    --lr 1e-3
    --weight_decay 0.05
    --beta1 0.9
    --beta2 0.999
    --psd_loss_weight 1.0
    --energy_loss_weight 0.5
    --label_smoothing 0.05
    --patience 1000
    --gradient_clip_val 0.5
    --precision 16-mixed
    --gpu_ids "0"
    --accelerator gpu
    --strategy auto
    --compile
    --use_detector
)

# Function to create data subset directory with symbolic links
create_data_subset() {
    local num_dirs=$1
    local subset_dir=$2
    
    # Create the subset directory
    mkdir -p "$subset_dir"
    
    # Link the first num_dirs training directories
    for i in $(seq 0 $((num_dirs - 1))); do
        src_dir="data/waveform_npz/MJD_Train_$i"
        dst_dir="$subset_dir/MJD_Train_$i"
        
        if [ -d "$src_dir" ] && [ ! -L "$dst_dir" ]; then
            ln -s "$(realpath $src_dir)" "$dst_dir"
            echo "Linked: $src_dir -> $dst_dir"
        fi
    done
}

# Main sweep loop
for data_size in "${DATA_SIZES[@]}"; do
    for epochs in "${EPOCHS[@]}"; do
        # Create a unique name for this configuration
        exp_name="finetune_data${data_size}_epoch${epochs}"
        echo "========================================="
        echo "Training: $exp_name"
        echo "Data size: ${data_size} directories"
        echo "Epochs: ${epochs}"
        echo "Pre-trained: $PRETRAINED_ENCODER"
        echo "========================================="
        
        # Create temporary data subset directory
        data_subset_dir="data/temp_subset_${data_size}"
        create_data_subset "$data_size" "$data_subset_dir"
        
        # Set seed for reproducibility (same seed as scratch for fair comparison)
        SEED=42
        
        # Calculate warmup epochs (10% of total, minimum 1, maximum 5)
        warmup_epochs=$(( epochs / 10 ))
        if [ $warmup_epochs -lt 1 ]; then
            warmup_epochs=1
        fi
        if [ $warmup_epochs -gt 5 ]; then
            warmup_epochs=5
        fi
        
        # Train the model
        PYTHONPATH=. python train/train_lightning.py \
            "${COMMON_ARGS[@]}" \
            --train_root "$data_subset_dir" \
            --max_epochs "$epochs" \
            --warmup_epochs "$warmup_epochs" \
            --checkpoint_dir "${BASE_CHECKPOINT_DIR}/${exp_name}" \
            --experiment_name "$exp_name" \
            --seed "$SEED"
        
        echo "Completed: $exp_name"
        echo ""
        
        # Clean up temporary subset directory
        rm -rf "$data_subset_dir"
    done
done

echo "========================================="
echo "Sweep completed!"
echo "Results saved in: $BASE_CHECKPOINT_DIR"
echo "Logs saved in: $BASE_LOG_DIR"
echo "========================================="
