# Majorana Waveform Transformer

A transformer-based model for pulse shape discrimination (PSD) and energy regression on waveforms from the [Majorana Demonstrator Data Release for AI/ML Applications](https://arxiv.org/abs/2308.10856). Includes a Masked Autoencoder (MAE) self-supervised pre-training stage and a Gradient-Boosted Decision Tree (GBDT) baseline for comparison.

## Overview

The project implements three training pipelines:

1. **MAE Pre-training** — Self-supervised pre-training by masking and reconstructing waveform patches, optionally including unlabeled (NPML) data.
2. **Supervised Fine-tuning / Training from Scratch** — A transformer encoder with four binary PSD classification heads (DCR, high AvsE, low AvsE, LQ) and an energy regression head.
3. **GBDT Baseline** — Classical gradient-boosted decision trees trained on hand-crafted geometric features for PSD classification.

## Project Structure

```
dataset/              Dataset loading, collation, normalisation statistics
  compute_stats.py    Compute mean/std/min/max from training waveforms
  constants.py        Detector mapping and normalisation statistics
  dataset.py          PyTorch Dataset with baseline subtraction, augmentation, windowing
  dataloader.py       DataLoader creation with train/val split
  collate.py          Custom collate function
model/                Model definitions
  transformer.py      Transformer encoder with FiLM conditioning and task heads
  mae_pretrain.py     MAE encoder, decoder, and masking logic
  lightning_model.py  Lightning module for supervised training
  lightning_mae.py    Lightning module for MAE pre-training
train/                Training scripts and shell launchers
  pretrain_mae.py     MAE pre-training entry point
  train_lightning.py  Supervised training / fine-tuning entry point
  run_pretrain.sh     Launch MAE pre-training
  run_scratch.sh      Launch training from scratch
  run_finetune.sh     Launch fine-tuning from MAE checkpoint
  run_gbdt.sh         Full GBDT pipeline (extract → train → evaluate)
gbdt/                 GBDT baseline
  features.py         Hand-crafted geometric feature extraction
  prepare_data.py     Parallel feature extraction over all waveform files
  train_gbdt.py       Train one GBDT per PSD task
  evaluate_gbdt.py    Evaluate GBDT on test set
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Lightning
- scikit-learn
- numpy
- h5py (for raw data conversion)
- joblib

Install dependencies:

```bash
pip install torch pytorch-lightning scikit-learn numpy h5py joblib
```

## Data

This project uses the Majorana Demonstrator public data release ([arXiv:2308.10856](https://arxiv.org/abs/2308.10856)). Waveform files should be placed under `data/waveform_npz/` with the directory structure:

```
data/waveform_npz/
  MJD_Train_0/
  MJD_Train_1/
  ...
  MJD_Test_0/
  MJD_NPML_0/   (optional, unlabeled data for MAE pre-training)
  ...
```

Each `.npz` file contains a waveform array and associated labels (energy, PSD cuts, detector ID).

### Compute Normalisation Statistics

Before training, compute normalisation statistics from the training data:

```bash
python dataset/compute_stats.py --data_root data/waveform_npz --output dataset/stats.json
```

## Usage

### 1. MAE Pre-training

```bash
bash train/run_pretrain.sh
```

Or run directly:

```bash
python train/pretrain_mae.py \
    --train_root data/waveform_npz \
    --include_npml \
    --d_model 64 --nhead 8 --num_layers 6 \
    --decoder_d_model 64 --decoder_num_layers 2 \
    --mask_ratio 0.5 --window_size 10 \
    --batch_size 512 --lr 1.5e-4 \
    --warmup_epochs 20 --max_epochs 200 \
    --save_encoder_path checkpoints/mae_encoder.pt \
    --save_encoder_every_n_epochs 25 \
    --gpu_ids "0"
```

#### MAE Pre-training Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_root` | `data/waveform_npz` | Root directory for training data |
| `--train_val_split` | `0.9` | Train/val split ratio |
| `--include_npml` / `--no_npml` | Enabled | Include unlabeled NPML data for pre-training |
| `--d_model` | `128` | Encoder hidden dimension |
| `--nhead` | `8` | Number of attention heads |
| `--num_layers` | `4` | Number of encoder layers |
| `--decoder_d_model` | `64` | Decoder hidden dimension |
| `--decoder_num_layers` | `2` | Number of decoder layers |
| `--dropout` | `0.1` | Dropout probability |
| `--mask_ratio` | `0.25` | Fraction of tokens to mask |
| `--window_size` | `10` | Window size for sequence reshaping |
| `--waveform_loss_weight` | `1.0` | Weight for waveform reconstruction loss |
| `--gradient_loss_weight` | `0.5` | Weight for gradient reconstruction loss |
| `--second_deriv_loss_weight` | `0.5` | Weight for second derivative reconstruction loss |
| `--use_second_deriv` | `False` | Use second derivative features |
| `--use_detector` | `False` | Enable detector-specific FiLM conditioning |
| `--shift_augmentation` | `0` | Max random temporal shift (0 = disabled) |
| `--shift_pad_mode` | `edge` | Padding mode for shift augmentation (`edge` or `zero`) |
| `--batch_size` | `64` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay |
| `--beta1` / `--beta2` | `0.9` / `0.999` | AdamW betas |
| `--warmup_epochs` | `10` | Warmup epochs |
| `--max_epochs` | `200` | Maximum training epochs |
| `--patience` | `20` | Early stopping patience |
| `--save_encoder_path` | `checkpoints/mae_encoder.pt` | Path to save final encoder weights |
| `--save_encoder_every_n_epochs` | `50` | Periodic encoder checkpoint interval |
| `--precision` | `16-mixed` | Training precision |
| `--seed` | `42` | Random seed |
| `--gpu_ids` | `None` | Comma-separated GPU IDs (e.g., `"0,2,3"`) |
| `--compile` | `False` | Use `torch.compile` (PyTorch 2.0+) |

### 2. Supervised Training from Scratch

```bash
bash train/run_scratch.sh
```

Or run directly:

```bash
python train/train_lightning.py \
    --train_root data/waveform_npz \
    --d_model 64 --nhead 8 --num_layers 6 \
    --drop_path 0.1 --pooling gap \
    --predict_energy --normalize_energy \
    --window_size 10 --batch_size 256 --lr 1e-3 \
    --warmup_epochs 5 --max_epochs 20 \
    --label_smoothing 0.05 --use_detector \
    --gpu_ids "0" --compile
```

### 3. Fine-tuning from MAE Checkpoint

```bash
bash train/run_finetune.sh
```

Or run directly:

```bash
python train/train_lightning.py \
    --pretrained_encoder checkpoints/mae_encoder.pt \
    --layer_decay 0.75 \
    --train_root data/waveform_npz \
    --d_model 64 --nhead 8 --num_layers 6 \
    --drop_path 0.1 --pooling gap \
    --predict_energy --normalize_energy \
    --window_size 10 --batch_size 256 --lr 1e-3 \
    --warmup_epochs 5 --max_epochs 20 \
    --label_smoothing 0.05 --use_detector \
    --gpu_ids "0" --compile
```

#### Supervised Training / Fine-tuning Arguments

| Argument | Default | Description |
|---|---|---|
| `--pretrained_encoder` | `None` | Path to MAE encoder weights for fine-tuning |
| `--pretrained_model` | `None` | Path to a full Lightning checkpoint |
| `--layer_decay` | `0.65` | Layer-wise LR decay (1.0 = uniform, 0.65–0.75 typical for fine-tuning) |
| `--train_root` | `data/waveform_npz` | Root directory for training data |
| `--val_root` | `data/waveform_npz/MJD_Test_0` | Validation data directory (ignored if `--train_val_split` is set) |
| `--train_val_split` | `0.9` | Train/val split ratio |
| `--d_model` | `128` | Transformer hidden dimension |
| `--nhead` | `8` | Number of attention heads |
| `--num_layers` | `4` | Number of transformer layers |
| `--dropout` | `0.1` | Dropout probability |
| `--drop_path` | `0.0` | Stochastic depth rate |
| `--pooling` | `gap` | Pooling strategy (`gap` or `cls`) |
| `--predict_energy` / `--no_predict_energy` | Enabled | Enable energy regression head |
| `--learned_pe` | `False` | Use learned positional encodings instead of sinusoidal |
| `--window_size` | `1` | Window size for sequence reshaping |
| `--normalize_energy` | `True` | Normalise energy labels |
| `--use_second_deriv` | `False` | Use second derivative features |
| `--use_detector` | `False` | Enable detector-specific FiLM conditioning |
| `--shift_augmentation` | `0` | Max random temporal shift (0 = disabled) |
| `--shift_pad_mode` | `edge` | Padding mode for shift augmentation (`edge` or `zero`) |
| `--batch_size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate (linearly scaled by effective batch size / 256) |
| `--weight_decay` | `1e-5` | Weight decay |
| `--beta1` / `--beta2` | `0.9` / `0.999` | AdamW betas |
| `--warmup_epochs` | `5` | Warmup epochs |
| `--max_epochs` | `100` | Maximum training epochs |
| `--psd_loss_weight` | `1.0` | Weight for PSD classification loss |
| `--energy_loss_weight` | `0.1` | Weight for energy regression loss |
| `--label_smoothing` | `0.05` | Label smoothing factor |
| `--use_pos_weights` | `False` | Compute and apply positive class weights |
| `--patience` | `15` | Early stopping patience |
| `--gradient_clip_val` | `1.0` | Gradient clipping value |
| `--precision` | `16-mixed` | Training precision |
| `--seed` | `42` | Random seed |
| `--gpu_ids` | `None` | Comma-separated GPU IDs |
| `--compile` | `False` | Use `torch.compile` |

### 4. GBDT Baseline

Run the full pipeline (feature extraction → training → evaluation):

```bash
bash train/run_gbdt.sh
```

Or run each step individually:

```bash
# Extract features
python gbdt/prepare_data.py \
    --data_root data/waveform_npz \
    --output_dir gbdt/prepared \
    --num_workers 16

# Train
python gbdt/train_gbdt.py \
    --prepared_dir gbdt/prepared \
    --output_dir gbdt/models \
    --n_estimators 500 --max_depth 5 --learning_rate 0.1

# Evaluate
python gbdt/evaluate_gbdt.py \
    --prepared_dir gbdt/prepared \
    --model_dir gbdt/models \
    --output_dir gbdt/results
```

## License

This project is released under the [MIT License](LICENSE).
