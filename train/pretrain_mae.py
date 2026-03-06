"""
Pre-training script for MAE waveform model.

Uses:
- MJD_Train_* files: same 90/10 split as supervised training
- MJD_NPML_* files: additional unlabeled data
- MJD_Test_* files: NEVER used (reserved for final evaluation)
"""
import sys
import os
import argparse

# Parse gpu_ids BEFORE any CUDA imports to set CUDA_VISIBLE_DEVICES early
if '--gpu_ids' in sys.argv:
    idx = sys.argv.index('--gpu_ids')
    if idx + 1 < len(sys.argv):
        gpu_ids = sys.argv[idx + 1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f'[GPU Setup] Setting CUDA_VISIBLE_DEVICES={gpu_ids}')
        print(f'[GPU Setup] Physical GPU(s) {gpu_ids} will be mapped to logical device(s) starting from 0')

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pathlib import Path

torch.set_float32_matmul_precision('high')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import MajoranaDataset, DatasetConstants, majorana_collate_fn
from model.lightning_mae import LitMAEWaveform


class PeriodicEncoderCheckpoint(Callback):
    """
    Callback to save encoder weights every N epochs for fine-tuning experiments.
    
    This allows you to test fine-tuning from different pre-training checkpoints
    without waiting for full pre-training to complete.
    """
    def __init__(self, save_dir, save_every_n_epochs=50, save_prefix="mae_encoder"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.save_prefix = save_prefix
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Save encoder weights at specified intervals."""
        epoch = trainer.current_epoch
        
        # Save at specified intervals (but not at epoch 0)
        if epoch > 0 and epoch % self.save_every_n_epochs == 0:
            filepath = self.save_dir / f"{self.save_prefix}_epoch{epoch:04d}.pt"
            pl_module.save_encoder_weights(str(filepath))
            print(f"\n[Checkpoint] Saved encoder at epoch {epoch} -> {filepath}")
    
    def on_train_end(self, trainer, pl_module):
        """Save final encoder weights."""
        epoch = trainer.current_epoch
        filepath = self.save_dir / f"{self.save_prefix}_final_epoch{epoch:04d}.pt"
        pl_module.save_encoder_weights(str(filepath))
        print(f"\n[Checkpoint] Saved final encoder at epoch {epoch} -> {filepath}")


class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True
        return bar


def create_pretrain_dataloaders(
    train_root, 
    batch_size=64, 
    num_workers=4,
    window_size=10,
    train_val_split=0.9,
    include_npml=True,
    shift_augmentation=0,
    shift_pad_mode='edge',
    use_second_deriv=False
):
    """
    Create dataloaders for pre-training with proper data splits.
    
    Uses same train/val split as supervised training (reproducible with seed=42).
    Optionally includes NPML (unlabeled) data for more pre-training samples.
    
    Args:
        train_root: root directory containing waveform_npz data
        batch_size: batch size
        num_workers: number of dataloader workers
        window_size: window size for sequence reshaping
        train_val_split: train/val split ratio (0.9 = 90% train, 10% val)
        include_npml: whether to include MJD_NPML_* files
    
    Returns:
        train_loader, val_loader
    """
    constants = DatasetConstants('dataset/stats.json')
    
    # Load MJD_Train_* dataset
    train_dataset = MajoranaDataset(
        root=train_root,
        window_size=window_size,
        constants=constants,
        to_torch=True,
        normalize_energy=False,  # Not needed for pre-training
        pattern_filter='MJD_Train',
        shift_augmentation=shift_augmentation,
        shift_pad_mode=shift_pad_mode,
        use_second_deriv=use_second_deriv
    )
    
    print(f'MJD_Train_* dataset: {len(train_dataset)} samples')
    
    # Split MJD_Train_* with same seed as supervised training
    from torch.utils.data import random_split
    train_size = int(train_val_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)  # Same seed as train_lightning.py
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], generator=generator
    )
    
    print(f'Split MJD_Train_*: {train_size} train, {val_size} val')
    
    # Optionally add NPML data to training set only
    if include_npml:
        npml_dataset = MajoranaDataset(
            root=train_root,
            window_size=window_size,
            constants=constants,
            to_torch=True,
            normalize_energy=False,
            pattern_filter='MJD_NPML',
            shift_augmentation=shift_augmentation,
            shift_pad_mode=shift_pad_mode,
            use_second_deriv=use_second_deriv
        )
        print(f'MJD_NPML_* dataset: {len(npml_dataset)} samples')
        
        # Concatenate train subset with NPML
        train_subset = ConcatDataset([train_subset, npml_dataset])
        print(f'Total pre-training samples: {len(train_subset)}')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=majorana_collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # For stable batch sizes
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=majorana_collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-train MAE Waveform Model')
    
    # Data args
    parser.add_argument('--train_root', type=str, default='data/waveform_npz',
                        help='Root directory for training data')
    parser.add_argument('--train_val_split', type=float, default=0.9,
                        help='Train/val split ratio (same as supervised training)')
    parser.add_argument('--include_npml', action='store_true', default=True,
                        help='Include MJD_NPML_* files for pre-training')
    parser.add_argument('--no_npml', action='store_false', dest='include_npml',
                        help='Exclude MJD_NPML_* files')
    
    # Model args
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer hidden dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of encoder layers')
    parser.add_argument('--decoder_d_model', type=int, default=64,
                        help='Decoder hidden dimension')
    parser.add_argument('--decoder_num_layers', type=int, default=2,
                        help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--learned_pe', action='store_true', default=False,
                        help='Use learned positional encodings')
    
    # MAE args
    parser.add_argument('--mask_ratio', type=float, default=0.25,
                        help='Ratio of tokens to mask (0.15-0.30 recommended)')
    parser.add_argument('--waveform_loss_weight', type=float, default=1.0,
                        help='Weight for waveform reconstruction loss')
    parser.add_argument('--gradient_loss_weight', type=float, default=0.5,
                        help='Weight for gradient reconstruction loss')
    parser.add_argument('--second_deriv_loss_weight', type=float, default=0.5,
                        help='Weight for second derivative reconstruction loss')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (can be larger than supervised training)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--window_size', type=int, default=10,
                        help='Window size for sequence reshaping')
    parser.add_argument('--shift_augmentation', type=int, default=0,
                        help='Maximum random temporal shift for augmentation (0=disabled)')
    parser.add_argument('--shift_pad_mode', type=str, default='edge', choices=['edge', 'zero'],
                        help='Padding mode for shift augmentation: edge or zero')
    parser.add_argument('--use_second_deriv', action='store_true', default=False,
                        help='Use second derivative features (disabled by default due to noise amplification)')
    parser.add_argument('--use_detector', action='store_true', default=False,
                        help='Enable detector-specific FiLM conditioning and energy calibration')
    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (higher than fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2 parameter')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum number of epochs')
    
    # Checkpoint args
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/mae_pretrain',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='lightning_logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--experiment_name', type=str, default='mae_pretrain',
                        help='Experiment name for logging')
    parser.add_argument('--save_encoder_path', type=str, default='checkpoints/mae_encoder.pt',
                        help='Path to save final encoder weights for fine-tuning')
    parser.add_argument('--save_encoder_every_n_epochs', type=int, default=50,
                        help='Save encoder weights every N epochs for intermediate fine-tuning tests')
    
    # Training config
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        help='Training precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # GPU args
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated GPU IDs (e.g., "0,2,3")')
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator type (auto, gpu, cpu)')
    parser.add_argument('--strategy', type=str, default='auto',
                        help='Training strategy (auto, ddp, etc.)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile (PyTorch 2.0+)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    
    # Determine number of devices (CUDA_VISIBLE_DEVICES already set at import time)
    if args.gpu_ids is not None:
        gpu_list = [int(i) for i in args.gpu_ids.split(',')]
        devices = len(gpu_list)
        print(f'\n[GPU Setup] Using {devices} GPU(s): physical IDs {args.gpu_ids} (mapped to logical IDs {list(range(devices))})')
        if torch.cuda.is_available():
            print(f'[GPU Setup] PyTorch sees {torch.cuda.device_count()} device(s)')
            for i in range(torch.cuda.device_count()):
                print(f'[GPU Setup]   Logical device {i}: {torch.cuda.get_device_name(i)}')
    else:
        devices = args.devices
        if torch.cuda.is_available():
            print(f'\n[GPU Setup] Using {devices} GPU(s) (no CUDA_VISIBLE_DEVICES restriction)')
            print(f'[GPU Setup] PyTorch sees {torch.cuda.device_count()} available device(s)')
    
    # Create dataloaders
    train_loader, val_loader = create_pretrain_dataloaders(
        train_root=args.train_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_size=args.window_size,
        train_val_split=args.train_val_split,
        include_npml=args.include_npml,
        shift_augmentation=args.shift_augmentation,
        shift_pad_mode=args.shift_pad_mode,
        use_second_deriv=args.use_second_deriv
    )
    
    print(f'\nPre-training data:')
    print(f'  Train batches: {len(train_loader)}')
    print(f'  Val batches: {len(val_loader)}')
    print(f'  Mask ratio: {args.mask_ratio:.2%}')
    print(f'  Window size: {args.window_size}')
    print(f'  NPML included: {args.include_npml}')
    print(f'  Second derivative: {args.use_second_deriv}')
    if args.shift_augmentation > 0:
        print(f'  Shift augmentation: ±{args.shift_augmentation} timesteps ({args.shift_pad_mode} padding)')
    print(f'\nNOTE: MJD_Test_* files are NOT used (reserved for final evaluation)\n')
    
    # Create model
    model = LitMAEWaveform(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        decoder_d_model=args.decoder_d_model,
        decoder_num_layers=args.decoder_num_layers,
        dropout=args.dropout,
        window_size=args.window_size,
        learned_pe=args.learned_pe,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        waveform_loss_weight=args.waveform_loss_weight,
        gradient_loss_weight=args.gradient_loss_weight,
        second_deriv_loss_weight=args.second_deriv_loss_weight,
        use_second_deriv=args.use_second_deriv,
        use_detector=args.use_detector
    )
    
    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            # Compile the inner model (MAEWaveformModel), not the Lightning wrapper
            model.model = torch.compile(
                model.model, 
                mode='default',
                fullgraph=True,  # Enforce full graph compilation
                dynamic=False     # Disable dynamic shapes
            )
            print("  ✓ Model compiled successfully (fullgraph=True)")
        except Exception as e:
            print(f"  ✗ Compilation failed: {e}")
            print("  Continuing without compilation...")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='mae-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    progress_bar = CustomProgressBar()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min',
        verbose=True
    )
    
    # Periodic encoder checkpoint callback
    encoder_checkpoint_dir = Path(args.checkpoint_dir) / 'encoder_checkpoints'
    periodic_encoder_callback = PeriodicEncoderCheckpoint(
        save_dir=encoder_checkpoint_dir,
        save_every_n_epochs=args.save_encoder_every_n_epochs,
        save_prefix='mae_encoder'
    )
    
    logger = TensorBoardLogger(args.log_dir, name=args.experiment_name)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=devices,
        strategy=args.strategy,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback, progress_bar, periodic_encoder_callback],
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=50,
        precision=args.precision
    )
    
    # Train
    print('\nStarting MAE pre-training...\n')
    trainer.fit(model, train_loader, val_loader)
    
    # Save encoder weights for fine-tuning
    print(f'\nPre-training complete!')
    print(f'Best checkpoint: {checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    main()
