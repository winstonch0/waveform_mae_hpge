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
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pathlib import Path

torch.set_float32_matmul_precision('high')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import MajoranaDataset, DatasetConstants, majorana_collate_fn
from model import LitWaveformTransformer

def create_dataloaders(train_root, val_root, batch_size=32, num_workers=4, 
                       window_size=1, normalize_energy=True, train_val_split=None,
                       shift_augmentation=0, shift_pad_mode='edge', use_second_deriv=False):
    constants = DatasetConstants('dataset/stats.json')
    
    if train_val_split is not None:
        full_dataset = MajoranaDataset(
            root=train_root,
            window_size=window_size,
            constants=constants,
            to_torch=True,
            normalize_energy=normalize_energy,
            pattern_filter='MJD_Train',
            shift_augmentation=shift_augmentation,
            shift_pad_mode=shift_pad_mode,
            use_second_deriv=use_second_deriv
        )
        
        from torch.utils.data import random_split
        train_size = int(train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Use fixed seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        
        print(f'Split dataset: {len(full_dataset)} total -> {train_size} train, {val_size} val')
    else:
        train_dataset = MajoranaDataset(
            root=train_root,
            window_size=window_size,
            constants=constants,
            to_torch=True,
            normalize_energy=normalize_energy,
            pattern_filter='MJD_Train',
            shift_augmentation=shift_augmentation,
            shift_pad_mode=shift_pad_mode,
            use_second_deriv=use_second_deriv
        )
        
        val_dataset = MajoranaDataset(
            root=val_root,
            window_size=window_size,
            constants=constants,
            to_torch=True,
            test=True,
            normalize_energy=normalize_energy,
            shift_augmentation=0,  # No augmentation for validation
            shift_pad_mode=shift_pad_mode,
            use_second_deriv=use_second_deriv
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=majorana_collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=majorana_collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader

def compute_pos_weights(train_loader):
    psd_counts = {
        'dcr': {'pos': 0, 'neg': 0},
        'high_avse': {'pos': 0, 'neg': 0},
        'low_avse': {'pos': 0, 'neg': 0},
        'lq': {'pos': 0, 'neg': 0}
    }
    
    print("Computing class weights from training data...")
    for i, batch in enumerate(train_loader):
        if i >= 100:
            break
        
        for label in ['dcr', 'high_avse', 'low_avse', 'lq']:
            key = f'psd_label_{label}'
            if key in batch:
                pos = batch[key].sum().item()
                neg = (batch[key] == 0).sum().item()
                psd_counts[label]['pos'] += pos
                psd_counts[label]['neg'] += neg
    
    pos_weights = {}
    for label, counts in psd_counts.items():
        if counts['pos'] > 0:
            pos_weights[label] = counts['neg'] / counts['pos']
            print(f"  {label}: pos={counts['pos']}, neg={counts['neg']}, weight={pos_weights[label]:.3f}")
        else:
            pos_weights[label] = 1.0
    
    return pos_weights


class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used for validation
        return bar


def parse_args():
    parser = argparse.ArgumentParser(description='Train Waveform Transformer')
    
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                        help='Path to pre-trained MAE encoder weights (optional)')
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help='Path to pre-trained Lightning model checkpoint (e.g., LitMAEWaveform)')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='Layer-wise LR decay from MAE paper (1.0 = no decay, 0.65-0.75 typical)')
    
    parser.add_argument('--train_root', type=str, default='data/waveform_npz',
                        help='Root directory for training data')
    parser.add_argument('--val_root', type=str, default='data/waveform_npz/MJD_Test_0',
                        help='Root directory for validation data (ignored if --train_val_split is set)')
    parser.add_argument('--train_val_split', type=float, default=0.9,
                        help='Train/val split ratio (e.g., 0.9 = 90%% train, 10%% val). Use None to load separate datasets.')
    
    parser.add_argument('--d_model', type=int, default=128,
                        help='Transformer hidden dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--drop_path', type=float, default=0.0,
                        help='Drop path rate for stochastic depth')
    parser.add_argument('--pooling', type=str, default='gap', choices=['gap', 'cls'],
                        help='Pooling strategy: gap or cls')
    parser.add_argument('--predict_energy', action='store_true', default=True,
                        help='Predict energy regression')
    parser.add_argument('--no_predict_energy', action='store_false', dest='predict_energy',
                        help='Disable energy prediction')
    parser.add_argument('--learned_pe', action='store_true', default=False,
                        help='Use learned positional encodings instead of sinusoidal')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--window_size', type=int, default=1,
                        help='Window size for sequence reshaping')
    parser.add_argument('--normalize_energy', action='store_true', default=True,
                        help='Normalize energy labels')
    parser.add_argument('--shift_augmentation', type=int, default=0,
                        help='Maximum random temporal shift for augmentation (0=disabled)')
    parser.add_argument('--shift_pad_mode', type=str, default='edge', choices=['edge', 'zero'],
                        help='Padding mode for shift augmentation: edge or zero')
    parser.add_argument('--use_second_deriv', action='store_true', default=False,
                        help='Use second derivative features (disabled by default due to noise amplification)')
    parser.add_argument('--use_detector', action='store_true', default=False,
                        help='Enable detector-specific FiLM conditioning and energy calibration')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='AdamW beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='AdamW beta2 parameter')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--psd_loss_weight', type=float, default=1.0,
                        help='Weight for PSD classification loss')
    parser.add_argument('--energy_loss_weight', type=float, default=0.1,
                        help='Weight for energy regression loss')
    parser.add_argument('--label_smoothing', type=float, default=0.05,
                        help='Label smoothing factor (0.0 = no smoothing, 0.1 = 10%% smoothing)')
    
    parser.add_argument('--use_pos_weights', action='store_true', default=False,
                        help='Compute and use positive class weights')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='lightning_logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--experiment_name', type=str, default='waveform_transformer',
                        help='Experiment name for logging')
    
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        help='Training precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='Comma-separated GPU IDs to use (e.g., "0,2,3"). Overrides --devices.')
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator type (auto, gpu, cpu, tpu)')
    parser.add_argument('--strategy', type=str, default='auto',
                        help='Training strategy (auto, ddp, ddp_find_unused_parameters_true, etc.)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch.compile for faster training (PyTorch 2.0+)')
    
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
    
    train_loader, val_loader = create_dataloaders(
        train_root=args.train_root,
        val_root=args.val_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_size=args.window_size,
        normalize_energy=args.normalize_energy,
        train_val_split=args.train_val_split,
        shift_augmentation=args.shift_augmentation,
        shift_pad_mode=args.shift_pad_mode,
        use_second_deriv=args.use_second_deriv
    )
    
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    
    # Scale learning rate based on effective batch size (batch_size × num_gpus)
    # Linear scaling rule: lr = base_lr × effective_batch_size / 256
    base_lr = args.lr
    effective_batch_size = args.batch_size * devices
    scaled_lr = base_lr * effective_batch_size / 256
    print(f'Learning rate: base={base_lr:.2e}, scaled={scaled_lr:.2e} (batch_size={args.batch_size}, devices={devices}, effective={effective_batch_size})')
    
    pos_weights = None
    if args.use_pos_weights:
        pos_weights = compute_pos_weights(train_loader)
    
    model = LitWaveformTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        drop_path=args.drop_path,
        pooling=args.pooling,
        predict_energy=args.predict_energy,
        lr=scaled_lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        psd_loss_weight=args.psd_loss_weight,
        energy_loss_weight=args.energy_loss_weight,
        pos_weight=pos_weights,
        window_size=args.window_size,
        learned_pe=args.learned_pe,
        label_smoothing=args.label_smoothing,
        layer_decay=args.layer_decay if (args.pretrained_encoder or args.pretrained_model) else 1.0,
        use_detector=args.use_detector,
        use_second_deriv=args.use_second_deriv
    )
    
    # Load pre-trained encoder weights if provided
    if args.pretrained_encoder is not None:
        print(f'\nLoading pre-trained encoder from: {args.pretrained_encoder}')
        try:
            checkpoint = torch.load(args.pretrained_encoder, map_location='cpu')
            encoder_dict = checkpoint['encoder_state_dict']
            
            # Handle _orig_mod prefix from torch.compile
            if any(k.startswith('_orig_mod.') for k in encoder_dict.keys()):
                print('  - Detected compiled model (_orig_mod prefix), removing prefix...')
                encoder_dict = {k.replace('_orig_mod.', ''): v for k, v in encoder_dict.items()}
            
            # Load encoder weights (strict=False allows missing task heads)
            missing, unexpected = model.model.load_state_dict(encoder_dict, strict=False)
            
            print(f'  - Loaded encoder weights successfully')
            print(f'  - Expected missing keys (task heads): {len(missing)}')
            print("  - Missing keys:", missing)
            if unexpected:
                print(f'  - Warning: Unexpected keys: {len(unexpected)}')
            print(f'  - Using layer-wise LR decay: {args.layer_decay:.2f}')
            print(f'    (deeper layers get lower LRs for stable fine-tuning)')
        
        except Exception as e:
            print(f'  Failed to load pre-trained encoder: {e}')
            print(f'  Continuing with random initialization...')
    
    # Load pre-trained encoder from Lightning model checkpoint
    elif args.pretrained_model is not None:
        print(f'\nLoading pre-trained encoder from Lightning model: {args.pretrained_model}')
        try:
            checkpoint = torch.load(args.pretrained_model, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            # Extract encoder weights from Lightning checkpoint
            encoder_dict = {}
            for key, value in state_dict.items():
                # Handle various prefixes:
                # - "model.encoder." (standard)
                # - "model._orig_mod.encoder." (compiled model)
                if key.startswith('model._orig_mod.encoder.'):
                    new_key = key.replace('model._orig_mod.encoder.', '')
                    encoder_dict[new_key] = value
                elif key.startswith('model.encoder.'):
                    new_key = key.replace('model.encoder.', '')
                    encoder_dict[new_key] = value
            
            if not encoder_dict:
                print('  - Warning: No encoder weights found in checkpoint')
                print('  - Continuing with random initialization...')
            else:
                # Load encoder weights (strict=False allows missing task heads)
                missing, unexpected = model.model.load_state_dict(encoder_dict, strict=False)
                
                print(f'  - Loaded encoder weights successfully')
                print(f'  - Extracted {len(encoder_dict)} encoder parameters')
                print(f'  - Expected missing keys (task heads): {len(missing)}')
                if unexpected:
                    print(f'  - Warning: Unexpected keys: {len(unexpected)}')
                print(f'  - Using layer-wise LR decay: {args.layer_decay:.2f}')
                print(f'    (deeper layers get lower LRs for stable fine-tuning)')
        
        except Exception as e:
            print(f'  Failed to load pre-trained encoder: {e}')
            print(f'  Continuing with random initialization...')
    
    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            # Compile the inner model (WaveformTransformer), not the Lightning wrapper
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
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename='waveform-{epoch:02d}-{val_loss:.4f}',
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
    
    callbacks = [checkpoint_callback, lr_monitor, early_stop_callback, progress_bar]
    
    logger = TensorBoardLogger(args.log_dir, name=args.experiment_name)
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=devices,
        strategy=args.strategy,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=50,
        precision=args.precision
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    print(f'\nTraining complete!')
    print(f'Best model: {checkpoint_callback.best_model_path}')

if __name__ == '__main__':
    main()
