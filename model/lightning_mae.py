"""
PyTorch Lightning module for MAE pre-training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from .mae_pretrain import MAEWaveformModel


class LitMAEWaveform(pl.LightningModule):
    """
    Lightning module for MAE pre-training of waveform transformers.
    """
    def __init__(
        self,
        d_model=128,
        nhead=8,
        num_layers=4,
        decoder_d_model=64,
        decoder_num_layers=2,
        dropout=0.1,
        window_size=10,
        learned_pe=False,
        mask_ratio=0.25,
        lr=1e-3,
        weight_decay=1e-5,
        beta1=0.9,
        beta2=0.95,
        warmup_epochs=5,
        max_epochs=100,
        waveform_loss_weight=1.0,
        gradient_loss_weight=0.5,
        second_deriv_loss_weight=0.5,
        use_second_deriv=False,
        use_detector=False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MAEWaveformModel(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            decoder_d_model=decoder_d_model,
            decoder_num_layers=decoder_num_layers,
            dropout=dropout,
            window_size=window_size,
            learned_pe=learned_pe,
            mask_ratio=mask_ratio,
            use_second_deriv=use_second_deriv,
            use_detector=use_detector
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.waveform_loss_weight = waveform_loss_weight
        self.gradient_loss_weight = gradient_loss_weight
        self.second_deriv_loss_weight = second_deriv_loss_weight
        self.use_second_deriv = use_second_deriv
    
    def forward(self, waveform, gradient, second_deriv, detector=None):
        return self.model(waveform, gradient, second_deriv, detector=detector)
    
    def _compute_loss(self, outputs, batch, mask):
        """
        Compute reconstruction loss only on masked tokens.
        """
        waveform_recon, gradient_recon, second_deriv_recon = outputs
        
        # Original inputs
        waveform = batch['waveform']
        gradient = batch['gradient']
        second_deriv = batch['second_deriv']
        
        # Compute loss only on masked tokens (mask is False for masked positions)
        mask_expanded = (~mask).unsqueeze(-1).expand_as(waveform)  # Match full shape
        
        # MSE loss on masked positions only
        waveform_loss = F.mse_loss(
            waveform_recon[mask_expanded], 
            waveform[mask_expanded],
            reduction='mean'
        )
        
        gradient_loss = F.mse_loss(
            gradient_recon[mask_expanded],
            gradient[mask_expanded],
            reduction='mean'
        )
        
        # Only compute second derivative loss if enabled
        if self.use_second_deriv and second_deriv_recon is not None:
            second_deriv_loss = F.mse_loss(
                second_deriv_recon[mask_expanded],
                second_deriv[mask_expanded],
                reduction='mean'
            )
            # Weighted combination
            total_loss = (
                self.waveform_loss_weight * waveform_loss +
                self.gradient_loss_weight * gradient_loss +
                self.second_deriv_loss_weight * second_deriv_loss
            )
        else:
            second_deriv_loss = torch.tensor(0.0, device=waveform.device)
            # Weighted combination without second derivative
            total_loss = (
                self.waveform_loss_weight * waveform_loss +
                self.gradient_loss_weight * gradient_loss
            )
        
        return {
            'total_loss': total_loss,
            'waveform_loss': waveform_loss,
            'gradient_loss': gradient_loss,
            'second_deriv_loss': second_deriv_loss
        }
    
    def training_step(self, batch, batch_idx):
        waveform = batch['waveform']
        gradient = batch['gradient']
        second_deriv = batch['second_deriv']
        detector = batch.get('detector', None)
        
        # Forward pass
        waveform_recon, gradient_recon, second_deriv_recon, mask = self.model(
            waveform, gradient, second_deriv, detector=detector
        )
        
        # Compute loss
        losses = self._compute_loss(
            (waveform_recon, gradient_recon, second_deriv_recon),
            batch,
            mask
        )
        
        # Log metrics
        self.log('train_loss', losses['total_loss'], prog_bar=True, sync_dist=True)
        self.log('train_waveform_loss', losses['waveform_loss'], sync_dist=True)
        self.log('train_gradient_loss', losses['gradient_loss'], sync_dist=True)
        self.log('train_second_deriv_loss', losses['second_deriv_loss'], sync_dist=True)
        
        # Log mask ratio
        mask_ratio_actual = (~mask).float().mean()
        self.log('train_mask_ratio', mask_ratio_actual, sync_dist=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        waveform = batch['waveform']
        gradient = batch['gradient']
        second_deriv = batch['second_deriv']
        detector = batch.get('detector', None)
        
        # Forward pass
        waveform_recon, gradient_recon, second_deriv_recon, mask = self.model(
            waveform, gradient, second_deriv, detector=detector
        )
        
        # Compute loss
        losses = self._compute_loss(
            (waveform_recon, gradient_recon, second_deriv_recon),
            batch,
            mask
        )
        
        # Log metrics
        self.log('val_loss', losses['total_loss'], prog_bar=True, sync_dist=True)
        self.log('val_waveform_loss', losses['waveform_loss'], sync_dist=True)
        self.log('val_gradient_loss', losses['gradient_loss'], sync_dist=True)
        self.log('val_second_deriv_loss', losses['second_deriv_loss'], sync_dist=True)
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        """
        Configure optimizer with proper weight decay (exclude bias/norm layers).
        Following MAE paper and timm's optim_factory.
        """
        # Separate parameters: apply weight decay only to weights, not bias/norm
        decay = set()
        no_decay = set()
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No decay for 1D params (bias, norm) and special parameters
            if param.ndim == 1 or 'bias' in name or 'norm' in name or 'mask_token' in name:
                no_decay.add(name)
            else:
                decay.add(name)
        
        param_dict = {name: param for name, param in self.model.named_parameters()}
        
        param_groups = [
            {'params': [param_dict[name] for name in sorted(decay)], 'weight_decay': self.weight_decay},
            {'params': [param_dict[name] for name in sorted(no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        
        def warmup_cosine_schedule(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def save_encoder_weights(self, filepath):
        """
        Save only encoder weights for fine-tuning.
        
        Args:
            filepath: path to save encoder checkpoint
        """
        encoder_dict = self.model.get_encoder_for_finetuning()
        torch.save({
            'encoder_state_dict': encoder_dict,
            'hparams': self.hparams
        }, filepath)
        print(f"Saved encoder weights to {filepath}")
