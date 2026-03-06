import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from .transformer import WaveformTransformer

class LitWaveformTransformer(pl.LightningModule):
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        drop_path=0.0,
        pooling='gap',
        predict_energy=True,
        lr=1e-4,
        weight_decay=1e-5,
        beta1=0.9,
        beta2=0.999,
        warmup_epochs=5,
        max_epochs=100,
        psd_loss_weight=1.0,
        energy_loss_weight=1.0,
        pos_weight=None,
        window_size=1,
        learned_pe=False,
        label_smoothing=0.0,
        use_ema=False,
        ema_decay=0.999,
        layer_decay=1.0,
        use_detector=False,
        use_second_deriv=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.label_smoothing = label_smoothing
        self.use_ema = use_ema
        self.layer_decay = layer_decay
        
        self.model = WaveformTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            drop_path=drop_path,
            pooling=pooling,
            predict_energy=predict_energy,
            window_size=window_size,
            learned_pe=learned_pe,
            use_detector=use_detector,
            use_second_deriv=use_second_deriv
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.psd_loss_weight = psd_loss_weight
        self.energy_loss_weight = energy_loss_weight
        
        if pos_weight is not None and len(pos_weight) > 0:
            self.register_buffer('pos_weight_dcr', torch.tensor(pos_weight.get('dcr', 1.0)))
            self.register_buffer('pos_weight_high_avse', torch.tensor(pos_weight.get('high_avse', 1.0)))
            self.register_buffer('pos_weight_low_avse', torch.tensor(pos_weight.get('low_avse', 1.0)))
            self.register_buffer('pos_weight_lq', torch.tensor(pos_weight.get('lq', 1.0)))
        else:
            self.pos_weight_dcr = None
            self.pos_weight_high_avse = None
            self.pos_weight_low_avse = None
            self.pos_weight_lq = None
    
    def forward(self, waveform, gradient, second_deriv, detector=None):
        return self.model(waveform, gradient, second_deriv, detector=detector)
    
    def _compute_loss(self, outputs, batch):
        losses = {}
        total_loss = 0
        
        psd_labels = ['dcr', 'high_avse', 'low_avse', 'lq']
        pos_weights = [self.pos_weight_dcr, self.pos_weight_high_avse, 
                      self.pos_weight_low_avse, self.pos_weight_lq]
        
        for label, pos_weight in zip(psd_labels, pos_weights):
            if f'psd_label_{label}' in batch:
                target = batch[f'psd_label_{label}'].float().unsqueeze(1)
                pred = outputs[f'psd_{label}']
                
                if self.label_smoothing > 0:
                    target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
                
                if pos_weight is not None:
                    loss = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
                else:
                    loss = F.binary_cross_entropy_with_logits(pred, target)
                
                losses[f'psd_{label}_loss'] = loss
                total_loss += self.psd_loss_weight * loss
        
        if 'energy' in outputs and 'energy_label' in batch:
            energy_pred = outputs['energy'].squeeze(1)
            energy_target = batch['energy_label'].squeeze(1)
            energy_loss = F.huber_loss(energy_pred, energy_target, delta=1.0)
            losses['energy_loss'] = energy_loss
            total_loss += self.energy_loss_weight * energy_loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def _compute_metrics(self, outputs, batch, prefix=''):
        metrics = {}
        
        psd_labels = ['dcr', 'high_avse', 'low_avse', 'lq']
        for label in psd_labels:
            if f'psd_label_{label}' in batch:
                target = batch[f'psd_label_{label}']
                pred = torch.sigmoid(outputs[f'psd_{label}']).squeeze(1) > 0.5
                acc = (pred == target).float().mean()
                metrics[f'{prefix}psd_{label}_acc'] = acc
        
        if 'energy' in outputs and 'energy_label' in batch:
            energy_pred = outputs['energy'].squeeze(1)
            energy_target = batch['energy_label'].squeeze(1)
            mae = F.l1_loss(energy_pred, energy_target)
            metrics[f'{prefix}energy_mae'] = mae
        
        return metrics
    
    def training_step(self, batch, batch_idx):
        waveform = batch['waveform']
        gradient = batch['gradient']
        second_deriv = batch['second_deriv']
        detector = batch['detector']
        
        outputs = self(waveform, gradient, second_deriv, detector)
        losses = self._compute_loss(outputs, batch)
        metrics = self._compute_metrics(outputs, batch, prefix='train_')
        
        self.log('train_loss', losses['total_loss'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for key, value in losses.items():
            if key != 'total_loss':
                self.log(f'train_{key}', value, on_step=False, on_epoch=True, sync_dist=True)
        
        for key, value in metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, sync_dist=True)
        
        return losses['total_loss']
    
    def validation_step(self, batch, batch_idx):
        waveform = batch['waveform']
        gradient = batch['gradient']
        second_deriv = batch['second_deriv']
        detector = batch['detector']
        
        outputs = self(waveform, gradient, second_deriv, detector)
        losses = self._compute_loss(outputs, batch)
        metrics = self._compute_metrics(outputs, batch, prefix='val_')
        
        self.log('val_loss', losses['total_loss'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for key, value in losses.items():
            if key != 'total_loss':
                self.log(f'val_{key}', value, on_step=False, on_epoch=True, sync_dist=True)
        
        for key, value in metrics.items():
            self.log(key, value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        """
        Configure optimizer with layer-wise learning rate decay (from MAE paper).
        Deeper layers get lower learning rates for stable fine-tuning.
        """
        # Separate parameters into groups with layer-wise LR decay
        if self.layer_decay < 1.0:
            param_groups = self._get_layer_wise_param_groups()
            print(f"Using layer-wise LR decay: {self.layer_decay:.3f}")
        else:
            param_groups = self._get_standard_param_groups()
        
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                return 1.0
        
        warmup_scheduler = LambdaLR(optimizer, lr_lambda)
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs - self.warmup_epochs,
            eta_min=self.lr * 0.01
        )
        
        scheduler = {
            'scheduler': SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs]
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def _get_layer_wise_param_groups(self):
        """
        Create parameter groups with layer-wise LR decay.
        Following MAE: https://github.com/facebookresearch/mae
        """
        # Get number of transformer layers
        num_layers = len(self.model.transformer)
        
        # Compute layer-wise LR scales
        layer_scales = [self.layer_decay ** (num_layers - i) for i in range(num_layers + 1)]
        
        param_groups = []
        
        # Group parameters by layer
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Determine layer ID
            layer_id = self._get_layer_id(name, num_layers)
            lr_scale = layer_scales[layer_id]
            
            # Determine weight decay
            if param.ndim == 1 or 'bias' in name or 'norm' in name:
                weight_decay = 0.0
            else:
                weight_decay = self.weight_decay
            
            # Add to appropriate group or create new one
            group_name = f"layer_{layer_id}_wd_{weight_decay}"
            
            # Find existing group or create new
            existing_group = None
            for group in param_groups:
                if group.get('lr_scale') == lr_scale and group.get('weight_decay') == weight_decay:
                    existing_group = group
                    break
            
            if existing_group is not None:
                existing_group['params'].append(param)
            else:
                param_groups.append({
                    'params': [param],
                    'lr_scale': lr_scale,
                    'weight_decay': weight_decay,
                    'layer_id': layer_id
                })
        
        return param_groups
    
    def _get_layer_id(self, name, num_layers):
        """
        Assign layer ID to parameter based on name.
        Following MAE/BEiT convention.
        """
        if 'pos_encoding' in name or 'input' in name:
            return 0  # Input layers
        elif 'waveform_projection' in name or 'gradient_projection' in name or 'second_deriv_projection' in name:
            return 0  # Projection layers
        elif 'transformer.' in name:
            # Extract layer number from name like "transformer.3.norm1.weight"
            # transformer is a ModuleList, so format is transformer.{layer_num}.{rest}
            try:
                layer_num = int(name.split('.')[1])
                return layer_num + 1
            except (ValueError, IndexError):
                return num_layers  # Default if parsing fails
        elif 'pooled' in name or 'psd_classifiers' in name or 'energy_head' in name:
            return num_layers  # Task heads get lowest layer_id (highest LR)
        else:
            return num_layers  # Default to task head LR
    
    def _get_standard_param_groups(self):
        """Standard parameter groups without layer-wise decay."""
        decay = set()
        no_decay = set()
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            if param.ndim <= 1 or name.endswith('.bias') or 'norm' in name.lower() or 'pe' in name or 'cls_token' in name:
                no_decay.add(name)
            else:
                decay.add(name)
        
        param_dict = {name: param for name, param in self.named_parameters()}
        
        optim_groups = [
            {'params': [param_dict[name] for name in sorted(decay)], 'weight_decay': self.weight_decay},
            {'params': [param_dict[name] for name in sorted(no_decay)], 'weight_decay': 0.0}
        ]
        
        return optim_groups
