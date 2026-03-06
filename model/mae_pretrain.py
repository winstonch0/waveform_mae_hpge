"""
Masked Autoencoder (MAE) style pre-training for Majorana waveforms.
Learns representations by reconstructing masked time windows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import WaveformTransformer, PositionalEncoding, LearnedPositionalEncoding, TransformerEncoderLayerWithDropPath


class MAEWaveformEncoder(nn.Module):
    """
    MAE encoder that processes only visible (non-masked) tokens.
    Based on WaveformTransformer but without task-specific heads.
    """
    def __init__(
        self,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        window_size=10,
        learned_pe=False,
        use_second_deriv=False,
        mask_ratio=0.25,
        use_detector=False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.window_size = window_size
        self.learned_pe = learned_pe
        self.use_second_deriv = use_second_deriv
        self.mask_ratio = mask_ratio
        self.use_detector = use_detector
        
        if self.use_detector:
            # Expect pre-mapped detector indices [0..25] from dataset
            self.num_detectors = 26
            # FiLM params: gamma + beta per detector
            self.detector_film = nn.Embedding(self.num_detectors, 2 * d_model)
            nn.init.zeros_(self.detector_film.weight)
        
        # Input projections (same as main model)
        self.waveform_projection = nn.Linear(window_size, d_model)
        self.gradient_projection = nn.Linear(window_size, d_model)
        if use_second_deriv:
            self.second_deriv_projection = nn.Linear(window_size, d_model)
        
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # Positional encoding
        if learned_pe:
            self.pos_encoding = LearnedPositionalEncoding(d_model, dropout=dropout, max_len=5000)
        else:
            self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=5000)
        
        # Transformer encoder
        self.transformer = nn.ModuleList([
            TransformerEncoderLayerWithDropPath(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                drop_path=0.0,
                activation='gelu'
            )
            for _ in range(num_layers)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, waveform, gradient, second_deriv, mask=None, detector=None):
        """
        Args:
            waveform: (batch_size, seq_len, window_size)
            gradient: (batch_size, seq_len, window_size)
            second_deriv: (batch_size, seq_len, window_size)
            mask: (batch_size, seq_len) boolean mask (True = keep, False = masked)
            detector: (batch_size,) detector indices for FiLM conditioning
        
        Returns:
            embeddings: (batch_size, visible_seq_len, d_model)
            ids_restore: (batch_size, seq_len) indices to restore original order
        """
        batch_size, seq_len, _ = waveform.shape
        
        # Project inputs
        wf_embed = self.waveform_projection(waveform)
        grad_embed = self.gradient_projection(gradient)
        
        # Combine features
        if self.use_second_deriv:
            sd_embed = self.second_deriv_projection(second_deriv)
            x = wf_embed + grad_embed + sd_embed
        else:
            x = wf_embed + grad_embed
        x = self.input_norm(x)
        
        # Apply detector-specific FiLM conditioning if enabled
        if self.use_detector and detector is not None:
            det_idx = detector.long().to(x.device)  # (B,)
            film = self.detector_film(det_idx)  # (B, 2*d_model)
            gamma, beta = film.chunk(2, dim=-1)  # (B, d_model), (B, d_model)
            x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]  # (B, T, d_model)
        
        x = self.input_dropout(x)
        
        # Apply masking if provided
        if mask is not None:
            # Keep only visible tokens - use static number based on mask ratio
            ids_shuffle = torch.argsort(mask.float(), dim=1, descending=True)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            # Use fixed number of visible tokens (static computation, no .item())
            num_visible = int(seq_len * (1 - self.mask_ratio))
            
            # Gather visible tokens
            x_visible = torch.gather(
                x, dim=1, 
                index=ids_shuffle[:, :num_visible].unsqueeze(-1).expand(-1, -1, self.d_model)
            )
            
            # Add positional encoding to visible tokens only
            x_visible = self.pos_encoding(x_visible)
            
            # Pass through transformer
            for layer in self.transformer:
                x_visible = layer(x_visible)
            
            return x_visible, ids_restore
        else:
            # No masking - process all tokens
            x = self.pos_encoding(x)
            for layer in self.transformer:
                x = layer(x)
            return x, None


class MAEWaveformDecoder(nn.Module):
    """
    Lightweight decoder to reconstruct masked tokens.
    """
    def __init__(
        self,
        d_model=128,
        decoder_d_model=64,
        decoder_num_layers=2,
        nhead=8,
        dropout=0.1,
        window_size=10,
        use_second_deriv=False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.decoder_d_model = decoder_d_model
        self.window_size = window_size
        self.use_second_deriv = use_second_deriv
        
        # Project encoder output to decoder dimension
        self.encoder_to_decoder = nn.Linear(d_model, decoder_d_model)
        
        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Positional encoding for decoder
        self.pos_encoding = PositionalEncoding(decoder_d_model, dropout=dropout, max_len=5000)
        
        # Decoder transformer
        self.decoder = nn.ModuleList([
            TransformerEncoderLayerWithDropPath(
                d_model=decoder_d_model,
                nhead=nhead,
                dim_feedforward=decoder_d_model * 4,
                dropout=dropout,
                drop_path=0.0,
                activation='gelu'
            )
            for _ in range(decoder_num_layers)
        ])
        
        # Reconstruction heads
        self.waveform_head = nn.Linear(decoder_d_model, window_size)
        self.gradient_head = nn.Linear(decoder_d_model, window_size)
        if use_second_deriv:
            self.second_deriv_head = nn.Linear(decoder_d_model, window_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x_encoded, ids_restore):
        """
        Args:
            x_encoded: (batch_size, visible_seq_len, d_model) encoder output
            ids_restore: (batch_size, seq_len) indices to restore original order
        
        Returns:
            waveform_recon: (batch_size, seq_len, window_size)
            gradient_recon: (batch_size, seq_len, window_size)
            second_deriv_recon: (batch_size, seq_len, window_size)
        """
        batch_size = x_encoded.shape[0]
        seq_len = ids_restore.shape[1]
        
        # Project to decoder dimension
        x = self.encoder_to_decoder(x_encoded)
        
        # Add mask tokens for masked positions
        mask_tokens = self.mask_token.expand(batch_size, seq_len - x.shape[1], -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        
        # Restore original order
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, self.decoder_d_model)
        )
        
        # Add positional encoding
        x_full = self.pos_encoding(x_full)
        
        # Decode
        for layer in self.decoder:
            x_full = layer(x_full)
        
        # Reconstruct
        waveform_recon = self.waveform_head(x_full)
        gradient_recon = self.gradient_head(x_full)
        if self.use_second_deriv:
            second_deriv_recon = self.second_deriv_head(x_full)
        else:
            # Return None when not using second derivatives
            second_deriv_recon = None
        
        return waveform_recon, gradient_recon, second_deriv_recon


class MAEWaveformModel(nn.Module):
    """
    Complete MAE model for waveform pre-training.
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
        use_second_deriv=False,
        use_detector=False
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.use_second_deriv = use_second_deriv
        self.use_detector = use_detector
        
        self.encoder = MAEWaveformEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            window_size=window_size,
            learned_pe=learned_pe,
            use_second_deriv=use_second_deriv,
            mask_ratio=mask_ratio,
            use_detector=use_detector
        )
        
        self.decoder = MAEWaveformDecoder(
            d_model=d_model,
            decoder_d_model=decoder_d_model,
            decoder_num_layers=decoder_num_layers,
            nhead=nhead,
            dropout=dropout,
            window_size=window_size,
            use_second_deriv=use_second_deriv
        )
    
    def random_masking(self, x, mask_ratio):
        """
        Random masking - fully vectorized for torch.compile compatibility.
        No .item() calls or dynamic Python loops.
        
        Args:
            x: (batch_size, seq_len, window_size)
            mask_ratio: fraction of tokens to mask
        
        Returns:
            mask: (batch_size, seq_len) boolean (True = keep, False = mask)
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate random noise for each position
        noise = torch.rand(batch_size, seq_len, device=x.device)
        
        # Sort by noise to get random shuffling (argsort is differentiable)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Calculate number of tokens to keep (static per batch, not dynamic per sample)
        num_keep = int(seq_len * (1 - mask_ratio))
        
        # Create mask: True for keep, False for mask
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_shuffle[:, :num_keep], True)
        
        return mask
    
    def forward(self, waveform, gradient, second_deriv, mask_ratio=None, detector=None):
        """
        Args:
            waveform: (batch_size, seq_len, window_size)
            gradient: (batch_size, seq_len, window_size)
            second_deriv: (batch_size, seq_len, window_size)
            mask_ratio: override default mask ratio
            detector: (batch_size,) detector indices for FiLM conditioning
        
        Returns:
            waveform_recon: (batch_size, seq_len, window_size)
            gradient_recon: (batch_size, seq_len, window_size)
            second_deriv_recon: (batch_size, seq_len, window_size)
            mask: (batch_size, seq_len) boolean mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Generate mask
        mask = self.random_masking(waveform, mask_ratio)
        
        # Encode visible tokens
        x_encoded, ids_restore = self.encoder(waveform, gradient, second_deriv, mask, detector)
        
        # Decode all tokens
        waveform_recon, gradient_recon, second_deriv_recon = self.decoder(x_encoded, ids_restore)
        
        return waveform_recon, gradient_recon, second_deriv_recon, mask
    
    def get_encoder_for_finetuning(self):
        """
        Extract encoder parameters and buffers to initialize downstream model.
        
        Returns:
            state_dict: encoder state dict compatible with WaveformTransformer
        """
        encoder_dict = {}
        
        # Map encoder parameters to WaveformTransformer naming
        for name, param in self.encoder.named_parameters():
            # Remove 'encoder.' prefix if present
            new_name = name.replace('encoder.', '')
            encoder_dict[new_name] = param.data.clone()
        
        # Also save buffers (e.g., positional encoding pe)
        for name, buffer in self.encoder.named_buffers():
            new_name = name.replace('encoder.', '')
            encoder_dict[new_name] = buffer.clone()
        
        return encoder_dict
