import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class TransformerEncoderLayerWithDropPath(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, drop_path=0.0, activation='gelu'):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        self.activation = F.gelu if activation == 'gelu' else F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.drop_path(self.dropout1(src2))
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.drop_path(self.dropout2(src2))
        return src

class WaveformTransformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        drop_path=0.0,
        pooling='gap',
        predict_energy=True,
        window_size=10,
        learned_pe=False,
        use_detector=False,
        use_second_deriv=False,
    ):
        super().__init__()
        
        assert pooling in ['gap', 'cls'], "pooling must be 'gap' or 'cls'"
        self.pooling = pooling
        self.predict_energy = predict_energy
        self.learned_pe = learned_pe
        self.use_detector = use_detector
        self.use_second_deriv = use_second_deriv

        if self.use_detector:
            # Expect pre-mapped detector indices [0..25] from dataset
            self.num_detectors = 26

            # FiLM params: gamma + beta per detector
            self.detector_film = nn.Embedding(self.num_detectors, 2 * d_model)
            nn.init.zeros_(self.detector_film.weight)

            # Per-detector energy calibration: scale + shift
            self.energy_affine = nn.Embedding(self.num_detectors, 2)
            with torch.no_grad():
                self.energy_affine.weight[:, 0].fill_(1.0)  # scale
                self.energy_affine.weight[:, 1].fill_(0.0)  # shift
        
        self.waveform_projection = nn.Linear(window_size, d_model)
        self.gradient_projection = nn.Linear(window_size, d_model)
        if use_second_deriv:
            self.second_deriv_projection = nn.Linear(window_size, d_model)
        
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        if learned_pe:
            self.pos_encoding = LearnedPositionalEncoding(d_model, dropout=dropout, max_len=5000)
        else:
            self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=5000)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.transformer = nn.ModuleList([
            TransformerEncoderLayerWithDropPath(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                drop_path=dpr[i],
                activation='gelu'
            )
            for i in range(num_layers)
        ])
        
        self.pooled_norm = nn.LayerNorm(d_model)
        self.pooled_dropout = nn.Dropout(dropout)
        
        self.psd_classifiers = nn.ModuleDict({
            'dcr': nn.Linear(d_model, 1),
            'high_avse': nn.Linear(d_model, 1),
            'low_avse': nn.Linear(d_model, 1),
            'lq': nn.Linear(d_model, 1)
        })
        
        if predict_energy:
            self.energy_head = nn.Linear(d_model, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'psd_classifiers' in name or 'energy_head' in name:
                    nn.init.xavier_uniform_(module.weight, gain=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Parameter) and 'cls_token' in name:
                nn.init.trunc_normal_(module, std=0.02)
    
    def forward(self, waveform, gradient, second_deriv, detector=None):
        batch_size = waveform.size(0)
        
        wf_embed = self.waveform_projection(waveform)
        grad_embed = self.gradient_projection(gradient)
        
        # Combine features
        if self.use_second_deriv:
            sd_embed = self.second_deriv_projection(second_deriv)
            x = wf_embed + grad_embed + sd_embed
        else:
            x = wf_embed + grad_embed
        x = self.input_norm(x)
        if self.use_detector:
            det_idx = detector.long().to(x.device)                     # (B,)
            film = self.detector_film(det_idx)                         # (B, 2*d_model)
            gamma, beta = film.chunk(2, dim=-1)                        # (B, d_model), (B, d_model)
            x = x * (1.0 + gamma[:, None, :]) + beta[:, None, :]       # (B, T, d_model)
        x = self.input_dropout(x)
        
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.pos_encoding(x)
        for layer in self.transformer:
            x = layer(x)
        
        if self.pooling == 'cls':
            pooled = x[:, 0]
        else:
            pooled = x.mean(dim=1)
        
        pooled = self.pooled_norm(pooled)
        pooled = self.pooled_dropout(pooled)
        
        psd_outputs = {
            'psd_dcr': self.psd_classifiers['dcr'](pooled),
            'psd_high_avse': self.psd_classifiers['high_avse'](pooled),
            'psd_low_avse': self.psd_classifiers['low_avse'](pooled),
            'psd_lq': self.psd_classifiers['lq'](pooled)
        }
        
        outputs = psd_outputs
        
        if self.predict_energy:
            e_raw = self.energy_head(pooled)  # (B, 1)
            if self.use_detector:
                # Detector is already pre-mapped to index [0..25] by dataset
                det_idx = detector.long().to(pooled.device)
                ab = self.energy_affine(det_idx)  # (B, 2)
                a = ab[:, 0:1]
                b = ab[:, 1:2]
                e = a * e_raw + b
            else:
                e = e_raw
            outputs["energy"] = e
        
        return outputs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        nn.init.trunc_normal_(self.pe, std=0.02)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
