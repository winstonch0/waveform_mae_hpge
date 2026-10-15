import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from model.mae_pretrain import MAEWaveformEncoder


torch.manual_seed(0)

seq_len = 10
d_model = 8
mask_ratio = 0.5

encoder = MAEWaveformEncoder(
    d_model=d_model,
    nhead=2,
    num_layers=1,
    dropout=0.0,
    window_size=1,
    learned_pe=False,
    mask_ratio=mask_ratio,
    use_detector=False,
)

encoder.eval()

# Remove waveform/gradient content so the input to the transformer
# consists only of positional information.
with torch.no_grad():
    encoder.waveform_projection.weight.zero_()
    encoder.waveform_projection.bias.zero_()
    encoder.gradient_projection.weight.zero_()
    encoder.gradient_projection.bias.zero_()

waveform = torch.zeros(1, seq_len, 1)
gradient = torch.zeros(1, seq_len, 1)
second_deriv = torch.zeros(1, seq_len, 1)

# True = visible
mask = torch.tensor([
    [
        True, False, True, False, False,
        True, False, True, False, True
    ]
])

# Capture what is actually sent into the first transformer layer.
captured = {}

def capture_transformer_input(module, inputs):
    captured["x"] = inputs[0].detach().clone()

hook = encoder.transformer[0].register_forward_pre_hook(
    capture_transformer_input
)

with torch.no_grad():
    encoder(
        waveform,
        gradient,
        second_deriv,
        mask=mask
    )

hook.remove()

actual = captured["x"]

ids_shuffle = torch.argsort(
    mask.float(),
    dim=1,
    descending=True
)

num_visible = int(seq_len * (1 - mask_ratio))
visible_positions = ids_shuffle[:, :num_visible]

# Positional encodings corresponding to the TRUE original positions
pe = encoder.pos_encoding.pe

expected_original = torch.gather(
    pe.expand(1, -1, -1),
    dim=1,
    index=visible_positions.unsqueeze(-1).expand(
        -1, -1, d_model
    )
)

# Positional encodings assigned by the compressed sequence
expected_compact = pe[:, :num_visible, :]

error_original = (
    actual - expected_original
).abs().max().item()

error_compact = (
    actual - expected_compact
).abs().max().item()

print("Mask:")
print(mask)

print("\nVisible tokens actually gathered from positions:")
print(visible_positions)

print("\nEncoder sequence length:")
print(actual.shape[1])

print("\nMax error vs ORIGINAL positions:")
print(error_original)

print("\nMax error vs COMPACT positions:")
print(error_compact)

if error_original < 1e-6:
    print("\nPASS: encoder retains original temporal positions.")
else:
    print("\nFAIL: encoder does NOT retain original temporal positions.")
