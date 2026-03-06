# Majorana Dataset

PyTorch dataset implementation for the Majorana Demonstrator waveform data.

## Setup

### 1. Compute Statistics

Before training, compute normalization statistics from the training data:

```bash
python dataset/compute_stats.py --data_root data/waveform_npz --output dataset/stats.json
```

For quick testing with fewer files:
```bash
python dataset/compute_stats.py --max_files 1000
```

This creates `dataset/stats.json` with mean, std, min, max for waveforms, gradients, and energy labels.

## Usage

### Basic Dataset

```python
from dataset import MajoranaDataset, DatasetConstants

constants = DatasetConstants('dataset/stats.json')

train_dataset = MajoranaDataset(
    root='data/waveform_npz',
    window_size=1,
    constants=constants,
    to_torch=True,
    normalize_energy=True
)

sample = train_dataset[0]
```

### DataLoader

```python
from torch.utils.data import DataLoader
from dataset import MajoranaDataset, DatasetConstants, majorana_collate_fn

constants = DatasetConstants()

dataset = MajoranaDataset(
    root='data/waveform_npz',
    constants=constants,
    to_torch=True
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=majorana_collate_fn,
    num_workers=4
)
```

### Quick Start with Helper

```python
from dataset.dataloader import get_dataloaders

train_loader, test_loader = get_dataloaders(
    train_root='data/waveform_npz',
    test_root='data/waveform_npz/MJD_Test_0',
    window_size=1,
    batch_size=32,
    normalize_energy=True
)
```

## Dataset Output

Each sample contains:

**Always present:**
- `waveform`: (seq_len, window_size) normalized waveform
- `gradient`: (seq_len, window_size) gradient of waveform

**When available (missing in NPML files):**
- `energy_label`: (1,) energy measurement
- `psd_label_dcr`: pulse shape discrimination label
- `psd_label_high_avse`: high A/E PSD label
- `psd_label_low_avse`: low A/E PSD label
- `psd_label_lq`: linear gate PSD label

**Metadata (when available):**
- `detector`: detector ID
- `id`: event ID
- `run_number`: run number
- `tp0`: time parameter

## Parameters

**MajoranaDataset:**
- `root`: Path to dataset directory
- `window_size`: Window size for reshaping sequences (default: 1)
- `constants`: DatasetConstants object for normalization
- `to_torch`: Return torch tensors vs raw numpy (default: False)
- `test`: Include filename in output (default: False)
- `normalize_energy`: Apply normalization to energy labels (default: False)
- `shift_augmentation`: Maximum random temporal shift in timesteps (default: 0, disabled). Applied during training only for regularization.
- `shift_pad_mode`: Padding mode for shifted waveforms: 'edge' (repeat edge values) or 'zero' (default: 'edge')

### Shift Augmentation

The `shift_augmentation` parameter enables temporal shift augmentation for improved regularization. When enabled (e.g., `shift_augmentation=10`), each waveform is randomly shifted by -N to +N timesteps during training:

- **Positive shift (right)**: Pads at the beginning, removes from end
- **Negative shift (left)**: Removes from beginning, pads at end
- **Padding**: Uses edge values (`shift_pad_mode='edge'`) or zeros (`shift_pad_mode='zero'`)
- **Only during training**: Disabled automatically when `test=True`

Example:
```python
train_dataset = MajoranaDataset(
    root='data/waveform_npz',
    shift_augmentation=10,  # Random shift ±10 timesteps
    shift_pad_mode='edge',   # Repeat edge values for padding
    constants=constants,
    to_torch=True
)
```

## Data Structure

Files are organized as:
```
data/waveform_npz/
├── MJD_Train_0/
│   ├── MJD_Train_0_0000000.npz
│   └── ...
├── MJD_Train_1/
├── MJD_Test_0/
└── MJD_NPML_0/
```

Each `.npz` file contains waveform data and available metadata fields.
