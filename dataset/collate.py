import torch
import numpy as np

def majorana_collate_fn(batch):
    waveforms = torch.stack([item['waveform'] for item in batch])
    gradients = torch.stack([item['gradient'] for item in batch])
    second_derivs = torch.stack([item['second_deriv'] for item in batch])
    
    collated = {
        'waveform': waveforms,
        'gradient': gradients,
        'second_deriv': second_derivs,
    }
    
    # Only include energy_label if ALL items in batch have it
    if all('energy_label' in item for item in batch):
        energy_labels = torch.stack([item['energy_label'] for item in batch])
        collated['energy_label'] = energy_labels
    
    scalar_keys = ['detector', 'id', 'run_number', 'tp0']
    for key in scalar_keys:
        if key in batch[0]:
            values = [item[key] for item in batch]
            if isinstance(values[0], (np.ndarray, np.generic)):
                values = [v.item() if hasattr(v, 'item') else v for v in values]
            collated[key] = torch.tensor(values)
    
    label_keys = ['psd_label_dcr', 'psd_label_high_avse', 'psd_label_low_avse', 'psd_label_lq']
    for key in label_keys:
        if all(key in item for item in batch):
            values = [item[key] for item in batch]
            if isinstance(values[0], (np.ndarray, np.generic)):
                values = [v.item() if hasattr(v, 'item') else v for v in values]
            collated[key] = torch.tensor(values, dtype=torch.long)
    
    if 'filename' in batch[0]:
        collated['filename'] = [item['filename'] for item in batch]
    
    return collated
