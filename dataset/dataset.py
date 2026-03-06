import random
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

class MajoranaDataset(Dataset):
    def __init__(self, root, window_size=1, constants=None, to_torch=False, test=False, 
                 normalize_energy=False, pattern_filter=None, shift_augmentation=0, 
                 shift_pad_mode='edge', use_second_deriv=False, **kwargs):
        self.root = root
        self.pattern_filter = pattern_filter
        self.data_files = self.processed_file_names
        self.total_events = len(self.data_files)
        self.window_size = window_size
        self.constants = constants
        self.to_torch = to_torch
        self.test = test
        self.normalize_energy = normalize_energy
        self.shift_augmentation = shift_augmentation
        self.shift_pad_mode = shift_pad_mode
        self.use_second_deriv = use_second_deriv

    @property
    def processed_dir(self):
        return f'{self.root}'

    @property
    def processed_file_names(self):
        all_files = sorted(glob(f'{self.processed_dir}/**/*.npz', recursive=True), key=lambda x: x.lower())
        if self.pattern_filter:
            all_files = [f for f in all_files if self.pattern_filter in f]
        return all_files

    def __len__(self):
        return self.total_events

    def __getitem__(self, idx):
        data = np.load(self.data_files[idx])
        
        if not self.to_torch:
            return data

        waveform = data['waveform']
        tp0 = int(data.get('tp0', 1000))
        
        # Baseline subtraction FIRST (using consistent pre-trigger region)
        baseline = waveform[:tp0].mean()
        waveform = waveform - baseline
        
        # Apply shift augmentation AFTER baseline subtraction
        if self.shift_augmentation > 0 and not self.test:
            # Random shift in range [-shift_augmentation, +shift_augmentation]
            shift = np.random.randint(-self.shift_augmentation, self.shift_augmentation + 1)
            if shift != 0:
                if shift > 0:
                    # Shift right: pad at the beginning
                    if self.shift_pad_mode == 'edge':
                        pad_value = waveform[0]
                        waveform = np.concatenate([np.full(shift, pad_value), waveform[:-shift]])
                    else:  # zero padding
                        waveform = np.concatenate([np.zeros(shift), waveform[:-shift]])
                else:  # shift < 0
                    # Shift left: pad at the end
                    shift_abs = abs(shift)
                    if self.shift_pad_mode == 'edge':
                        pad_value = waveform[-1]
                        waveform = np.concatenate([waveform[shift_abs:], np.full(shift_abs, pad_value)])
                    else:  # zero padding
                        waveform = np.concatenate([waveform[shift_abs:], np.zeros(shift_abs)])

        if self.constants is not None:
            waveform = (waveform - self.constants.WAVEFORM_MEAN) / self.constants.WAVEFORM_STD

        gradient = np.gradient(waveform)
        
        if self.use_second_deriv:
            second_deriv = np.gradient(gradient)
            if self.constants is not None:
                gradient = (gradient - self.constants.GRADIENT_MEAN) / self.constants.GRADIENT_STD
                second_deriv = (second_deriv - self.constants.SECOND_DERIV_MEAN) / self.constants.SECOND_DERIV_STD
        else:
            # Use zeros as placeholder when second derivative is disabled
            second_deriv = np.zeros_like(gradient)
            if self.constants is not None:
                gradient = (gradient - self.constants.GRADIENT_MEAN) / self.constants.GRADIENT_STD

        if self.window_size > 1:
            waveform = waveform.reshape(-1, self.window_size)
            gradient = gradient.reshape(-1, self.window_size)
            second_deriv = second_deriv.reshape(-1, self.window_size)
        else:
            waveform = waveform.reshape(-1, 1)
            gradient = gradient.reshape(-1, 1)
            second_deriv = second_deriv.reshape(-1, 1)

        waveform = torch.from_numpy(waveform).float()
        gradient = torch.from_numpy(gradient).float()
        second_deriv = torch.from_numpy(second_deriv).float()

        output = {
            'waveform': waveform,
            'gradient': gradient,
            'second_deriv': second_deriv,
        }

        if 'energy_label' in data:
            energy_label = float(data['energy_label'])
            if self.normalize_energy and self.constants is not None:
                energy_label = (energy_label - self.constants.ENERGY_MEAN) / self.constants.ENERGY_STD
            output['energy_label'] = torch.tensor([energy_label]).float()

        # Convert detector ID to index if present
        if 'detector' in data:
            detector_id = int(data['detector'])
            if self.constants is not None:
                detector_idx = self.constants.detector_to_index(detector_id)
                output['detector'] = detector_idx
            else:
                output['detector'] = detector_id
        
        for key in ['id', 'psd_label_dcr', 'psd_label_high_avse', 
                    'psd_label_low_avse', 'psd_label_lq', 'run_number', 'tp0']:
            if key in data:
                output[key] = data[key]

        if self.test:
            output['filename'] = self.data_files[idx]

        return output