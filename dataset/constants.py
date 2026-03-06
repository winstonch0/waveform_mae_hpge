import json
from pathlib import Path
import numpy as np

class DatasetConstants:
    # Detector ID to index mapping
    DETECTOR_IDS = [112, 113, 114, 122, 123, 132, 133, 134, 152, 153, 161, 163, 164, 172, 173, 174,
                    212, 213, 214, 231, 232, 253, 254, 261, 262, 273]
    
    # Create lookup table: raw detector id -> index in [0..25], else -1
    _max_detector_id = max(DETECTOR_IDS)
    DETECTOR_LUT = np.full(_max_detector_id + 1, -1, dtype=np.int32)
    for _i, _did in enumerate(DETECTOR_IDS):
        DETECTOR_LUT[_did] = _i
    
    @staticmethod
    def detector_to_index(detector_id):
        """Convert raw detector ID to index [0..25], or raise ValueError if unknown."""
        if detector_id < 0 or detector_id >= len(DatasetConstants.DETECTOR_LUT):
            raise ValueError(f"Detector id out of range: {detector_id}")
        idx = DatasetConstants.DETECTOR_LUT[detector_id]
        if idx < 0:
            raise ValueError(f"Unknown detector id: {detector_id}")
        return int(idx)
    
    def __init__(self, stats_file='dataset/stats.json'):
        stats_path = Path(stats_file)
        
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Statistics file not found: {stats_file}\n"
                f"Please run: python dataset/compute_stats.py"
            )
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        self.WAVEFORM_MEAN = stats['waveform']['mean']
        self.WAVEFORM_STD = stats['waveform']['std']
        self.WAVEFORM_MIN = stats['waveform']['min']
        self.WAVEFORM_MAX = stats['waveform']['max']
        
        self.GRADIENT_MEAN = stats['gradient']['mean']
        self.GRADIENT_STD = stats['gradient']['std']
        
        self.SECOND_DERIV_MEAN = stats.get('second_deriv', {}).get('mean', 0.0)
        self.SECOND_DERIV_STD = stats.get('second_deriv', {}).get('std', 1.0)
        
        if 'energy' in stats:
            self.ENERGY_MEAN = stats['energy']['mean']
            self.ENERGY_STD = stats['energy']['std']
            self.ENERGY_MIN = stats['energy']['min']
            self.ENERGY_MAX = stats['energy']['max']
            self.ENERGY_MEDIAN = stats['energy']['median']
        else:
            self.ENERGY_MEAN = None
            self.ENERGY_STD = None
            self.ENERGY_MIN = None
            self.ENERGY_MAX = None
            self.ENERGY_MEDIAN = None
    
    def __repr__(self):
        energy_str = f"{self.ENERGY_MEAN:.6f}" if self.ENERGY_MEAN is not None else "None"
        return (f"DatasetConstants(\n"
                f"  WAVEFORM_MEAN={self.WAVEFORM_MEAN:.6f},\n"
                f"  WAVEFORM_STD={self.WAVEFORM_STD:.6f},\n"
                f"  GRADIENT_MEAN={self.GRADIENT_MEAN:.6f},\n"
                f"  GRADIENT_STD={self.GRADIENT_STD:.6f},\n"
                f"  SECOND_DERIV_STD={self.SECOND_DERIV_STD:.6f},\n"
                f"  ENERGY_MEAN={energy_str}\n"
                f")")
