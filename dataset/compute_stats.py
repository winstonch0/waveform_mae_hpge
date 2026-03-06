import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from glob import glob
import json

def compute_statistics(data_root, output_file='stats.json', max_files=None):
    train_files = sorted(glob(f'{data_root}/MJD_Train_*/**/*.npz', recursive=True))
    
    if max_files is not None:
        train_files = train_files[:max_files]
    
    print(f"Computing statistics from {len(train_files)} training files...")
    
    waveform_sum = 0.0
    waveform_sq_sum = 0.0
    waveform_count = 0
    waveform_min = float('inf')
    waveform_max = float('-inf')
    
    energy_values = []
    gradient_sum = 0.0
    gradient_sq_sum = 0.0
    gradient_count = 0
    
    second_deriv_sum = 0.0
    second_deriv_sq_sum = 0.0
    second_deriv_count = 0
    
    for file_path in tqdm(train_files, desc="Processing files"):
        try:
            data = np.load(file_path)
            
            waveform = data['waveform'].astype(np.float64)
            tp0 = int(data.get('tp0', 1000))
            
            baseline = waveform[:tp0].mean()
            waveform_baseline_subtracted = waveform - baseline
            
            waveform_sum += waveform_baseline_subtracted.sum()
            waveform_sq_sum += (waveform_baseline_subtracted ** 2).sum()
            waveform_count += waveform_baseline_subtracted.size
            waveform_min = min(waveform_min, waveform_baseline_subtracted.min())
            waveform_max = max(waveform_max, waveform_baseline_subtracted.max())
            
            gradient = np.gradient(waveform_baseline_subtracted.astype(np.float64))
            gradient_sum += gradient.sum()
            gradient_sq_sum += (gradient ** 2).sum()
            gradient_count += gradient.size
            
            second_deriv = np.gradient(gradient)
            second_deriv_sum += second_deriv.sum()
            second_deriv_sq_sum += (second_deriv ** 2).sum()
            second_deriv_count += second_deriv.size
            
            if 'energy_label' in data:
                energy_values.append(float(data['energy_label']))
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    waveform_mean = waveform_sum / waveform_count
    waveform_std = np.sqrt(waveform_sq_sum / waveform_count - waveform_mean ** 2)
    
    gradient_mean = gradient_sum / gradient_count
    gradient_std = np.sqrt(gradient_sq_sum / gradient_count - gradient_mean ** 2)
    
    second_deriv_mean = second_deriv_sum / second_deriv_count
    second_deriv_std = np.sqrt(second_deriv_sq_sum / second_deriv_count - second_deriv_mean ** 2)
    
    stats = {
        'waveform': {
            'mean': float(waveform_mean),
            'std': float(waveform_std),
            'min': float(waveform_min),
            'max': float(waveform_max),
            'count': int(waveform_count)
        },
        'gradient': {
            'mean': float(gradient_mean),
            'std': float(gradient_std),
            'count': int(gradient_count)
        },
        'second_deriv': {
            'mean': float(second_deriv_mean),
            'std': float(second_deriv_std),
            'count': int(second_deriv_count)
        },
        'n_files': len(train_files)
    }
    
    if energy_values:
        energy_array = np.array(energy_values)
        stats['energy'] = {
            'mean': float(energy_array.mean()),
            'std': float(energy_array.std()),
            'min': float(energy_array.min()),
            'max': float(energy_array.max()),
            'median': float(np.median(energy_array)),
            'count': len(energy_values)
        }
    
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStatistics saved to {output_path}")
    print("\nWaveform statistics:")
    print(f"  Mean: {stats['waveform']['mean']:.6f}")
    print(f"  Std:  {stats['waveform']['std']:.6f}")
    print(f"  Min:  {stats['waveform']['min']:.6f}")
    print(f"  Max:  {stats['waveform']['max']:.6f}")
    
    print("\nGradient statistics:")
    print(f"  Mean: {stats['gradient']['mean']:.6f}")
    print(f"  Std:  {stats['gradient']['std']:.6f}")
    
    print("\nSecond derivative statistics:")
    print(f"  Mean: {stats['second_deriv']['mean']:.6f}")
    print(f"  Std:  {stats['second_deriv']['std']:.6f}")
    
    if 'energy' in stats:
        print("\nEnergy statistics:")
        print(f"  Mean:   {stats['energy']['mean']:.6f}")
        print(f"  Std:    {stats['energy']['std']:.6f}")
        print(f"  Min:    {stats['energy']['min']:.6f}")
        print(f"  Max:    {stats['energy']['max']:.6f}")
        print(f"  Median: {stats['energy']['median']:.6f}")
    
    return stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute statistics for Majorana dataset')
    parser.add_argument('--data_root', type=str, default='data/waveform_npz',
                        help='Root directory containing the dataset')
    parser.add_argument('--output', type=str, default='dataset/stats.json',
                        help='Output JSON file for statistics')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    compute_statistics(args.data_root, args.output, args.max_files)
