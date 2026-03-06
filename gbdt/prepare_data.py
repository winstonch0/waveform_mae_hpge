"""
Prepare feature matrices for GBDT training and testing.

Extracts features from all waveform .npz files using multiprocessing
and saves the results as compressed .npz files for fast loading.

Usage:
    python gbdt/prepare_data.py --data_root data/waveform_npz --output_dir gbdt/prepared
"""

import os
import sys
import argparse
import numpy as np
from glob import glob
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gbdt.features import extract_from_file, FEATURE_NAMES, PSD_LABELS, META_COLS


def process_chunk(file_list):
    """Process a list of files and return stacked arrays."""
    n = len(file_list)
    features = np.empty((n, 12), dtype=np.float64)
    labels = np.empty((n, 4), dtype=np.float64)
    meta = np.empty((n, 2), dtype=np.float64)
    
    for i, f in enumerate(file_list):
        try:
            features[i], labels[i], meta[i] = extract_from_file(f)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            features[i] = np.nan
            labels[i] = np.nan
            meta[i] = np.nan
    
    return features, labels, meta


def process_single_file(filepath):
    """Wrapper for multiprocessing Pool.map."""
    try:
        return extract_from_file(filepath)
    except Exception as e:
        return (
            np.full(12, np.nan, dtype=np.float64),
            np.full(4, np.nan, dtype=np.float64),
            np.full(2, np.nan, dtype=np.float64),
        )


def prepare_split(data_root, pattern_filter, output_path, num_workers=16):
    """
    Extract features from all files matching pattern_filter and save to disk.

    Parameters
    ----------
    data_root : str
        Root directory containing waveform .npz files.
    pattern_filter : str
        Pattern to filter directories (e.g., 'MJD_Train', 'MJD_Test').
    output_path : str
        Path to save the prepared .npz file.
    num_workers : int
        Number of parallel workers for feature extraction.
    """
    # Collect all files
    all_files = sorted(glob(os.path.join(data_root, '**', '*.npz'), recursive=True),
                       key=lambda x: x.lower())
    files = [f for f in all_files if pattern_filter in f]
    print(f"Found {len(files)} files matching '{pattern_filter}'")
    
    if len(files) == 0:
        print("No files found. Check data_root and pattern_filter.")
        return
    
    # Process in parallel
    t0 = time.time()
    print(f"Extracting features with {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        # Use imap_unordered for progress tracking
        results = []
        done = 0
        total = len(files)
        
        for result in pool.imap(process_single_file, files, chunksize=500):
            results.append(result)
            done += 1
            if done % 50000 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  {done}/{total} ({100*done/total:.1f}%) - "
                      f"{rate:.0f} files/s - ETA: {eta:.0f}s")
    
    # Stack results
    features = np.array([r[0] for r in results], dtype=np.float64)
    labels = np.array([r[1] for r in results], dtype=np.float64)
    meta = np.array([r[2] for r in results], dtype=np.float64)
    
    # Check for NaN rows (failed extractions)
    nan_mask = np.any(np.isnan(features), axis=1)
    n_failed = np.sum(nan_mask)
    if n_failed > 0:
        print(f"WARNING: {n_failed} files failed feature extraction. Removing them.")
        features = features[~nan_mask]
        labels = labels[~nan_mask]
        meta = meta[~nan_mask]
    
    elapsed = time.time() - t0
    print(f"Feature extraction complete: {len(features)} samples in {elapsed:.1f}s")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape:   {labels.shape}")
    print(f"Meta shape:     {meta.shape}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        meta=meta,
        feature_names=FEATURE_NAMES,
        label_names=PSD_LABELS,
        meta_names=META_COLS,
    )
    fsize_mb = os.path.getsize(output_path + '.npz' if not output_path.endswith('.npz') else output_path) / 1e6
    print(f"Saved to {output_path} ({fsize_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Prepare feature matrices for GBDT')
    parser.add_argument('--data_root', type=str, default='data/waveform_npz',
                        help='Root directory containing waveform .npz folders')
    parser.add_argument('--output_dir', type=str, default='gbdt/prepared',
                        help='Output directory for prepared data')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of parallel workers')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training data preparation')
    parser.add_argument('--skip_test', action='store_true',
                        help='Skip test data preparation')
    args = parser.parse_args()
    
    if not args.skip_train:
        print("=" * 60)
        print("Preparing TRAINING data (MJD_Train_*)")
        print("=" * 60)
        prepare_split(
            data_root=args.data_root,
            pattern_filter='MJD_Train',
            output_path=os.path.join(args.output_dir, 'train_features.npz'),
            num_workers=args.num_workers,
        )
    
    if not args.skip_test:
        print("\n" + "=" * 60)
        print("Preparing TEST data (MJD_Test_*)")
        print("=" * 60)
        prepare_split(
            data_root=args.data_root,
            pattern_filter='MJD_Test',
            output_path=os.path.join(args.output_dir, 'test_features.npz'),
            num_workers=args.num_workers,
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
