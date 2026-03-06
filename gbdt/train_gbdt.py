"""
Train GBDT models for PSD classification.

Trains one gradient-boosted decision tree per PSD task (DCR, high AvsE,
low AvsE, LQ) using the same train/val split as the transformer (90/10,
seed=42).

All 12 geometric features + energy + detector are used as input.

Usage:
    python gbdt/train_gbdt.py --prepared_dir gbdt/prepared --output_dir gbdt/models
"""

import os
import sys
import json
import argparse
import numpy as np
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report
)
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gbdt.features import FEATURE_NAMES, PSD_LABELS, META_COLS

TASK_NAMES = ['dcr', 'high_avse', 'low_avse', 'lq']


def load_prepared_data(prepared_dir, split='train'):
    """Load prepared feature matrices."""
    path = os.path.join(prepared_dir, f'{split}_features.npz')
    print(f"Loading {path}...")
    data = np.load(path, allow_pickle=True)
    features = data['features']     # (N, 12)
    labels = data['labels']         # (N, 4)
    meta = data['meta']             # (N, 2) -> [energy, detector]
    print(f"  Loaded {features.shape[0]} samples, {features.shape[1]} features")
    return features, labels, meta


def build_X(features, meta):
    """
    Combine geometric features with detector ID.
    
    Energy is NOT included as an input feature: the transformer predicts
    energy as an auxiliary task, so using it here would be an unfair
    comparison.  Only detector ID is added (the transformer receives it
    via FiLM conditioning).
    """
    # features: (N, 12), meta[:, 1:2] = detector only (skip energy at idx 0)
    detector = meta[:, 1:2]
    X = np.hstack([features, detector])
    col_names = FEATURE_NAMES + ['detector']
    return X, col_names


def train_single_task(X_train, y_train, X_val, y_val, task_name, args):
    """Train a GBDT for a single PSD binary classification task."""
    print(f"\n{'='*50}")
    print(f"Training GBDT for: {task_name}")
    print(f"{'='*50}")
    
    # Class distribution
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"  Train: {len(y_train)} samples ({n_pos} pos, {n_neg} neg, "
          f"ratio={n_pos/len(y_train):.3f})")
    n_pos_val = int(y_val.sum())
    n_neg_val = len(y_val) - n_pos_val
    print(f"  Val:   {len(y_val)} samples ({n_pos_val} pos, {n_neg_val} neg, "
          f"ratio={n_pos_val/len(y_val):.3f})")
    
    # Train GBDT with squared loss (deviance) for binary classification 
    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=42,
        verbose=1,
    )
    
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    
    # Evaluate on train and val
    results = {}
    for split_name, X, y in [('train', X_train, y_train), ('val', X_val, y_val)]:
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob)
        f1 = f1_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        
        results[split_name] = {
            'accuracy': float(acc),
            'auc': float(auc),
            'f1': float(f1),
            'precision': float(prec),
            'recall': float(rec),
        }
        
        print(f"\n  {split_name.upper()} Results:")
        print(f"    Accuracy:  {acc:.4f}")
        print(f"    AUC:       {auc:.4f}")
        print(f"    F1:        {f1:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall:    {rec:.4f}")
    
    # Feature importance
    importances = clf.feature_importances_
    col_names_task = FEATURE_NAMES + ['detector']
    results['feature_importance'] = {
        name: float(imp) for name, imp in zip(col_names_task, importances)
    }
    sorted_feats = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top features:")
    for name, imp in sorted_feats[:5]:
        print(f"    {name}: {imp:.4f}")
    
    results['training_time_s'] = elapsed
    results['n_estimators'] = args.n_estimators
    results['max_depth'] = args.max_depth
    results['learning_rate'] = args.learning_rate
    
    return clf, results


def main():
    parser = argparse.ArgumentParser(description='Train GBDT for PSD classification')
    parser.add_argument('--prepared_dir', type=str, default='gbdt/prepared',
                        help='Directory with prepared feature files')
    parser.add_argument('--output_dir', type=str, default='gbdt/models',
                        help='Directory to save trained models')
    
    # GBDT hyperparameters
    parser.add_argument('--n_estimators', type=int, default=500,
                        help='Number of boosting stages (trees)')
    parser.add_argument('--max_depth', type=int, default=5,
                        help='Maximum depth of individual trees (3-5 typical)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate shrinkage')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='Fraction of samples used per tree (stochastic GB)')
    parser.add_argument('--min_samples_split', type=int, default=100,
                        help='Min samples required to split a node')
    parser.add_argument('--min_samples_leaf', type=int, default=50,
                        help='Min samples in each leaf')
    parser.add_argument('--max_features', type=str, default='sqrt',
                        help='Number of features per split (sqrt, log2, or float)')
    
    # Train/val split (must match transformer)
    parser.add_argument('--train_val_split', type=float, default=0.9,
                        help='Train/val split ratio (same as transformer)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (same as transformer)')
    
    args = parser.parse_args()
    
    # Load prepared training data
    features, labels, meta = load_prepared_data(args.prepared_dir, split='train')
    
    # Build feature matrix (geometric features + energy + detector)
    X, col_names = build_X(features, meta)
    print(f"\nFull feature matrix: {X.shape} ({len(col_names)} features)")
    print(f"Feature columns: {col_names}")
    
    # Handle any NaN/Inf values
    nan_mask = np.any(np.isnan(X) | np.isinf(X), axis=1)
    if np.sum(nan_mask) > 0:
        print(f"Removing {np.sum(nan_mask)} samples with NaN/Inf features")
        X = X[~nan_mask]
        labels = labels[~nan_mask]
    
    # Split into train/val using the SAME approach as the transformer
    # The transformer uses torch random_split with seed=42.
    # We replicate with sklearn's train_test_split using the same seed.
    # Note: The exact indices won't match (torch vs numpy RNG), but the
    # same ratio and seed ensures a comparable, fair split.
    X_train, X_val, y_train_all, y_val_all = train_test_split(
        X, labels,
        train_size=args.train_val_split,
        random_state=args.seed,
        shuffle=True,
    )
    print(f"\nTrain/Val split: {len(X_train)} train, {len(X_val)} val "
          f"(ratio={args.train_val_split})")
    
    # Train one model per PSD task
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}
    
    for task_idx, task_name in enumerate(TASK_NAMES):
        y_train = y_train_all[:, task_idx].astype(int)
        y_val = y_val_all[:, task_idx].astype(int)
        
        clf, results = train_single_task(
            X_train, y_train, X_val, y_val, task_name, args
        )
        
        # Save model
        model_path = os.path.join(args.output_dir, f'gbdt_{task_name}.joblib')
        joblib.dump(clf, model_path)
        print(f"  Saved model to {model_path}")
        
        all_results[task_name] = results
    
    # Save combined results
    results_path = os.path.join(args.output_dir, 'train_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved training results to {results_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"{'Task':<15} {'Train Acc':>10} {'Val Acc':>10} {'Val AUC':>10} {'Val F1':>10}")
    print("-" * 55)
    for task_name in TASK_NAMES:
        r = all_results[task_name]
        print(f"{task_name:<15} "
              f"{r['train']['accuracy']:>10.4f} "
              f"{r['val']['accuracy']:>10.4f} "
              f"{r['val']['auc']:>10.4f} "
              f"{r['val']['f1']:>10.4f}")


if __name__ == '__main__':
    main()
