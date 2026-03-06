"""
Feature extraction for GBDT classifier.

Extracts 12 hand-crafted geometric features from each waveform that capture
key pulse shape characteristics relevant to PSD discrimination tasks.

Features:
  Charge:     Q_max, Q_t_max, A_0
  Current:    A_max, A_t_max, PTR
  Derivative: J_max, J_t_max, delta
  Timing:     t_50, t_99, rise_time
"""

import numpy as np

FEATURE_NAMES = [
    'Q_max',      # Max normalized charge
    'Q_t_max',    # Time index of max charge (relative to tp0)
    'A_0',        # Integrated charge from trigger to end
    'A_max',      # Max normalized current
    'A_t_max',    # Time index of max current (relative to tp0)
    'PTR',        # Pulse-to-tail ratio
    'J_max',      # Max normalized jerk
    'J_t_max',    # Time index of max jerk (relative to tp0)
    'delta',      # Normalized linear approximation of decay trail
    't_50',       # Time index at 50% of max charge (relative to tp0)
    't_99',       # Time index at 99% of max charge (relative to tp0)
    'rise_time',  # Rise time: t_99 - t_50
]

PSD_LABELS = ['psd_label_dcr', 'psd_label_high_avse', 'psd_label_low_avse', 'psd_label_lq']
META_COLS = ['energy_label', 'detector']


def extract_features(waveform, tp0):
    """
    Extract 12 geometric features from a baseline-subtracted waveform.

    Parameters
    ----------
    waveform : np.ndarray, shape (N,)
        Raw waveform from the charge-sensitive preamplifier.
    tp0 : int
        Trigger point index separating baseline from signal region.

    Returns
    -------
    features : np.ndarray, shape (12,)
        Array of extracted features in the order given by FEATURE_NAMES.
    """
    # --- Baseline subtraction ---
    baseline = np.mean(waveform[:tp0]) if tp0 > 0 else 0.0
    wf = waveform - baseline

    # Signal region (from trigger onward)
    signal = wf[tp0:]
    n_signal = len(signal)

    # ========== Charge features ==========
    # A_0: integrated charge from trigger to end (area under the charge trace)
    A_0 = np.sum(signal)
    A_0_safe = A_0 if abs(A_0) > 1e-10 else 1e-10  # avoid division by zero

    # Q_max: maximum charge value, normalized by A_0
    Q_max_abs = np.max(signal)
    Q_max = Q_max_abs / A_0_safe

    # Q_t_max: time index of maximum charge (relative to tp0)
    Q_t_max = float(np.argmax(signal))

    # ========== Current features (first derivative) ==========
    current = np.gradient(wf)
    current_signal = current[tp0:]

    # A_max: maximum current, normalized by A_0
    A_max = np.max(current_signal) / A_0_safe

    # A_t_max: time index of maximum current (relative to tp0)
    max_current_idx = int(np.argmax(current_signal))
    A_t_max = float(max_current_idx)

    # PTR: pulse-to-tail ratio (integrated current before / after its maximum)
    integral_before = np.sum(np.abs(current_signal[:max_current_idx])) if max_current_idx > 0 else 0.0
    integral_after = np.sum(np.abs(current_signal[max_current_idx:])) if max_current_idx < n_signal else 0.0
    PTR = integral_before / (integral_after + 1e-10)

    # ========== Derivative features (jerk = second derivative) ==========
    jerk = np.gradient(current)
    jerk_signal = jerk[tp0:]

    # J_max: maximum jerk, normalized by A_0
    J_max = np.max(jerk_signal) / A_0_safe

    # J_t_max: time index of maximum jerk (relative to tp0)
    J_t_max = float(np.argmax(jerk_signal))

    # delta: normalized linear approximation of the decay trail
    # Fit a line to the current after its peak and take the slope
    decay_trail = current_signal[max_current_idx:]
    if len(decay_trail) > 10:
        # Use a fast linear regression with only endpoints and midpoint
        n_decay = len(decay_trail)
        x = np.arange(n_decay, dtype=np.float64)
        # Least squares slope: cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = decay_trail.mean()
        delta = np.sum((x - x_mean) * (decay_trail - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-10)
        delta = delta / A_0_safe
    else:
        delta = 0.0

    # ========== Timing features ==========
    # t_50: time when charge reaches 50% of its maximum value
    half_max = 0.5 * Q_max_abs
    t_50_candidates = np.where(signal >= half_max)[0]
    t_50 = float(t_50_candidates[0]) if len(t_50_candidates) > 0 else 0.0

    # t_99: time when charge reaches 99% of its maximum value
    near_max = 0.99 * Q_max_abs
    t_99_candidates = np.where(signal >= near_max)[0]
    t_99 = float(t_99_candidates[0]) if len(t_99_candidates) > 0 else float(n_signal - 1)

    # rise_time: t_99 - t_50 (characterizes how fast the pulse rises)
    rise_time = t_99 - t_50

    return np.array([
        Q_max, Q_t_max, A_0,
        A_max, A_t_max, PTR,
        J_max, J_t_max, delta,
        t_50, t_99, rise_time
    ], dtype=np.float64)


def extract_from_file(filepath):
    """
    Load a single .npz file and extract features + labels + metadata.

    Returns
    -------
    features : np.ndarray, shape (12,)
    labels : np.ndarray, shape (4,)  – [dcr, high_avse, low_avse, lq]
    meta : np.ndarray, shape (2,)    – [energy_label, detector]
    """
    data = np.load(filepath)
    waveform = data['waveform']
    tp0 = int(data.get('tp0', 1000))
    
    features = extract_features(waveform, tp0)
    
    labels = np.array([
        float(data['psd_label_dcr']),
        float(data['psd_label_high_avse']),
        float(data['psd_label_low_avse']),
        float(data['psd_label_lq']),
    ], dtype=np.float64)
    
    meta = np.array([
        float(data['energy_label']),
        float(data['detector']),
    ], dtype=np.float64)
    
    return features, labels, meta
