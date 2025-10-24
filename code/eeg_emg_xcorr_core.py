"""
EEG/EMG high-frequency RMS cross-correlation analysis (no plotting)

Computes lag and correlation between EEG and EMG high-frequency (64–256 Hz)
Hilbert-RMS envelopes. This is a simplified, analysis-only version of the
cross-correlation method used in:

Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H. (2025, submitted)
'Electromyographic components contaminating the scalp EEG ...'
Epilepsy Research (under review).

What it does
------------
- Compute Hilbert envelope (64–256 Hz)
- Compute sliding-window RMS (1 s window, 0.25 s step)
- Normalize (0–1)
- Compute Pearson cross-correlation at multiple lags (±30 s)
- Return max correlation (r_max) and lag_sec

No plotting or file I/O. Safe for publication.

Dependencies: numpy, scipy
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import pearsonr


def compute_hilbert_envelope(sig: np.ndarray, sfreq: float,
                             band=(64, 256), order: int = 4) -> np.ndarray:
    """Band-pass filter and compute Hilbert envelope."""
    nyq = 0.5 * sfreq
    sos = butter(order, [band[0]/nyq, band[1]/nyq], btype="band", output="sos")
    filtered = sosfiltfilt(sos, sig.astype(float))
    env = np.abs(hilbert(filtered))
    return env.astype(np.float32)


def compute_hf_rms(env: np.ndarray, sfreq: float,
                   win_sec: float = 1.0, step_sec: float = 0.25,
                   smooth_sec: float = 6.0) -> np.ndarray:
    """Compute sliding RMS of HF envelope (normalized 0–1, smoothed)."""
    win_smp = int(win_sec * sfreq)
    step_smp = int(step_sec * sfreq)
    rms = []
    for i in range(0, len(env) - win_smp, step_smp):
        w = env[i:i + win_smp]
        rms.append(np.sqrt(np.mean(w * w)))
    rms = np.array(rms, dtype=float)
    if rms.size == 0:
        return rms
    rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-8)
    if smooth_sec > 0:
        w = max(1, int(smooth_sec / step_sec))
        kernel = np.ones(w) / w
        rms = np.convolve(rms, kernel, mode="same")
    return rms.astype(np.float32)


def compute_rms_crosscorr(
    eeg: np.ndarray,
    emg: np.ndarray,
    sfreq: float,
    hf_band=(64, 256),
    win_sec: float = 1.0,
    step_sec: float = 0.25,
    max_lag_sec: float = 20.0,
) -> dict:
    """
    Compute EEG–EMG RMS cross-correlation.

    Parameters
    ----------
    eeg, emg : ndarray
        1D EEG and EMG signals (µV).
    sfreq : float
        Sampling frequency (Hz).
    hf_band : tuple
        High-frequency band for Hilbert envelope (Hz).
    win_sec, step_sec : float
        RMS window and step durations (s).
    max_lag_sec : float
        Maximum lag to evaluate (± seconds).

    Returns
    -------
    result : dict
        {
          'r_max': float,
          'p': float,
          'lag_sec': float,
          'lead': 'EEG→EMG' or 'EMG→EEG' or 'sync',
          'win_sec': float,
          'step_sec': float
        }
    """
    env_eeg = compute_hilbert_envelope(eeg, sfreq, hf_band)
    env_emg = compute_hilbert_envelope(emg, sfreq, hf_band)
    x = compute_hf_rms(env_eeg, sfreq, win_sec, step_sec)
    y = compute_hf_rms(env_emg, sfreq, win_sec, step_sec)
    if x.size == 0 or y.size == 0:
        return dict(r_max=np.nan, p=np.nan, lag_sec=np.nan, lead="NA",
                    win_sec=win_sec, step_sec=step_sec)

    max_lag = int(max_lag_sec / step_sec)
    lags = np.arange(-max_lag, max_lag + 1)
    rs = []
    for lag in lags:
        if lag < 0:
            xx, yy = x[:lag], y[-lag:]
        elif lag > 0:
            xx, yy = x[lag:], y[:-lag]
        else:
            xx, yy = x, y
        if len(xx) < 3 or len(yy) < 3:
            rs.append(np.nan)
        else:
            rs.append(np.corrcoef(xx, yy)[0, 1])
    rs = np.array(rs, dtype=float)

    if np.all(np.isnan(rs)):
        return dict(r_max=np.nan, p=np.nan, lag_sec=np.nan, lead="NA",
                    win_sec=win_sec, step_sec=step_sec)

    k = int(np.nanargmax(rs))
    lag_sec = lags[k] * step_sec
    r_max = rs[k]

    if lags[k] > 0:
        xx, yy = x[lags[k]:], y[:-lags[k]]
    elif lags[k] < 0:
        xx, yy = x[:lags[k]], y[-lags[k]:]
    else:
        xx, yy = x, y
    p_val = pearsonr(xx, yy)[1] if (len(xx) > 2 and len(xx) == len(yy)) else np.nan

    lead = "EMG→EEG" if lag_sec > 0 else ("EEG→EMG" if lag_sec < 0 else "sync")
    return dict(r_max=float(r_max), p=float(p_val),
                lag_sec=float(lag_sec), lead=lead,
                win_sec=win_sec, step_sec=step_sec)



