"""
Utility functions for EEG-embedded EMG analysis.
Saito et al., Epilepsy Research (2025)
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from scipy.signal import hilbert
from typing import Tuple, Optional
import mne
import pandas as pd
import os

# ----------------------------
# IO
# ----------------------------
def load_timeseries(path: str, ch_names: Optional[list[str]] = None) -> Tuple[np.ndarray, float, list[str]]:
    """
    Load time series from CSV or EDF.
    CSV format: columns = channels (optionally the first column 'time' is ignored if monotonic)
    EDF format: loads all channels, returns data in shape (n_samples, n_channels)

    Returns:
        data: (n_samples, n_channels), float32
        fs: sampling frequency (Hz)
        channels: list of names
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if "time" in df.columns[0].lower():
            df = df.iloc[:, 1:]
        if ch_names is not None:
            df = df[ch_names]
        data = df.values.astype(np.float32)
        # CSV does not carry fs; fall back to common defaults via env var or 1000 Hz
        fs = float(os.environ.get("FS_HZ", "1000"))
        channels = list(df.columns)
        return data, fs, channels

    elif ext in (".edf", ".bdf"):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        if ch_names is not None:
            raw.pick(ch_names)
        fs = float(raw.info["sfreq"])
        data = raw.get_data().T.astype(np.float32)  # (n_samples, n_channels)
        channels = raw.ch_names
        return data, fs, channels
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ----------------------------
# Filters & envelopes
# ----------------------------
def bandpass(data: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass (sosfiltfilt). data: (n_samples, n_channels) or (n_samples,)"""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    if data.ndim == 1:
        return sosfiltfilt(sos, data)
    return np.stack([sosfiltfilt(sos, data[:, i]) for i in range(data.shape[1])], axis=1)

def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Amplitude envelope via analytic signal."""
    if x.ndim == 1:
        return np.abs(hilbert(x))
    return np.abs(hilbert(x, axis=0))

def rms(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Root-mean-square along axis (default per-sample across channels if axis=1 set by caller)."""
    return np.sqrt(np.mean(np.square(x), axis=axis))

def integrated_rms(env: np.ndarray, fs: float, t_start: float = 0.0, t_end: Optional[float] = None) -> float:
    """Integral of envelope (iRMS) over [t_start, t_end] seconds."""
    n = env.shape[0]
    if t_end is None:
        t_end = n / fs
    i0 = int(max(0, round(t_start * fs)))
    i1 = int(min(n, round(t_end * fs)))
    segment = env[i0:i1]
    return float(np.trapz(segment, dx=1.0/fs))

# ----------------------------
# PSD & correlations
# ----------------------------
def psd(x: np.ndarray, fs: float, nperseg: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD (averaged across channels if 2D)."""
    if x.ndim == 1:
        f, pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
        return f, pxx
    # average channel PSDs
    pxxs = []
    for i in range(x.shape[1]):
        f, pxx_i = welch(x[:, i], fs=fs, nperseg=min(nperseg, len(x)))
        pxxs.append(pxx_i)
    return f, np.mean(np.stack(pxxs, axis=0), axis=0)

def psd_similarity(x: np.ndarray, y: np.ndarray, fs: float, nperseg: int = 1024) -> float:
    """Pearson correlation between PSDs of two signals (averaged across channels if multi-channel)."""
    f, pxx_x = psd(x, fs, nperseg=nperseg)
    _, pxx_y = psd(y, fs, nperseg=nperseg)
    if pxx_x.size != pxx_y.size:
        m = min(pxx_x.size, pxx_y.size)
        pxx_x, pxx_y = pxx_x[:m], pxx_y[:m]
    if np.allclose(np.std(pxx_x), 0) or np.allclose(np.std(pxx_y), 0):
        return np.nan
    r = np.corrcoef(pxx_x, pxx_y)[0, 1]
    return float(r)

# ----------------------------
# Rise slope (10–90%)
# ----------------------------
def rise_slope_10_90(env: np.ndarray, fs: float) -> Tuple[float, float, float]:
    """
    Compute 10–90% rise slope on an envelope trace.
    Returns (slope_per_s, t10_s, t90_s). If flat signal, returns (nan, nan, nan).
    """
    v = env.astype(float)
    v_min, v_max = np.nanmin(v), np.nanmax(v)
    if np.isclose(v_max - v_min, 0.0):
        return (np.nan, np.nan, np.nan)
    v10, v90 = v_min + 0.1*(v_max - v_min), v_min + 0.9*(v_max - v_min)

    # first index where envelope crosses thresholds
    def _first_cross(th):
        idx = np.where(v >= th)[0]
        return idx[0] if idx.size else None

    i10 = _first_cross(v10)
    i90 = _first_cross(v90)
    if i10 is None or i90 is None or i90 <= i10:
        return (np.nan, np.nan, np.nan)

    dt = (i90 - i10) / fs
    slope = (v90 - v10) / dt  # units: amplitude per second
    return float(slope), float(i10/fs), float(i90/fs)
