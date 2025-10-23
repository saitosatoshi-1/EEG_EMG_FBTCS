"""
EEG/EMG PSD shape-correlation (analysis-only, no plotting)

Core function to reproduce the main PSD-based result in:
Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H. (2025, submitted)
'Electromyographic components contaminating the scalp EEG ...'

What it does
------------
- Welch PSD (EEG, EMG)
- log10 transform within a HF band (default 64–256 Hz)
- Pearson correlation between EEG and EMG PSD shapes

This file contains no plotting code and no patient-specific I/O.
Safe to publish; can be called from scripts or notebooks.

Dependencies: numpy, scipy
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr


def compute_psd_shape_corr(
    eeg: np.ndarray,
    emg: np.ndarray,
    sfreq: float,
    hf_band: Tuple[float, float] = (64.0, 256.0),
    nperseg: Optional[int] = None,
    noverlap_frac: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute log–log PSD shape correlation (Pearson r, p) between EEG and EMG.

    Parameters
    ----------
    eeg, emg : ndarray
        1D signals in microvolts (µV). Must be same sampling rate.
    sfreq : float
        Sampling frequency in Hz.
    hf_band : (low, high)
        Frequency range (Hz) to evaluate correlation. Default (64, 256).
    nperseg : int or None
        Welch segment length. Default: int(0.5 * sfreq).
    noverlap_frac : float
        Overlap ratio for Welch (0..1). Default 0.5.

    Returns
    -------
    r, p : float, float
        Pearson correlation coefficient and two-sided p-value.
        If not enough points in the band, returns (nan, nan).
    """
    eeg = np.asarray(eeg, dtype=float)
    emg = np.asarray(emg, dtype=float)
    if eeg.ndim != 1 or emg.ndim != 1:
        raise ValueError("eeg and emg must be 1D arrays.")
    if len(eeg) < 4 or len(emg) < 4:
        return float("nan"), float("nan")

    if nperseg is None:
        nperseg = max(8, int(0.5 * float(sfreq)))
    noverlap = int(nperseg * float(noverlap_frac))

    f_eeg, Pxx_eeg = welch(eeg, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    f_emg, Pxx_emg = welch(emg, fs=sfreq, nperseg=nperseg, noverlap=noverlap)

    # Band mask and alignment
    mask_eeg = (f_eeg >= hf_band[0]) & (f_eeg <= hf_band[1])
    mask_emg = (f_emg >= hf_band[0]) & (f_emg <= hf_band[1])
    if mask_eeg.sum() < 3 or mask_emg.sum() < 3:
        return float("nan"), float("nan")

    m = min(mask_eeg.sum(), mask_emg.sum())
    log_eeg = np.log10(Pxx_eeg[mask_eeg][:m] + 1e-12)
    log_emg = np.log10(Pxx_emg[mask_emg][:m] + 1e-12)

    # If either spectrum is (near) constant, correlation is undefined
    if np.isclose(np.std(log_eeg), 0) or np.isclose(np.std(log_emg), 0):
        return float("nan"), float("nan")

    r, p = pearsonr(log_eeg, log_emg)
    return float(r), float(p)


# --- Minimal doc example (not executed) ---
"""
Example
-------
>>> import numpy as np
>>> from eeg_emg_psd_core import compute_psd_shape_corr
>>> fs = 1000
>>> t = np.arange(0, 5, 1/fs)
>>> eeg = np.sin(2*np.pi*10*t) + 0.3*np.random.randn(t.size)
>>> emg = np.random.randn(t.size) * np.sin(2*np.pi*120*t)
>>> r, p = compute_psd_shape_corr(eeg, emg, sfreq=fs, hf_band=(64,256))
>>> print(r, p)
"""
