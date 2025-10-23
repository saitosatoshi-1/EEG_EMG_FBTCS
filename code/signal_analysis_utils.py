"""
Signal analysis utilities for EEG/EMG research (public, reproducible).

This module provides core signal-processing and visualization functions used in:
Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H.
"Electromyographic components contaminating the scalp EEG during focal to bilateral tonic–clonic seizures
as potential markers for seizure detection and lateralization: an exploratory study."
Submitted to Epilepsy Research (2025).

Functions:
- Band-pass filtering (Butterworth, SOS)
- Hilbert envelope computation with optional smoothing
- Dynamic spectral analysis (DSA) plotting in decibel scale

All functions are self-contained and safe for open publication.
No clinical or patient-specific data are included.
"""

from __future__ import annotations
import numpy as np
from functools import lru_cache
from scipy.signal import butter, sosfiltfilt, hilbert, spectrogram
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Band-pass filter (cached SOS design)
# ---------------------------------------------------------------------
@lru_cache(maxsize=None)
def _bp_sos(sfreq: float, low: float, high: float, order: int = 4):
    """
    Design a Butterworth band-pass filter (second-order sections form).

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    low, high : float
        Lower and upper cutoff frequencies in Hz.
    order : int
        Filter order (default: 4).

    Returns
    -------
    sos : ndarray
        Second-order section coefficients for filtering.
    """
    nyq = 0.5 * float(sfreq)
    return butter(int(order), [low / nyq, high / nyq], btype="band", output="sos")


def bandpass_1d(x: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase band-pass filter to a 1D signal.

    Parameters
    ----------
    x : ndarray
        Input signal (1D array).
    sfreq : float
        Sampling frequency in Hz.
    low, high : float
        Passband frequencies in Hz.
    order : int
        Filter order.

    Returns
    -------
    x_filt : ndarray (float32)
        Band-pass filtered signal.
    """
    sos = _bp_sos(float(sfreq), float(low), float(high), int(order))
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float32)).astype(np.float32)


# ---------------------------------------------------------------------
# 2. Hilbert envelope computation
# ---------------------------------------------------------------------
def compute_hilbert_envelope(
    signal: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    order: int = 4,
    smooth_sec: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Band-pass filter -> Hilbert transform -> envelope magnitude.

    Optionally applies moving-average smoothing to the envelope.

    Parameters
    ----------
    signal : ndarray
        Input 1D signal.
    sfreq : float
        Sampling frequency in Hz.
    band : tuple(float, float)
        Frequency band for band-pass filtering (e.g., (64, 256)).
    order : int
        Butterworth filter order.
    smooth_sec : float or None
        Smoothing window in seconds. If None, no smoothing.

    Returns
    -------
    env_smooth : ndarray
        Smoothed envelope (if smoothing applied, else identical to raw envelope).
    env_raw : ndarray
        Unsmoothened envelope (absolute Hilbert transform).
    """
    low, high = band
    sig_f = bandpass_1d(signal, sfreq, low, high, order)
    env_raw = np.abs(hilbert(sig_f))

    if smooth_sec and smooth_sec > 0:
        win = max(1, int(sfreq * smooth_sec))
        kernel = np.ones(win, dtype=np.float32) / win
        env_smooth = np.convolve(env_raw, kernel, mode="same").astype(np.float32)
        return env_smooth, env_raw

    return env_raw.astype(np.float32), env_raw


# ---------------------------------------------------------------------
# 3. Dynamic spectral analysis (DSA) plotting
# ---------------------------------------------------------------------
def plot_dsa_db(
    ax: plt.Axes,
    signal: np.ndarray,
    sfreq: float,
    title: str = "DSA (0–300 Hz, dB)",
    win_sec: float = 0.5,
    overlap: float = 0.5,
    fmax: float = 300,
    cmap: str = "jet",
    add_colorbar: bool = False,
    fig: plt.Figure | None = None,
    vmin: float | None = None,
    vmax: float | None = None
):
    """
    Plot dynamic spectral analysis (DSA) in decibel scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis for plotting.
    signal : ndarray
        Input 1D signal.
    sfreq : float
        Sampling frequency in Hz.
    title : str
        Plot title.
    win_sec : float
        Window length in seconds for STFT.
    overlap : float
        Overlap ratio between 0 and 1.
    fmax : float
        Maximum frequency to display (Hz).
    cmap : str
        Colormap.
    add_colorbar : bool
        Whether to add a colorbar.
    fig : matplotlib.figure.Figure or None
        Figure handle for colorbar (if add_colorbar=True).
    vmin, vmax : float or None
        Color scale limits. If None, set to 5th–95th percentile.

    Returns
    -------
    f : ndarray
        Frequency bins (Hz).
    t_spec : ndarray
        Time bins (s).
    Sxx_db : ndarray
        Power spectral density (dB).
    """
    x = np.asarray(signal)
    nperseg = max(8, int(sfreq * win_sec))
    noverlap = int(nperseg * overlap)

    if x.size < nperseg:
        ax.text(0.5, 0.5, "Data too short", ha="center", va="center")
        ax.set_title(title)
        return None, None, None

    f, t_spec, Sxx = spectrogram(
        x, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
        scaling="density", mode="psd", padded=True, boundary="zeros"
    )
    Sxx_db = (10.0 * np.log10(Sxx + 1e-12)).astype(np.float32)

    if vmin is None or vmax is None:
        vmin = float(np.percentile(Sxx_db, 5))
        vmax = float(np.percentile(Sxx_db, 95))

    pcm = ax.pcolormesh(
        t_spec, f, Sxx_db, shading="gouraud", cmap=cmap,
        vmin=vmin, vmax=vmax, rasterized=True
    )
    ax.set_ylim(0, fmax)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

    if add_colorbar and fig is not None:
        fig.colorbar(
            pcm, ax=ax, orientation="horizontal",
            fraction=0.07, pad=0.15, label="Power (dB)"
        )

    return f, t_spec, Sxx_db


# ---------------------------------------------------------------------
# Example usage (documentation only)
# ---------------------------------------------------------------------
"""
Example
-------
>>> from signal_analysis_utils import compute_hilbert_envelope, plot_dsa_db
>>> import matplotlib.pyplot as plt
>>> fs = 1000
>>> t = np.arange(0, 5, 1/fs)
>>> x = np.sin(2*np.pi*120*t) + 0.2*np.random.randn(t.size)
>>> env, _ = compute_hilbert_envelope(x, fs, band=(64, 256), smooth_sec=0.01)
>>> fig, ax = plt.subplots(figsize=(6,3))
>>> plot_dsa_db(ax, x, fs, title="Synthetic EMG DSA", fmax=300)
>>> plt.show()
"""
