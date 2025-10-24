"""
EEG embedded EMG high-frequency RMS metrics (analysis-only, no plotting)

Computes envelope-based RMS metrics from high-frequency (64–256 Hz) content:
- RMS_area (integral of RMS over time)
- RMS_area_per_sec
- Rise_time  (10% → 90% time-to-rise)
- Rise_slope (ΔRMS / Rise_time)

This implements the core of the 'Figure 5' analysis in:
Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H. (2025, submitted)
'Electromyographic components contaminating the scalp EEG ...'

Design goals
------------
- No patient data or file I/O
- No plotting
- Pure functions usable from scripts or notebooks

Dependencies: numpy, scipy, (optional) pandas
"""

from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.integrate import trapezoid

try:
    import pandas as pd
except Exception:
    pd = None  # DataFrame output is optional


# ---------------------------------------------------------------------
# Core building blocks
# ---------------------------------------------------------------------
def _bandpass_hilbert_envelope(
    x: np.ndarray,
    sfreq: float,
    band: Tuple[float, float] = (64.0, 256.0),
    order: int = 4,
) -> np.ndarray:
    """Band-pass filter (zero-phase) then Hilbert magnitude envelope."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 8:
        return np.array([], dtype=np.float32)
    nyq = 0.5 * float(sfreq)
    sos = butter(int(order), [band[0] / nyq, band[1] / nyq], btype="band", output="sos")
    xf = sosfiltfilt(sos, x)
    env = np.abs(hilbert(xf))
    return env.astype(np.float32)


def _rms_series(
    env: np.ndarray,
    sfreq: float,
    win_sec: float = 1.0,
    step_sec: float = 0.25,
    smooth_sec: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding RMS over the envelope. Returns (rms_values, time_centers).

    If smooth_sec > 0, applies simple moving average to the RMS series.
    """
    if env.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=float)

    win = max(1, int(round(win_sec * sfreq)))
    step = max(1, int(round(step_sec * sfreq)))
    if env.size < win:
        return np.array([], dtype=np.float32), np.array([], dtype=float)

    # Compute RMS in sliding windows
    idx = np.arange(0, env.size - win + 1, step, dtype=int)
    rms = np.sqrt(np.mean(np.square(np.stack([env[i:i+win] for i in idx], axis=0)), axis=1))
    rms = rms.astype(np.float32)

    # Time centers aligned to original signal
    centers = idx + win // 2
    dt = 1.0 / float(sfreq)
    t = centers * dt

    # Optional smoothing of RMS series
    if smooth_sec and smooth_sec > 0:
        k = max(1, int(round(smooth_sec / step_sec)))
        kernel = np.ones(k, dtype=np.float32) / float(k)
        rms = np.convolve(rms, kernel, mode="same")

    return rms.astype(np.float32), t.astype(float)


def _rise_metrics_from_rms(
    rms: np.ndarray,
    t: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute RMS_area, RMS_area_per_sec, Rise_time (10–90%), Rise_slope.

    Returns (area, area_per_sec, rise_time, rise_slope).
    If metrics cannot be computed, returns NaNs.
    """
    if rms.size < 2 or t.size != rms.size:
        return (np.nan, np.nan, np.nan, np.nan)

    # Integrals
    try:
        area = float(trapezoid(rms, x=t))
        dur = float(t[-1] - t[0])
    except Exception:
        # Fallback with uniform spacing
        if t.size > 1:
            dx = float(np.median(np.diff(t)))
            area = float(trapezoid(rms, dx=dx))
            dur = dx * (t.size - 1)
        else:
            return (np.nan, np.nan, np.nan, np.nan)

    area_per_sec = area / max(dur, 1e-12)

    # 10–90% rise metrics
    rmin, rmax = float(np.min(rms)), float(np.max(rms))
    if not np.isfinite(rmin) or not np.isfinite(rmax) or rmax <= rmin:
        return (area, area_per_sec, np.nan, np.nan)

    th10 = rmin + 0.10 * (rmax - rmin)
    th90 = rmin + 0.90 * (rmax - rmin)

    # First indices where threshold is crossed
    i10 = int(np.argmax(rms >= th10))
    i90 = int(np.argmax(rms >= th90))
    if i90 <= i10 or i90 >= rms.size or i10 >= rms.size:
        return (area, area_per_sec, np.nan, np.nan)

    rise_time = float(t[i90] - t[i10])
    rise_slope = float((rms[i90] - rms[i10]) / max(rise_time, 1e-12))
    return (area, area_per_sec, rise_time, rise_slope)

