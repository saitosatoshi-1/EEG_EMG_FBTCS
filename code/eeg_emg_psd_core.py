"""
EEG/EMG PSD shape-correlation (analysis-only, no plotting)
- Welch PSD (EEG, EMG)
- log10 transform within a HF band (default 64–256 Hz)
- Correlate PSD "shapes" (Pearson by default; Spearman optional)
"""

from typing import Tuple, Optional, Literal
import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr

CorrMethod = Literal["pearson", "spearman"]

def compute_psd_shape_corr(
    eeg: np.ndarray,
    emg: np.ndarray,
    sfreq: float,
    hf_band: Tuple[float, float] = (64.0, 256.0),
    nperseg: Optional[int] = None,
    noverlap_frac: float = 0.5,
) -> Tuple[float, float, int]:
    """
    Compute log10-PSD shape correlation between EEG and EMG within a given band.
    eeg, emg : 1D ndarray
        Signals in microvolts (µV). Same sampling rate is assumed.
    sfreq : float
        Sampling frequency [Hz].
    hf_band : (low, high)
        Frequency range [Hz] used for correlation. Default (64, 256).
    nperseg : int or None
        Welch segment length. Default: int(0.5 * sfreq), min 8.
    noverlap_frac : float
        0..1 overlap ratio for Welch. Default 0.5.
    corr : {"pearson","spearman"}
        Correlation method. Default "pearson".
    min_bins : int
        Minimum number of frequency bins required to return a valid result.
    """
    
    # ---- Basic checks ----
    eeg = np.asarray(eeg, dtype=float)
    emg = np.asarray(emg, dtype=float)
    if eeg.ndim != 1 or emg.ndim != 1:
        raise ValueError("eeg and emg must be 1D arrays.")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("sfreq must be positive.")
    if len(eeg) < 4 or len(emg) < 4:
        return float("nan"), float("nan"), 0

    low, high = float(hf_band[0]), float(hf_band[1])
    if not (0 <= low < high):
        raise ValueError("hf_band must satisfy 0 <= low < high.")
    nyq = sfreq * 0.5
    if high > nyq:
        # Outside measurable range
        return float("nan"), float("nan"), 0

    if nperseg is None:
        nperseg = max(8, int(0.5 * float(sfreq)))
    noverlap = int(nperseg * float(noverlap_frac))
    if noverlap >= nperseg:
        noverlap = max(0, nperseg - 1)

    # ---- Welch PSDs ----
    #f_eeg: 周波数軸（例：0, 2, 4, …, 256Hz）
    #Pxx_eeg: パワー（同じ長さのベクトル）
    f_eeg, Pxx_eeg = welch(
        eeg, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
        window=window, detrend=detrend, scaling=scaling, average=average
    )
    f_emg, Pxx_emg = welch(
        emg, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
        window=window, detrend=detrend, scaling=scaling, average=average
    )

    # ---- Strict band mask & frequency alignment by intersection ----
    # Pick indices within band
    band_mask_eeg = (f_eeg >= low) & (f_eeg <= high)
    band_mask_emg = (f_emg >= low) & (f_emg <= high)
    f1 = f_eeg[band_mask_eeg]
    f2 = f_emg[band_mask_emg]
    if f1.size < 3 or f2.size < 3:
        return float("nan"), float("nan"), 0

    # Intersect frequency grids (exact equality should hold if Welch params match;
    # but we still intersect robustly to be future-proof).
    # 念のため “共通して存在する周波数”だけ を使う。
    f_common = np.intersect1d(f1, f2, assume_unique=False)
    if f_common.size < min_bins:
        return float("nan"), float("nan"), int(f_common.size)

    # Indices mapping
    idx1 = np.nonzero(np.in1d(f1, f_common))[0]
    idx2 = np.nonzero(np.in1d(f2, f_common))[0]

    # ---- Log10 transform (epsilon to avoid log(0)) ----
    eps = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        log_eeg = np.log10(Pxx_eeg[band_mask_eeg][idx1] + eps)
        log_emg = np.log10(Pxx_emg[band_mask_emg][idx2] + eps)

    # Finite-only & non-constant checks
    valid = np.isfinite(log_eeg) & np.isfinite(log_emg)
    if valid.sum() < min_bins:
        return float("nan"), float("nan"), int(valid.sum())

    x = log_eeg[valid]
    y = log_emg[valid]
    if np.isclose(np.std(x), 0) or np.isclose(np.std(y), 0):
        return float("nan"), float("nan"), int(valid.sum())

    # ---- Correlation ----
    if corr == "pearson":
        r, p = pearsonr(x, y)
    elif corr == "spearman":
        r, p = spearmanr(x, y)
    else:
        raise ValueError("corr must be 'pearson' or 'spearman'.")

    return float(r), float(p), int(valid.sum())

