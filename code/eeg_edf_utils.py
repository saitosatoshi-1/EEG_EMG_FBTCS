"""
EEG/EMG EDF loader and minimal preprocessing utilities (public, reproducible).

This module provides core functions used in:
Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H.
"Electromyographic components contaminating the scalp EEG during focal to bilateral tonic–clonic seizures
as potential markers for seizure detection and lateralization: an exploratory study."
Submitted to Epilepsy Research (2025).

Functions include:
- Safe EDF loading via MNE
- Channel name normalization
- EMG channel type assignment
- Optional notch and high-pass filtering
- Optional time cropping
- Extraction of EEG/EMG signals in microvolts (µV)

All functions are designed for reproducibility and contain no patient-specific data.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import mne


# ---------------------------------------------------------------------
# 1. EDF loading
# ---------------------------------------------------------------------
def read_edf(edf_path: str | Path, preload: bool = True) -> mne.io.BaseRaw:
    """
    Safely read an EDF file without exposing header text.

    Parameters
    ----------
    edf_path : str or Path
        Path to the EDF file.
    preload : bool
        If True, load all data into memory.

    Returns
    -------
    raw : mne.io.BaseRaw
        MNE Raw object containing EEG/EMG signals.
    """
    edf_path = Path(edf_path).expanduser()
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=preload, verbose="ERROR")
    return raw


# ---------------------------------------------------------------------
# 2. Channel normalization and labeling
# ---------------------------------------------------------------------
def normalize_channel_names(raw: mne.io.BaseRaw) -> None:
    """
    Remove vendor-specific prefixes or suffixes (non-destructive).

    Example: "EEG Fp1-Ref" → "Fp1"
    """
    rename_map = {}
    for ch in raw.ch_names:
        new = ch.replace("EEG ", "").replace("-Ref", "")
        if new != ch:
            rename_map[ch] = new
    if rename_map:
        raw.rename_channels(rename_map)


def set_emg_types(raw: mne.io.BaseRaw, emg_names: list[str]) -> None:
    """
    Mark listed channels as EMG type if they exist.

    Parameters
    ----------
    emg_names : list of str
        Channel names to mark as EMG.
    """
    to_emg = {ch: "emg" for ch in emg_names if ch in raw.ch_names}
    if to_emg:
        raw.set_channel_types(to_emg)


# ---------------------------------------------------------------------
# 3. Filtering and cropping
# ---------------------------------------------------------------------
def apply_filters(
    raw: mne.io.BaseRaw,
    notch: list[float] | None = None,
    hp: float | None = None
) -> None:
    """
    Apply notch and/or high-pass filtering to EEG/EMG channels.

    Parameters
    ----------
    notch : list of float or None
        Frequencies (e.g., [50, 100, 150, 200, 250]) to apply notch filters.
    hp : float or None
        High-pass cutoff frequency in Hz. If None, skip high-pass filtering.
    """
    picks = mne.pick_types(raw.info, eeg=True, emg=True, exclude=[])
    if notch:
        raw.notch_filter(freqs=notch, picks=picks)
    if hp is not None:
        raw.filter(l_freq=hp, h_freq=None, picks=picks, fir_design="firwin")


def crop_by_time(raw: mne.io.BaseRaw, t_start: float | None, t_end: float | None) -> None:
    """
    Crop the recording in-place between t_start and t_end (in seconds).

    Parameters
    ----------
    t_start : float or None
        Start time in seconds. None means from beginning.
    t_end : float or None
        End time in seconds. None means until the end.
    """
    if t_start is None and t_end is None:
        return
    if t_start is None:
        t_start = 0.0
    raw.crop(tmin=max(0.0, t_start), tmax=t_end)


# ---------------------------------------------------------------------
# 4. Data extraction
# ---------------------------------------------------------------------
def get_uv(raw: mne.io.BaseRaw, ch_list: list[str]) -> dict[str, np.ndarray]:
    """
    Return specified channels as microvolt (µV) arrays.

    Parameters
    ----------
    ch_list : list of str
        Channel names to extract.

    Returns
    -------
    out : dict
        Dictionary mapping each channel to a 1D NumPy array (float32, µV).
    """
    out = {}
    for ch in ch_list:
        if ch in raw.ch_names:
            out[ch] = (raw.get_data(picks=[ch])[0] * 1e6).astype(np.float32)
    return out


# ---------------------------------------------------------------------
# Example usage (for documentation only)
# ---------------------------------------------------------------------
"""
Example
-------
>>> from eeg_edf_utils import read_edf, normalize_channel_names, apply_filters, get_uv
>>> raw = read_edf("data/sample.edf")
>>> normalize_channel_names(raw)
>>> apply_filters(raw, notch=[50,100,150,200,250], hp=2.0)
>>> eeg = get_uv(raw, ["Fp1","Fp2","T3","T4"])
>>> print(list(eeg.keys()))
['Fp1', 'Fp2', 'T3', 'T4']
"""
