"""
EEG/EMG EDF loader and minimal preprocessing utilities (public, reproducible).
Functions include:
- Safe EDF loading via MNE
- Channel name normalization
- EMG channel type assignment
- Optional notch and high-pass filtering
- Optional time cropping
- Extraction of EEG/EMG signals in microvolts (µV)
"""

# ---------------------------------------------------------------------
# 1. EDF loading
# ---------------------------------------------------------------------
import mne

def read_edf(edf_path: str, preload: bool = True) -> mne.io.BaseRaw:
    """
    preload : bool, default True
        If True, read data into memory (recommended for filtering/cropping).
    """
    raw = mne.io.read_raw_edf(edf_path, preload=preload, verbose="ERROR")
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


def set_emg_types(raw: mne.io.BaseRaw, emg_names: list[str] | None = None) -> None:
    """
    Mark EMG channels (e.g., X5, X7, or Deltoid) as 'emg' type if present.
    """
    if emg_names is None:
        emg_names = [ch for ch in raw.ch_names if any(k in ch.upper() for k in ["X", "EMG", "DELTOID"])]
    to_emg = {ch: "emg" for ch in emg_names if ch in raw.ch_names}
    if to_emg:
        raw.set_channel_types(to_emg)
        print(f"Set as EMG: {list(to_emg.keys())}")


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
    """
    if t_start is None and t_end is None:
        return
    if t_start is None:
        t_start = 0.0
    raw.crop(tmin=max(0.0, t_start), tmax=t_end)


# ---------------------------------------------------------------------
# 4. Data extraction
# ---------------------------------------------------------------------
import numpy as np

def get_uv(raw: mne.io.BaseRaw, ch_list: list[str]) -> dict[str, np.ndarray]:
    """
    Return specified channels as microvolt (µV) arrays.
    out : dict
        Dictionary mapping each channel to a 1D NumPy array (float32, µV).
    """
    out = {}
    for ch in ch_list:
        if ch in raw.ch_names:
            out[ch] = (raw.get_data(picks=[ch])[0] * 1e6).astype(np.float32)
    return out

