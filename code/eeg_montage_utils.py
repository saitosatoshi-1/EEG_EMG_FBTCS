"""
Montage & re-referencing utilities (CAR, CSD, Cz) for EEG research (public, reproducible).

This module provides safe, publication-ready helpers to:
- Prepare a Common Average Reference (CAR) montage
- Compute Current Source Density (CSD; spherical spline) from CAR
- Retrieve signals in microvolts (µV) after CAR/CSD
- Create a Cz-referenced pseudo-bipolar (channel - Cz) trace
- Build channel->waveform dictionaries robustly

Notes
-----
- No patient-specific data are exposed.
- Channel name normalization to 10–20 labels is supported (e.g., T7→T3, P7→T5).
- CSD requires sufficient spatial coverage; with too few valid electrodes, it may be unstable.

References
----------
Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H.
"Electromyographic components contaminating the scalp EEG during focal to bilateral tonic–clonic seizures
as potential markers for seizure detection and lateralization: an exploratory study."
Submitted to Epilepsy Research (2025).
"""

from __future__ import annotations
from typing import Iterable, Sequence, Tuple, Dict, Optional
import numpy as np
import mne
from mne.preprocessing import compute_current_source_density


__all__ = [
    "clean_eeg_make_car",
    "compute_csd",
    "prepare_car_csd",  # thin wrapper for backward-compat
    "get_car_signal",
    "get_csd_signal",
    "get_cz_bipolar",
    "make_eeg_dict",
]




# ---------------------------------------------------------------------
# Step 1: Prepare CAR (EEG pick -> normalize -> montage -> valid coords -> CAR)
# ---------------------------------------------------------------------
def clean_eeg_make_car(
    raw: mne.io.BaseRaw,
    *,
    exclude_chs: Iterable[str] = ("A1", "A2", "T1", "T2", "M1", "M2"),
    montage: str = "standard_1020",
    apply_10_20_alias: bool = True,
    verbose: bool = True,
) -> Tuple[mne.io.BaseRaw, Tuple[str, ...]]:
    """
    Prepare a CAR-referenced EEG Raw.

    Pipeline
    --------
    1) Pick EEG channels only, drop 'bads' if present.
    2) Normalize channel names, optionally alias 10-10 to 10-20 (e.g., T7->T3).
    3) Attach montage; ignore channels without coordinates.
    4) Keep only channels with valid 3D locations and not in 'exclude_chs'.
    5) Set CAR reference.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Input Raw.
    exclude_chs : Iterable[str]
        Channels to exclude from CAR (e.g., mastoids without standard coords).
    montage : str
        Montage name accepted by MNE (e.g., 'standard_1020').
    apply_10_20_alias : bool
        If True, alias some 10-10 labels to 10-20 equivalents (T7->T3, T8->T4, P7->T5, P8->T6).
    verbose : bool
        If True, print brief info.

    Returns
    -------
    raw_car : mne.io.BaseRaw
        EEG-only Raw with CAR applied.
    used_chs : tuple of str
        Channel names used in CAR.
    """
    # EEG only, drop 'bads'
    raw_car = raw.copy().pick_types(eeg=True, eog=False, emg=False, meg=False)
    bads = set(getattr(raw, "info", {}).get("bads", []) or [])
    to_drop = [ch for ch in raw_car.ch_names if ch in bads]
    if to_drop:
        raw_car.drop_channels(to_drop)

    # Normalize names: strip vendor tokens, optional 10-10→10-20 alias
    def _norm_name(name: str) -> str:
        base = name.replace("-Ref", "").replace("EEG ", "")
        if apply_10_20_alias:
            alias = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
            return alias.get(base, base)
        return base

    rename_map = {ch: _norm_name(ch) for ch in raw_car.ch_names}
    raw_car.rename_channels(rename_map)
    raw_car.set_channel_types({ch: "eeg" for ch in raw_car.ch_names})

    # Attach montage; ignore missing positions
    raw_car.set_montage(montage, on_missing="ignore")

    # Keep channels with valid 3D coords and not excluded
    valid: list[str] = []
    for ch in raw_car.ch_names:
        loc = raw_car.info["chs"][raw_car.ch_names.index(ch)]["loc"][:3]
        if (not np.allclose(loc, 0.0)) and (not np.isnan(loc).any()):
            if ch not in exclude_chs:
                valid.append(ch)

    if len(valid) < 16 and verbose:
        print(
            f"[clean_eeg_make_car] Warning: only {len(valid)} channels have valid coordinates; "
            "CSD may be unstable."
        )

    raw_car.pick_channels(valid)
    raw_car.set_eeg_reference(ref_channels="average", projection=False)

    if verbose:
        print("[clean_eeg_make_car] CAR channels:", ", ".join(raw_car.ch_names))

    return raw_car, tuple(raw_car.ch_names)





# ---------------------------------------------------------------------
# Step 2: Compute CSD (from CAR). Graceful fallback on failure.
# ---------------------------------------------------------------------
def compute_csd(
    raw_car: mne.io.BaseRaw,
    csd_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> Optional[mne.io.BaseRaw]:
    """
    Compute CSD (spherical spline) from a CAR-referenced Raw.

    Parameters
    ----------
    raw_car : mne.io.BaseRaw
        CAR-referenced EEG Raw.
    csd_kwargs : dict or None
        Keyword arguments for `mne.preprocessing.compute_current_source_density`,
        e.g., dict(lambda2=1e-5, stiffness=4, n_legendre_terms=50).
    verbose : bool
        If True, print brief info on failure.

    Returns
    -------
    raw_csd : mne.io.BaseRaw or None
        CSD-transformed Raw; None if computation fails.
    """
    csd_kwargs = csd_kwargs or dict(lambda2=1e-5, stiffness=4, n_legendre_terms=50)
    try:
        return compute_current_source_density(raw_car.copy(), **csd_kwargs)
    except Exception as e:
        if verbose:
            print("[compute_csd] CSD computation failed:", repr(e))
            print("  → Check montage/valid channels/exclusions or adjust csd_kwargs.")
        return None





# ---------------------------------------------------------------------
# Backward-compatible wrapper (CAR + CSD)
# ---------------------------------------------------------------------
def prepare_car_csd(
    raw: mne.io.BaseRaw,
    *,
    exclude_chs: Iterable[str] = ("A1", "A2", "T1", "T2", "M1", "M2"),
    montage: str = "standard_1020",
    apply_10_20_alias: bool = True,
    csd_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> Tuple[mne.io.BaseRaw, Optional[mne.io.BaseRaw], Tuple[str, ...]]:
    """
    Convenience wrapper: prepare CAR, then attempt CSD.
    """
    raw_car, used = clean_eeg_make_car(
        raw,
        exclude_chs=exclude_chs,
        montage=montage,
        apply_10_20_alias=apply_10_20_alias,
        verbose=verbose,
    )
    raw_csd = compute_csd(raw_car, csd_kwargs=csd_kwargs, verbose=verbose)
    return raw_car, raw_csd, used




# ---------------------------------------------------------------------
# Helpers to extract waveforms (µV) and derive Cz pseudo-bipolar
# ---------------------------------------------------------------------
def _get_uv_from_raw(raw_like: Optional[mne.io.BaseRaw], ch_name: str) -> Optional[np.ndarray]:
    """
    Return a single channel waveform in microvolts (µV). If missing, return None.
    """
    if (raw_like is None) or (ch_name not in raw_like.ch_names):
        return None
    return (raw_like.get_data(picks=[ch_name])[0] * 1e6).astype(np.float32)


def get_car_signal(raw_car: mne.io.BaseRaw, ch_name: str) -> np.ndarray:
    """
    Get CAR-referenced signal in µV. Raises ValueError if channel missing.
    """
    sig = _get_uv_from_raw(raw_car, ch_name)
    if sig is None:
        raise ValueError(f"[get_car_signal] Channel not found: {ch_name}")
    return sig


def get_csd_signal(
    raw_csd: Optional[mne.io.BaseRaw],
    ch_name: str,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Get CSD-transformed signal with an optional display scaling.

    Notes
    -----
    - CSD has pseudo-units (V/m^2-like); for plotting alongside µV signals,
      a visual scaling (e.g., 1/10000) can be applied.
    """
    if raw_csd is None:
        raise ValueError("[get_csd_signal] raw_csd is None (CSD not computed).")
    sig = _get_uv_from_raw(raw_csd, ch_name)  # convert to µV-like for visual comparability
    if sig is None:
        raise ValueError(f"[get_csd_signal] Channel not found: {ch_name}")
    return sig * float(scale)


def get_cz_bipolar(raw_car: mne.io.BaseRaw, ch_name: str, cz_name: str = "Cz") -> np.ndarray:
    """
    Return a pseudo-bipolar trace (ch - Cz) in µV within the CAR space.
    If Cz is absent, returns the CAR trace of 'ch' unchanged.
    """
    sig = get_car_signal(raw_car, ch_name)
    if cz_name in raw_car.ch_names:
        cz = get_car_signal(raw_car, cz_name)
        n = min(sig.size, cz.size)
        return (sig[:n] - cz[:n]).astype(np.float32)
    return sig.astype(np.float32)




# ---------------------------------------------------------------------
# Build channel dictionaries robustly
# ---------------------------------------------------------------------
def make_eeg_dict(
    raw_obj: Optional[mne.io.BaseRaw],
    target_chs: Sequence[str],
    getter,
) -> Dict[str, np.ndarray]:
    """
    Build {channel: waveform} using a provided getter (e.g., get_car_signal).

    Skips channels that do not exist in 'raw_obj'. Raises ValueError if none found.

    Parameters
    ----------
    raw_obj : Raw or None
        Source Raw (CAR or CSD). If None, returns ValueError.
    target_chs : sequence of str
        Channel names to attempt.
    getter : callable
        Function(raw_obj, ch_name) -> ndarray

    Returns
    -------
    out : dict
        Mapping from existing channel names to 1D waveforms (float32).
    """
    out: Dict[str, np.ndarray] = {}
    if raw_obj is not None:
        for ch in target_chs:
            if ch in raw_obj.ch_names:
                out[ch] = getter(raw_obj, ch)
    if not out:
        raise ValueError(
            "[make_eeg_dict] No valid channels. "
            "Check 'target_chs' against 'raw_obj.ch_names'."
        )
    return out





# ---------------------------------------------------------------------
# Documentation-only usage example (do not execute on import)
# ---------------------------------------------------------------------
"""
Example
-------
>>> # Step 1: CAR
>>> raw_car, used = prepare_car(
...     raw, exclude_chs=("A1","A2","T1","T2","M1","M2"),
...     montage="standard_1020", apply_10_20_alias=True, verbose=True
... )
>>> # Step 2: CSD (optional)
>>> raw_csd = compute_csd(raw_car, csd_kwargs=dict(lambda2=1e-5, stiffness=4, n_legendre_terms=50))
>>>
>>> # Build per-montage dictionaries for typical temporal/occipital channels:
>>> target = ["Fp1","Fp2","F7","F8","T3","T4","T5","T6","O1","O2"]
>>> eeg_car = make_eeg_dict(raw_car, target, get_car_signal)
>>> eeg_csd = make_eeg_dict(raw_csd, target, lambda r, ch: get_csd_signal(r, ch, scale=1/10000))
>>> eeg_cz  = make_eeg_dict(raw_car, target, get_cz_bipolar) if "Cz" in used else {}
"""
