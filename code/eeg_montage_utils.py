"""
Montage & re-referencing utilities (CAR, CSD, Cz) for EEG research (public, reproducible).

This module provides safe, publication-ready helpers to:
- Clean and prepare EEG with a standard montage (3D coords) and CAR
- Re-reference to Cz
- Compute Current Source Density (CSD; spherical spline) from CAR
- Retrieve signals in microvolts (µV) after CAR/CSD/Cz
- Build channel→waveform dictionaries robustly

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
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Dict, Optional, Callable
import numpy as np
import mne
from mne.preprocessing import compute_current_source_density


__all__ = [
    "EEGRefs",
    "prepare_eeg_and_car",
    "re_reference_to_cz",
    "compute_csd_from_car",
    "signal_from_car",
    "signal_from_cz",
    "signal_from_csd",
    "make_eeg_dict",
]


# ---------------------------------------------------------------------
# Data container for EEG references
# ---------------------------------------------------------------------
@dataclass
class EEGRefs:
    """Container holding EEG signals after re-referencing."""
    eeg_pre: mne.io.BaseRaw                   # pre-reference (cleaned)
    eeg_car: mne.io.BaseRaw                   # CAR-referenced
    eeg_cz: Optional[mne.io.BaseRaw]          # Cz-referenced
    eeg_csd: Optional[mne.io.BaseRaw]         # CSD (optional)
    used_channels: Tuple[str, ...]            # valid EEG channels (for CAR/CSD)
    montage: str                              # montage name


# ---------------------------------------------------------------------
# Step 1: Clean EEG and compute CAR
# ---------------------------------------------------------------------
def prepare_eeg_and_car(
    raw: mne.io.BaseRaw,
    *,
    exclude_chs: Iterable[str] = ("A1", "A2", "T1", "T2", "M1", "M2"),
    montage: str = "standard_1020",
    apply_10_20_alias: bool = True,
    verbose: bool = True,
) -> EEGRefs:
    """
    Clean EEG, attach montage (3D coords), and create CAR-referenced copy.
    """
    eeg_pre = raw.copy().pick_types(eeg=True, eog=False, emg=False, meg=False)

    # Drop bads
    bads = list(raw.info.get("bads", [])) if hasattr(raw, "info") else []
    if bads:
        eeg_pre.drop_channels([ch for ch in eeg_pre.ch_names if ch in bads])
        if verbose:
            print(f"[prepare_eeg_and_car] Dropped bad channels: {bads}")

    # Normalize names
    alias_map = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    def _normalize(name: str) -> str:
        base = name.replace("EEG ", "").replace("-Ref", "")
        return alias_map.get(base, base) if apply_10_20_alias else base

    eeg_pre.rename_channels({ch: _normalize(ch) for ch in eeg_pre.ch_names})
    eeg_pre.set_channel_types({ch: "eeg" for ch in eeg_pre.ch_names})

    # Attach montage
    try:
        eeg_pre.set_montage(montage, on_missing="ignore")
    except Exception as e:
        raise ValueError(f"[prepare_eeg_and_car] Montage '{montage}' could not be set: {e}")

    # Keep channels with valid 3D coordinates
    exclude_upper = set(s.upper() for s in exclude_chs)
    valid = []
    for ch in eeg_pre.ch_names:
        idx = eeg_pre.ch_names.index(ch)
        loc3 = eeg_pre.info["chs"][idx]["loc"][:3]
        has_pos = (not np.allclose(loc3, 0.0)) and (not np.isnan(loc3).any())
        if has_pos and (ch.upper() not in exclude_upper):
            valid.append(ch)

    if verbose:
        print(f"[prepare_eeg_and_car] Valid channels with coords: {len(valid)}")
    if len(valid) < 16 and verbose:
        print("[prepare_eeg_and_car] Warning: <16 valid channels; CSD may be unstable.")

    eeg_pre.pick_channels(valid)

    # Create CAR copy
    eeg_car = eeg_pre.copy()
    eeg_car.set_eeg_reference(ref_channels="average", projection=False)
    if verbose:
        print(f"[prepare_eeg_and_car] {len(eeg_car.ch_names)} channels used for CAR.")
        print(", ".join(eeg_car.ch_names))

    return EEGRefs(
        eeg_pre=eeg_pre,
        eeg_car=eeg_car,
        eeg_cz=None,
        eeg_csd=None,
        used_channels=tuple(eeg_car.ch_names),
        montage=montage,
    )


# ---------------------------------------------------------------------
# Cz reference
# ---------------------------------------------------------------------
def re_reference_to_cz(eeg_pre: mne.io.BaseRaw, cz_name: str = "Cz") -> mne.io.BaseRaw:
    """Re-reference EEG to Cz."""
    if cz_name not in eeg_pre.ch_names:
        raise ValueError(f"[re_reference_to_cz] Cz not found in channels: {cz_name}")
    return eeg_pre.copy().set_eeg_reference(ref_channels=[cz_name], projection=False)


# ---------------------------------------------------------------------
# Step 2: Compute CSD (from CAR)
# ---------------------------------------------------------------------
def compute_csd_from_car(
    eeg_car: mne.io.BaseRaw,
    *,
    verbose: bool = True,
) -> Optional[mne.io.BaseRaw]:
    """Compute CSD (spherical spline) from CAR-referenced EEG (MNE defaults)."""
    try:
        return compute_current_source_density(eeg_car.copy())
    except Exception as e:
        if verbose:
            print("[compute_csd_from_car] CSD computation failed:", repr(e))
            print("  → Check montage or electrode coverage.")
        return None


# ---------------------------------------------------------------------
# Step 3: Signal extractors (µV)
# ---------------------------------------------------------------------
def _get_uv_from_raw(raw: Optional[mne.io.BaseRaw], ch_name: str) -> Optional[np.ndarray]:
    """Extract a channel waveform from an MNE Raw object and convert to µV."""
    if (raw is None) or (ch_name not in raw.ch_names):
        return None
    return (raw.get_data(picks=[ch_name])[0] * 1e6).astype(np.float32)

def signal_from_car(eeg_car: mne.io.BaseRaw, ch_name: str) -> np.ndarray:
    sig = _get_uv_from_raw(eeg_car, ch_name)
    if sig is None:
        raise ValueError(f"[signal_from_car] Channel not found: {ch_name}")
    return sig

def signal_from_cz(eeg_cz: Optional[mne.io.BaseRaw], ch_name: str) -> np.ndarray:
    if eeg_cz is None:
        raise ValueError("[signal_from_cz] eeg_cz is None (Cz not computed or Cz missing).")
    sig = _get_uv_from_raw(eeg_cz, ch_name)
    if sig is None:
        raise ValueError(f"[signal_from_cz] Channel not found: {ch_name}")
    return sig

def signal_from_csd(
    eeg_csd: Optional[mne.io.BaseRaw],
    ch_name: str,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    if eeg_csd is None:
        raise ValueError("[signal_from_csd] eeg_csd is None (CSD not computed).")
    sig = _get_uv_from_raw(eeg_csd, ch_name)
    if sig is None:
        raise ValueError(f"[signal_from_csd] Channel not found: {ch_name}")
    return sig * float(scale)


# ---------------------------------------------------------------------
# Step 4: Build channel dictionaries
# ---------------------------------------------------------------------
def make_eeg_dict(
    raw_obj: Optional[mne.io.BaseRaw],
    target_chs: Sequence[str],
    getter: Callable[[mne.io.BaseRaw, str], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Build a {channel: waveform} dictionary using a given getter function."""
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
# Documentation-only usage example
# ---------------------------------------------------------------------
"""
Example
-------
>>> refs = prepare_eeg_and_car(raw, montage="standard_1020")
>>> eeg_car_dict = make_eeg_dict(refs.eeg_car, ["Fp1","Fp2","T3","T4"], signal_from_car)
>>> eeg_cz = re_reference_to_cz(refs.eeg_pre, cz_name="Cz")
>>> eeg_cz_dict = make_eeg_dict(eeg_cz, ["Fp1","Fp2","T3","T4"], signal_from_cz)
>>> eeg_csd = compute_csd_from_car(refs.eeg_car)
>>> eeg_csd_dict = make_eeg_dict(
...     eeg_csd, ["Fp1","Fp2","T3","T4"],
...     lambda r, ch: signal_from_csd(r, ch, scale=1/10000)
... ) if eeg_csd is not None else {}
"""
