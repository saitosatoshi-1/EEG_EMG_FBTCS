"""
Montage & re-referencing utilities (CAR, CSD, Cz) for EEG research (public, reproducible).

This module provides safe, publication-ready helpers to:
- Clean and prepare EEG with a standard montage (3D coords) and CAR
- Re-reference to Cz
- Compute Current Source Density (CSD; spherical spline) from CAR

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
from typing import Iterable, Tuple, Optional
import numpy as np
import mne
from mne.preprocessing import compute_current_source_density

__all__ = ["prepare_eeg_and_car", "with_cz", "with_csd"]

# -------------------------------
# Step 1: prepare & CAR
# -------------------------------
def prepare_eeg_and_car(
    raw: mne.io.BaseRaw,
    *,
    exclude_chs: Iterable[str] = ("A1", "A2", "M1", "M2", "T1", "T2", "TP9", "TP10"),
    montage: str = "standard_1020",
    apply_10_20_alias: bool = True,
    verbose: bool = True,
) -> Tuple[mne.io.BaseRaw, mne.io.BaseRaw, Tuple[str, ...]]:
    """
    Prepare EEG (drop bads, attach montage, CAR reference).
    Returns (eeg_pre, eeg_car, used_channels).
    """
    eeg_pre = raw.copy().pick_types(eeg=True, eog=False, emg=False, meg=False)
    bads = list(eeg_pre.info.get("bads", []))
    if bads:
        eeg_pre.drop_channels(bads)
        if verbose: print(f"[prepare_eeg_and_car] Dropped bad channels: {bads}")

    # Normalize names
    alias = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    def _norm(n): 
        n = n.replace("EEG ", "").replace("-REF", "").replace("-Ref", "").strip()
        return alias.get(n, n) if apply_10_20_alias else n
    eeg_pre.rename_channels({ch: _norm(ch) for ch in eeg_pre.ch_names})

    # Montage
    eeg_pre.set_montage(montage, on_missing="ignore")

    # Filter valid channels
    ex = {x.upper() for x in exclude_chs}
    valid = []
    for ch in eeg_pre.ch_names:
        loc3 = eeg_pre.info["chs"][eeg_pre.ch_names.index(ch)]["loc"][:3]
        if (not np.allclose(loc3, 0.0)) and (not np.isnan(loc3).any()) and (ch.upper() not in ex):
            valid.append(ch)
    eeg_pre.pick_channels(valid)
    if verbose:
        print(f"[prepare_eeg_and_car] Valid channels: {len(valid)}")

    # CAR
    eeg_car = eeg_pre.copy().set_eeg_reference("average", projection=False)
    return eeg_pre, eeg_car, tuple(eeg_car.ch_names)

# -------------------------------
# Step 2: Cz reference
# -------------------------------
def with_cz(eeg_pre: mne.io.BaseRaw, used_channels: Tuple[str, ...], cz_name: str = "Cz") -> Optional[mne.io.BaseRaw]:
    """Return Cz-referenced EEG or None if Cz missing."""
    if cz_name not in eeg_pre.ch_names:
        print(f"[with_cz] '{cz_name}' not found; skipped.")
        return None
    eeg_cz = eeg_pre.copy().pick_channels(list(used_channels))
    eeg_cz.set_eeg_reference([cz_name], projection=False)
    return eeg_cz

# -------------------------------
# Step 3: CSD from CAR (V/m²)
# -------------------------------
def with_csd(eeg_car: mne.io.BaseRaw, verbose: bool = True, **kwargs) -> Optional[mne.io.BaseRaw]:
    """Compute Current Source Density (unit: V/m²)."""
    try:
        return compute_current_source_density(eeg_car.copy(), **kwargs)
    except Exception as e:
        if verbose:
            print("[with_csd] CSD computation failed:", repr(e))
        return None

"""
Usage
-----
>>> eeg_pre, eeg_car, used = prepare_eeg_and_car(raw)
>>> eeg_cz  = with_cz(eeg_pre, used)
>>> eeg_csd = with_csd(eeg_car)
"""
