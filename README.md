# EEG-Embedded EMG Analysis During Focal to Bilateral Tonic–Clonic Seizures

This repository provides the reproducible Python analysis pipeline used in:

**Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H.**  
*Electromyographic components contaminating the scalp EEG during focal to bilateral tonic–clonic seizures as potential markers for seizure detection and lateralization: an exploratory study.*  
Submitted to *Epilepsy Research* (2025).

---

### 🧠 Overview
This project characterizes **high-frequency (64–256 Hz) electromyographic (EMG) components embedded in scalp EEG** during focal-to-bilateral tonic–clonic seizures (FBTCS).  
The aim is to determine whether these “artifact” components can provide **supplementary biomarkers for seizure detection and lateralization**, particularly when reliable video monitoring is unavailable.

Key quantitative measures:
- Power spectral density (PSD) correlation between EEG and deltoid EMG  
- Cross-correlation of RMS envelopes (temporal similarity)  
- Integrated RMS (iRMS) and 10–90 % rise slope (tonic contraction rate)

All scripts are designed for **open, safe, and patient-free** analysis and reproduce the procedures used for Figures 2–5 of the manuscript.

---

### ⚙️ Requirements
Install dependencies:
```bash
pip install numpy scipy matplotlib mne
```

---

### 🧩 Code structure

- `code/eeg_edf_utils.py`	-

Safe EDF loader, channel normalization, high-pass / notch filters

*Methods: Data acquisition and preprocessing*

- `code/eeg_montage_utils.py` -

Common Average Reference (CAR), Cz-reference, and Current Source Density (CSD) montages

*Methods: Montage comparison (Cz, average, Laplacian)*

- `code/signal_analysis_utils.py`	-

Core filtering, Hilbert envelope, and dynamic spectral analysis (DSA) plotting

*Figures 1–2*

- `code/eeg_emg_psd_core.py` -

Log-log PSD shape correlation between EEG and EMG within 64–256 Hz band

*Figure 3*

- `code/eeg_emg_xcorr_core.py` -

RMS cross-correlation and lag estimation (EEG ↔ EMG)

*Figure 3*

- `code/eeg_riseslope_core.py` -

iRMS area and 10–90 % rise-slope computation

*Figure 4–5*

All modules are analysis-only (no patient data, plotting optional).

---

### 📂 Data availability

No real EEG or EMG data are included in this repository due to patient confidentiality.

---

### 🧾 Ethical statement

All data analyzed in the study were fully anonymized before processing.
The study was approved by the Institutional Review Board of the National Center of Neurology and Psychiatry (NCNP, approval A2025-037).

---

### 📜 License

Released under the MIT License.

---

### 📚 Citation

If you use this code, please cite:
Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H.
Electromyographic components contaminating the scalp EEG during focal to bilateral tonic–clonic seizures as potential markers for seizure detection and lateralization: an exploratory study.
Epilepsy Research (2025).

---

### 📘 Zenodo DOI

https://doi.org/10.5281/zenodo.17421104
