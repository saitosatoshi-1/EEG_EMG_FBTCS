# EEG-Embedded EMG Analysis During Focal to Bilateral Tonic–Clonic Seizures

This repository contains the Python scripts used in the study:

**Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H.**  
*Electromyographic components contaminating the scalp EEG during focal to bilateral tonic–clonic seizures as potential markers for seizure detection and lateralization: an exploratory study.*  
Submitted to *Epilepsy Research* (2025).

---

### 🧠 Overview
This project characterizes high-frequency (64–256 Hz) electromyographic (EMG) components embedded in scalp electroencephalography (EEG) during focal-to-bilateral tonic–clonic seizures (FBTCS).  
The analysis pipeline includes:
- Band-pass filtering (64–256 Hz)  
- Hilbert-based RMS envelope computation  
- Power spectral density (PSD) and cross-correlation analysis  
- Quantitative indices (integrated RMS [iRMS] and 10–90 % rise slope)

---

### ⚙️ Requirements
```bash
pip install -r requirements.txt

python code/eeg_emg_preprocessing.py
python code/eeg_emg_psd_correlation.py
python code/eeg_emg_riseslope_analysis.py

---

### 📂 Data availability
No real EEG or EMG data are included in this repository due to patient confidentiality.  
All analysis scripts are fully executable using any EEG/EMG dataset in CSV or EDF format that follows the same channel structure and sampling frequency described in the manuscript.  
Synthetic or anonymized demonstration data may be made available upon reasonable request.

---

### 🧾 Ethical statement
All data analyzed in the study were fully anonymized before processing.  
The study was approved by the Institutional Review Board of the National Center of Neurology and Psychiatry (NCNP, approval A2025-037).

---

### 📜 License
This project is released under the MIT License (see `LICENSE`).

---

### 📚 Citation
If you use this code, please cite:  
Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H.  
*Epilepsy Research* (2025).

---

### 📘 Zenodo DOI
[![DOI](https://zenodo.org/badge/1081201518.svg)](https://doi.org/10.5281/zenodo.17421105)

