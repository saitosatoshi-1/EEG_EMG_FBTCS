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
- Quantitative indices (integrated RMS [iRMS] and 10–90% rise slope)

All scripts are designed for **reproducibility** and **transparency**, with no patient data included.

---

### ⚙️ Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

---

🧩 Code structure

`code/eeg_edf_utils.py`	
EDF loader and preprocessing utilities (channel renaming, notch/high-pass, cropping).

`code/eeg_montage_utils.py`
Montage preparation (CAR, CSD, Cz reference).

`code/signal_analysis_utils.py`	
Core filtering, Hilbert envelope, and DSA (spectrogram) functions.

`code/eeg_emg_psd_core.py`
Log–log PSD shape-correlation between EEG and EMG (r, p).

`code/eeg_emg_xcorr_core.py`
RMS cross-correlation to estimate EEG–EMG lag direction.

`code/eeg_emg_riseslope_core.py`
RMS-based indices (iRMS area, 10–90% rise time and slope).

Each file contains only analysis logic — no plotting or I/O — for safe, fully reproducible research use.

---

💻 Example usage
1. Compute PSD shape correlation

```python
from eeg_emg_psd_core import compute_psd_shape_corr
r, p = compute_psd_shape_corr(eeg, emg, sfreq=1000.0, hf_band=(64,256))
print(r, p)
```


2. Cross-correlation of RMS envelopes

```python
from eeg_emg_xcorr_core import compute_rms_crosscorr
res = compute_rms_crosscorr(eeg, emg, sfreq=1000)
print(res)
```


3. RMS rise-slope metrics
   
```python
from eeg_emg_riseslope_core import compute_hf_rms_metrics
m = compute_hf_rms_metrics(signal, sfreq=1000.0, hf_band=(64,256))
print(m)
```

---

📂 Data availability

No real EEG or EMG data are included in this repository due to patient confidentiality.
All analysis scripts are fully executable using any EEG/EMG dataset in CSV or EDF format that follows the same channel structure and sampling frequency described in the manuscript.

---

🧾 Ethical statement
All data analyzed in the study were fully anonymized before processing.
The study was approved by the Institutional Review Board of the National Center of Neurology and Psychiatry (NCNP, approval A2025-037).

---

📜 License
This project is released under the MIT License (see LICENSE).

---

📚 Citation
If you use this code, please cite:
Saito S, Kuramochi I, Taniguchi G, Kondo S, Tanaka H.
Electromyographic components contaminating the scalp EEG during focal to bilateral tonic–clonic seizures as potential markers for seizure detection and lateralization: an exploratory study.
Epilepsy Research (2025).

---

📘 Zenodo DOI
https://doi.org/10.5281/zenodo.17421105
