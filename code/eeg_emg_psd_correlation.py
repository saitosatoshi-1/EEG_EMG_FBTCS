"""
PSD similarity (Welch) and envelope cross-correlation between EEG and EMG.
Corresponds to Figure 3 core analysis in Saito et al., Epilepsy Research (2025).
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils_signaltools import load_timeseries, bandpass, hilbert_envelope, psd, psd_similarity

def normalized(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x)
    s = np.std(x)
    return x / s if s > 0 else x

def main(args):
    if args.demo:
        fs = 1000.0
        t = np.arange(0, 5, 1/fs)
        eeg = np.sin(2*np.pi*10*t) + 0.2*np.random.randn(t.size)
        emg = np.random.randn(t.size) * np.sin(2*np.pi*120*t) * 0.5
        data = np.stack([eeg, emg], axis=1)
        ch = ["EEG", "EMG"]
    else:
        data, fs, ch = load_timeseries(args.input, ch_names=args.channels)

    if data.shape[1] < 2:
        raise ValueError("Need at least two channels (EEG and EMG).")

    # PSD similarity (raw broadband)
    r_psd = psd_similarity(data[:, 0], data[:, 1], fs, nperseg=args.nperseg)
    print(f"PSD correlation (EEG vs EMG): r = {r_psd:.4f}")

    # Envelope cross-correlation in 64–256 Hz
    hf = bandpass(data, 64, 256, fs, order=4)
    env = hilbert_envelope(hf)
    e_eeg = normalized(env[:, 0])
    e_emg = normalized(env[:, 1])

    # Cross-correlation (normalized) and max lag
    xcorr = np.correlate(e_eeg, e_emg, mode="full")
    lags = np.arange(-len(e_eeg)+1, len(e_eeg))
    max_idx = int(np.argmax(xcorr))
    max_lag_samples = lags[max_idx]
    max_lag_ms = 1000.0 * max_lag_samples / fs
    r_env = float(np.max(xcorr) / (len(e_eeg)))  # normalized by N (since signals z-scored)
    print(f"Envelope cross-corr: r_max ≈ {r_env:.4f} at lag = {max_lag_ms:.1f} ms")

    if args.plot:
        f, pxx_eeg = psd(data[:, 0], fs, nperseg=args.nperseg)
        _, pxx_emg = psd(data[:, 1], fs, nperseg=args.nperseg)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(f, pxx_eeg, lw=1, label="EEG"); ax[0].plot(f, pxx_emg, lw=1, label="EMG")
        ax[0].set_title("Welch PSD"); ax[0].set_xlabel("Frequency (Hz)"); ax[0].set_ylabel("Power"); ax[0].legend()
        ax[1].plot(lags / fs * 1000.0, xcorr, lw=1)
        ax[1].axvline(max_lag_ms, ls="--"); ax[1].set_title("Envelope cross-correlation (64–256 Hz)")
        ax[1].set_xlabel("Lag (ms)"); ax[1].set_ylabel("Correlation (a.u.)")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="EEG-EMG PSD similarity & envelope cross-correlation.")
    p.add_argument("--input", type=str, default=None, help="CSV/EDF path. Use --demo if omitted.")
    p.add_argument("--channels", nargs="+", default=None, help="Channel names to pick (EDF/CSV).")
    p.add_argument("--nperseg", type=int, default=1024, help="Welch segment length.")
    p.add_argument("--plot", action="store_true", help="Plot PSD and cross-correlation.")
    p.add_argument("--demo", action="store_true", help="Use synthetic demo data.")
    main(p.parse_args())
