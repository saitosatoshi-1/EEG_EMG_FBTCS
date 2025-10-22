"""
Preprocessing: 64–256 Hz band-pass, Hilbert envelope, RMS & iRMS.
Corresponds to preprocessing used in Saito et al., Epilepsy Research (2025).
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils_signaltools import load_timeseries, bandpass, hilbert_envelope, rms, integrated_rms

def main(args):
    if args.demo:
        # Synthetic demo (5 s @ 1000 Hz)
        fs = 1000.0
        t = np.arange(0, 5, 1/fs)
        eeg = np.sin(2*np.pi*10*t) + 0.2*np.random.randn(t.size)
        emg = (np.random.randn(t.size) * np.sin(2*np.pi*120*t)) * 0.5
        data = np.stack([eeg, emg], axis=1)
        ch = ["EEG_T3-T4", "EMG_deltoid"]
    else:
        data, fs, ch = load_timeseries(args.input, ch_names=args.channels)

    # High-frequency band (64–256 Hz)
    hf = bandpass(data, 64, 256, fs, order=4)
    env = hilbert_envelope(hf)  # per-channel
    env_eeg = env[:, 0]
    env_emg = env[:, 1] if env.shape[1] > 1 else None

    # RMS (global) and iRMS (time-integral)
    env_rms_eeg = float(rms(env_eeg))
    irms_eeg = integrated_rms(env_eeg, fs, t_start=args.t_start, t_end=args.t_end)
    print(f"[EEG] RMS={env_rms_eeg:.6f}, iRMS={irms_eeg:.6f}")

    if env_emg is not None:
        env_rms_emg = float(rms(env_emg))
        irms_emg = integrated_rms(env_emg, fs, t_start=args.t_start, t_end=args.t_end)
        print(f"[EMG] RMS={env_rms_emg:.6f}, iRMS={irms_emg:.6f}")

    if args.plot:
        t = np.arange(hf.shape[0]) / fs
        fig, ax = plt.subplots(2 if env_emg is not None else 1, 1, figsize=(10, 5), sharex=True)
        if not isinstance(ax, np.ndarray):
            ax = np.array([ax])
        ax[0].plot(t, env_eeg, lw=1)
        ax[0].set_title("EEG envelope (64–256 Hz)")
        if env_emg is not None:
            ax[1].plot(t, env_emg, lw=1)
            ax[1].set_title("EMG envelope (64–256 Hz)")
        for a in ax: a.set_xlabel("Time (s)"); a.set_ylabel("Amplitude (a.u.)")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="EEG/EMG preprocessing: 64–256 Hz envelope & RMS/iRMS.")
    p.add_argument("--input", type=str, help="Path to CSV/EDF. If omitted, use --demo.", default=None)
    p.add_argument("--channels", nargs="+", help="Channel names to pick (for EDF/CSV).", default=None)
    p.add_argument("--t_start", type=float, default=0.0, help="Start time (s) for iRMS.")
    p.add_argument("--t_end", type=float, default=None, help="End time (s) for iRMS.")
    p.add_argument("--plot", action="store_true", help="Show envelope plots.")
    p.add_argument("--demo", action="store_true", help="Run with synthetic demo data.")
    main(p.parse_args())
