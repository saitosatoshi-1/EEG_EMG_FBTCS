"""
Rise-slope (10–90%) analysis on 64–256 Hz envelope.
Corresponds to Figure 5 core metric in Saito et al., Epilepsy Research (2025).
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils_signaltools import load_timeseries, bandpass, hilbert_envelope, rise_slope_10_90

def main(args):
    if args.demo:
        fs = 1000.0
        t = np.arange(0, 5, 1/fs)
        # EEG envelope-like synthetic: gradual rise
        eeg = 0.1*np.random.randn(t.size) + 0.2*np.sin(2*np.pi*10*t)
        emg = 0.1*np.random.randn(t.size) + (t > 2.0) * 0.8*np.sin(2*np.pi*120*t)  # onset ~2s
        data = np.stack([eeg, emg], axis=1)
        ch = ["EEG", "EMG"]
    else:
        data, fs, ch = load_timeseries(args.input, ch_names=args.channels)

    hf = bandpass(data, 64, 256, fs, order=4)
    env = hilbert_envelope(hf)

    results = []
    for i in range(env.shape[1]):
        slope, t10, t90 = rise_slope_10_90(env[:, i], fs)
        results.append((ch[i] if i < len(ch) else f"ch{i}", slope, t10, t90))
        print(f"{results[-1][0]}: slope={slope:.6f} a.u./s, t10={t10:.3f}s, t90={t90:.3f}s")

    if args.plot:
        import pandas as pd
        t = np.arange(env.shape[0]) / fs
        n = env.shape[1]
        fig, ax = plt.subplots(n, 1, figsize=(10, 4+2*n), sharex=True)
        if not isinstance(ax, np.ndarray): ax = np.array([ax])
        for i in range(n):
            slope, t10, t90 = rise_slope_10_90(env[:, i], fs)
            ax[i].plot(t, env[:, i], lw=1)
            if np.isfinite(t10): ax[i].axvline(t10, ls="--")
            if np.isfinite(t90): ax[i].axvline(t90, ls="--")
            ax[i].set_title(f"{results[i][0]} envelope (64–256 Hz), slope={slope:.3f} a.u./s")
            ax[i].set_ylabel("Amplitude (a.u.)")
        ax[-1].set_xlabel("Time (s)")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rise-slope (10–90%) on 64–256 Hz envelope.")
    p.add_argument("--input", type=str, default=None, help="CSV/EDF path. If omitted, use --demo.")
    p.add_argument("--channels", nargs="+", default=None, help="Channel names to pick.")
    p.add_argument("--plot", action="store_true", help="Plot envelopes with 10/90% markers.")
    p.add_argument("--demo", action="store_true", help="Use synthetic demo data.")
    main(p.parse_args())
