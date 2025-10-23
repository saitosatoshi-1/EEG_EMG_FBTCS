# ========== 解析関数群 ==========
@lru_cache(maxsize=None)
def _bp_sos(sfreq, low, high, order=4):
    nyq = 0.5 * float(sfreq)
    return butter(int(order), [low/nyq, high/nyq], btype='band', output='sos')



def bandpass_1d(x, sfreq, low, high, order=4):
    sos = _bp_sos(float(sfreq), float(low), float(high), int(order))
    return sosfiltfilt(sos, x).astype(np.float32)



def compute_hilbert_envelope(signal, sfreq, band, order=4, smooth_sec=None):
    """
    1D: bandpass -> Hilbert -> |.|、必要なら移動平均で平滑化
    """
    low, high = band
    sig_f = bandpass_1d(np.asarray(signal, dtype=np.float32), sfreq, low, high, order)
    env = np.abs(hilbert(sig_f))
    if smooth_sec and smooth_sec > 0:
        win = max(1, int(sfreq * smooth_sec))
        k = np.ones(win, dtype=np.float32) / win
        env_s = np.convolve(env, k, mode='same')
        return env_s.astype(np.float32), env
    return env.astype(np.float32), env



def plot_dsa_db(ax, signal, sfreq, title='DSA (0–300 Hz, dB)',
                win_sec=0.5, overlap=0.5, fmax=300, cmap='jet',
                add_colorbar=False, fig=None, vmin=None, vmax=None):
    """
    DSA（dB）描画。win_sec/overlapを厳密に反映。
    vmin/vmax を与えると共通カラースケールで描画可能。
    """
    x = np.asarray(signal)
    nperseg  = max(8, int(sfreq * win_sec))
    noverlap = int(nperseg * overlap)

    if x is None or x.size < nperseg:
        ax.text(0.5, 0.5, 'Data too short', ha='center', va='center')
        ax.set_title(title)
        return None, None, None

    f, t_spec, Sxx = spectrogram(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                                 scaling='density', mode='psd', padded=True, boundary='zeros')
    Sxx_db = (10.0 * np.log10(Sxx + 1e-12)).astype(np.float32)

    if vmin is None or vmax is None:
        vmin = float(np.percentile(Sxx_db, 5))
        vmax = float(np.percentile(Sxx_db, 95))

    pcm = ax.pcolormesh(t_spec, f, Sxx_db, shading='gouraud', cmap=cmap,
                        vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_ylim(0, fmax)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)

    if add_colorbar and fig is not None:
        fig.colorbar(pcm, ax=ax, orientation='horizontal',
                     fraction=0.07, pad=0.15, label='Power (dB)')
    return f, t_spec, Sxx_db
