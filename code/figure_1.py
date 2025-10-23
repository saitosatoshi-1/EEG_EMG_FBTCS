# === EEG+EMG: T3 vs T4 比較（EMG, Cz, CAR, CSD 並び） =========================
def plot_figure2_triple_with_emg_median_freq(
    sigs_T3,  # {"Cz": array, "CAR": array, "CSD": array}  ※T3のみ
    sigs_T4,  # {"Cz": array, "CAR": array, "CSD": array}  ※T4のみ
    analysis_times, sfreq, title,
    show=False, order=("EMG", "Cz", "CAR", "CSD"),
    # EMG 左右（列"EMG"の左=emg_left, 右=emg_right）
    emg_left=None, emg_right=None,
    save_csv=False, csv_path=None,
    save_dir=True,
    constrained_layout=True,
    save_png_when_show=True, png_dpi=1000, png_dir="./figures",
    png_facecolor="w", png_edgecolor="w",
    # ★追加: 全チャネルの計算に使う辞書と対象チャネル
    eeg_cz_dict=None, eeg_car_dict=None, eeg_csd_dict=None,
    channels=('Fp1','Fp2','F7','F8','T3','T4','T5','T6','O1','O2'),
    box_aspect=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10        # 図全体の基本サイズ
    plt.rcParams["axes.titlesize"] = 12   # タイトル
    plt.rcParams["axes.labelsize"] = 11   # 軸ラベル
    plt.rcParams["legend.fontsize"] = 10   # 凡例
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    from scipy.signal import spectrogram, welch
    from scipy.integrate import trapezoid
    import pandas as pd
    import re, os


    
    # ---- helpers ----
    def _trim(x, n):
        x = np.asarray(x, dtype=np.float32)
        return x[:n] if x.size else x

    def _maxabs(*arrs):
        vals = [np.nanmax(np.abs(a)) for a in arrs if a is not None and len(a)]
        return float(max(vals)) if vals else 0.0

    def _rms_series_from_env(env_raw, win_sec=0.5, step_sec=0.25):
        if env_raw is None or len(env_raw) == 0:
            return [], []
        win  = int(round(win_sec  * sfreq))
        step = int(round(step_sec * sfreq))
        if win < 1 or step < 1 or len(env_raw) < win:
            return [], []
        out, tt = [], []
        last = len(env_raw) - win
        for start in range(0, last + 1, step):
            w = env_raw[start:start+win]
            out.append(float(np.sqrt(np.mean(w**2))))
            tt.append(analysis_times[start + win//2])
        return out, tt

    def _smooth(x, win_pts):
        win = max(1, int(win_pts))
        if win == 1 or len(x) == 0:
            return np.asarray(x, dtype=float)
        k = np.ones(win, dtype=float) / win
        return np.convolve(np.asarray(x, dtype=float), k, mode='same')

    def _safe_name(s):
        s = re.sub(r"[\\/:*?\"<>|]+", "_", str(s))
        s = re.sub(r"\s+", "_", s).strip("_")
        return s or "figure"

    
    # ---- layout/params ----
    order = ("EMG", "Cz", "CAR", "CSD")  # 固定
    analysis_times = np.asarray(analysis_times, dtype=float)
    n = len(analysis_times)
    
    # 2x2ブロック配置
    block_pos = {"EMG": (0, 0), "Cz": (0, 1), "CAR": (1, 0), "CSD": (1, 1)}
    
    def _disp(lbl: str) -> str:
        # 表示名マッピング：CAR→AV、CSD→SD（その他はそのまま）
        mapping = {"CAR": "AV", "CSD": "SD"}
        return mapping.get(lbl, lbl)

    
    rows_per_block = 3
    n_block_rows, n_block_cols = 2, 2
    
    
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(title, fontsize=10)
    gs_outer = fig.add_gridspec(
        n_block_rows * rows_per_block + 1,  # ← +1 行
        n_block_cols,
        hspace=0.6, wspace=0.2,
        height_ratios=[1,1,1,0.01, 1,1,1]  # 4行目(0.3)をスペースとして使う
    )

    def _maybe_set_aspect(ax):
        if box_aspect is not None:
            try:
                ax.set_box_aspect(box_aspect)
            except Exception:
                pass

    
        
    # ブロック内の段(A=0, B=1, C=2)から外側GridSpec上の(行,列)に変換
    def _cell(lbl, stage_idx):
        br, bc = block_pos[lbl]           # br=0:上段, br=1:下段
        row = br * rows_per_block + stage_idx
        if br == 1:
            row += 1  # 余白行ぶん、下段を +1 行オフセット
        col = bc
        return row, col



    hf_band = (64, 256)
    center  = 6
    fmax    = 300
    nperseg = max(8, int(0.5 * sfreq))   # 0.5s
    noverlap= int(0.5 * nperseg)

    # MedF（計算のみ）
    #win_ms, step_ms = 250, 125
    #win  = int(round(sfreq * win_ms / 1000.0))
    #step = int(round(sfreq * step_ms / 1000.0))

    # 結果（CSV互換フィールドも含む）
    result = {
        "RMS": {},
        "MedianFreq": {},
        "Analysis_duration_sec": 0.0,
        # 互換: 後段CSVコードが読む平均値格納先（※後で埋める）
        "Cz": {}, "CAR": {}, "CSD": {}
    }

    # 列ラベル → (左,右) 取得
    def _get_pair(lbl):
        if lbl == "EMG":
            if emg_left is None or emg_right is None:
                return np.asarray([], np.float32), np.asarray([], np.float32)
            return _trim(emg_left, n), _trim(emg_right, n)
        return _trim(sigs_T3.get(lbl, []), n), _trim(sigs_T4.get(lbl, []), n)

    def _unit(lbl):
        return 'μV/cm²' if lbl == 'CSD' else 'μV'

    
    # === A: Raw ===
    for lbl in order:
        r, c = _cell(lbl, stage_idx=0)  # A段=0
        gs = gs_outer[r, c].subgridspec(1, 2, wspace=0.2)
        axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1], sharey=axL)
        _maybe_set_aspect(axL); _maybe_set_aspect(axR)
        sigL, sigR = _get_pair(lbl)
        axL.plot(analysis_times, sigL, lw=0.7)
        axR.plot(analysis_times, sigR, lw=0.7)

        sideL, sideR = ("Left","Right") if lbl=="EMG" else ("T3","T4")
        lab = _disp(lbl)
        axL.set_title(f"A: Raw — {_disp(lbl)} ({sideL})")
        axR.set_title(f"A: Raw — {_disp(lbl)} ({sideR})")
        axL.set_ylabel(f"Amplitude ({_unit(lbl)})")
        axL.set_xlim(analysis_times[0], analysis_times[-1])
        axR.set_xlim(analysis_times[0], analysis_times[-1])
        m = _maxabs(sigL, sigR)
        if m > 0:
            ylim = (-1.1 * m, 1.1 * m)
            axL.set_ylim(*ylim); axR.set_ylim(*ylim)

    # === B: DSA（両パネル共通カラースケール） ===
    for lbl in order:
        r, c = _cell(lbl, stage_idx=1)  # B段=1
        gs = gs_outer[r, c].subgridspec(1, 2, wspace=0.2)
        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[0, 1], sharex=axL)
        _maybe_set_aspect(axL); _maybe_set_aspect(axR)
        sigL, sigR = _get_pair(lbl)

        def _dsa(sig):
            if sig.size >= nperseg:
                f, t_spec, Sxx = spectrogram(sig, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                                             scaling='density', mode='psd')
                m = f <= fmax
                f = f[m]
                Sxx_db = 10.0 * np.log10(Sxx[m] + 1e-12).astype(np.float32)
                dt = (t_spec[1] - t_spec[0]) if len(t_spec) > 1 else 1.0
                df = (f[1] - f[0]) if len(f) > 1 else 1.0
                t_edges = np.concatenate([t_spec - dt/2, [t_spec[-1] + dt/2]])
                f_edges = np.concatenate([f - df/2, [f[-1] + df/2]])
                return (Sxx_db, t_edges, f_edges)
            return (None, None, None)

        dL, tLe, fLe = _dsa(sigL)
        dR, tRe, fRe = _dsa(sigR)

        pool = []
        if dL is not None: pool.append(dL.ravel())
        if dR is not None: pool.append(dR.ravel())
        vmin = vmax = None
        if pool:
            pool = np.concatenate(pool)
            vmin = float(np.nanpercentile(pool, 5))
            vmax = float(np.nanpercentile(pool, 95))

        pcmL = pcmR = None
        if dL is not None:
            pcmL = axL.pcolormesh(tLe, fLe, dL, shading="flat", cmap="jet",
                                  vmin=vmin, vmax=vmax, rasterized=True)
            axL.set_ylim(0, fmax)
        else:
            axL.text(0.5, 0.5, "Data too short", ha="center", va="center", transform=axL.transAxes)

        if dR is not None:
            pcmR = axR.pcolormesh(tRe, fRe, dR, shading="flat", cmap="jet",
                                  vmin=vmin, vmax=vmax, rasterized=True)
            axR.set_ylim(0, fmax)
        else:
            axR.text(0.5, 0.5, "Data too short", ha="center", va="center", transform=axR.transAxes)

        sideL, sideR = ("Left","Right") if lbl=="EMG" else ("T3","T4")
        lab = _disp(lbl)
        axL.set_title(f"B: DSA — {lbl} ({sideL})")
        axR.set_title(f"B: DSA — {lbl} ({sideR})")
        axL.set_ylabel("Frequency (Hz)")
        axL.set_xlim(analysis_times[0], analysis_times[-1])
        axR.set_xlim(analysis_times[0], analysis_times[-1])

        mappable = pcmL if pcmL is not None else pcmR
        if mappable is not None:
            cax = axR.inset_axes([1.02, 0.0, 0.03, 1.0], transform=axR.transAxes)
            cb = plt.colorbar(mappable, cax=cax)
            cb.set_label("Power (dB)", fontsize=10)
            cb.ax.tick_params(labelsize=10)


    # === C: RMS(Hilbert-HF envelope) の表示（LF/HFプロットは廃止） ===
    for lbl in order:
        r, c = _cell(lbl, stage_idx=2)  # C段=2
        gs = gs_outer[r, c].subgridspec(1, 2, wspace=0.2)
        axL = fig.add_subplot(gs[0, 0])
        axR = fig.add_subplot(gs[0, 1], sharey=axL)
        _maybe_set_aspect(axL); _maybe_set_aspect(axR)
        sigL, sigR = _get_pair(lbl)

        
        # 変更後（envelopeは無平滑、RMSを1秒に）
        _, hfL_raw = compute_hilbert_envelope(sigL, sfreq, hf_band, order=4, smooth_sec=0.0)
        _, hfR_raw = compute_hilbert_envelope(sigR, sfreq, hf_band, order=4, smooth_sec=0.0)
        rmsL, tL = _rms_series_from_env(hfL_raw, win_sec=1.0, step_sec=0.25)
        rmsR, tR = _rms_series_from_env(hfR_raw, win_sec=1.0, step_sec=0.25)
        
    
        # 3) プロット（RMS(HF envelope)）
        axL.plot(tL, rmsL)
        axR.plot(tR, rmsR)
    
        sideL, sideR = ("Left","Right") if lbl == "EMG" else ("T3","T4")
        lab = _disp(lbl)
        axL.set_title(f"C: RMS(Hilbert-HF env) — {lbl} ({sideL})")
        axR.set_title(f"C: RMS(Hilbert-HF env) — {lbl} ({sideR})")
    
        # 軸・凡例
        axL.set_ylabel(f"RMS Envelope ({_unit(lbl)})")
        # X軸は解析範囲に合わせたい場合（※RMS時系列の端は窓長ぶん短くなります）
        axL.set_xlim(analysis_times[0], analysis_times[-1])
        axR.set_xlim(analysis_times[0], analysis_times[-1])
    
        # Yスケールを左右で共有
        ymax = 0.0
        for v in (rmsL, rmsR):
            if v is not None and len(v):
                ymax = max(ymax, float(np.nanmax(v)))
        if ymax > 0:
            axL.set_ylim(0, 1.1 * ymax)
            axR.set_ylim(0, 1.1 * ymax)
    
        #axL.legend(loc="upper right", fontsize=9)
    
        # 4) 集計（max/area/area_per_sec はRMSベースに更新）
        def _summ(r, t):
            if r is None or len(r) == 0:
                return dict(max=0.0, area=0.0, area_per_sec=0.0)
            r = np.asarray(r, dtype=float); t = np.asarray(t, dtype=float)
            mx   = float(np.nanmax(r))
            area = float(trapezoid(r, x=t)) if len(t) > 1 else 0.0
            dur  = float(t[-1] - t[0]) if len(t) > 1 else 0.0
            aps  = float(area / dur) if dur > 0 else 0.0
            return dict(max=mx, area=area, area_per_sec=aps)
    
        # 側（Left/Right）→ T3/T4 にマッピング（EMGも比較の都合でT3/T4表記に統一）
        result["RMS"].setdefault(lbl, {})["T3"] = _summ(rmsL, tL)
        result["RMS"].setdefault(lbl, {})["T4"] = _summ(rmsR, tR)




    # === 互換キー（Cz/CAR/CSD 直下）に平均値を格納（CSV互換用） ===
    def _avg(a, b):
        vals = [v for v in (a, b) if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan
    for ref in ("Cz", "CAR", "CSD"):
        rms_T3 = result["RMS"].get(ref, {}).get("T3", {})
        rms_T4 = result["RMS"].get(ref, {}).get("T4", {})
        result.setdefault(ref, {})
        result[ref]["RMS_area"] = _avg(rms_T3.get("area", np.nan), rms_T4.get("area", np.nan))
        result[ref]["RMS_area_per_sec"] = _avg(rms_T3.get("area_per_sec", np.nan), rms_T4.get("area_per_sec", np.nan))

    # 軸体裁
    for ax in fig.axes:
        ax.tick_params(axis="x", labelrotation=0)

    # ★PNG 保存先修正（png_dirを使う）
    if show:
        if save_png_when_show:
            os.makedirs(png_dir, exist_ok=True)
            fname = f"{_safe_name(title)}.png"
            out_path = os.path.join(png_dir, fname)  # ← 修正
            fig.savefig(out_path, dpi=png_dpi, bbox_inches="tight",
                        facecolor=png_facecolor, edgecolor=png_edgecolor)
            print(f"[Saved] {out_path} ({png_dpi} dpi)")
        plt.show(block=True)
    plt.close(fig)

    # 解析時間
    analysis_duration_sec = float(analysis_times[-1] - analysis_times[0]) if len(analysis_times) > 1 else np.nan
    result['Analysis_duration_sec'] = analysis_duration_sec





    
    # ★ここから: 一気通貫のCSV出力（関数内）
    if save_csv:
        assert csv_path is not None, "save_csv=True の場合は csv_path を指定してください。"
        # 必要辞書が無ければ例外（ch計算のため）
        assert eeg_cz_dict is not None and eeg_car_dict is not None and eeg_csd_dict is not None, \
            "eeg_cz_dict / eeg_car_dict / eeg_csd_dict を渡してください。"

        def _compute_rms_area(sig):
            if sig is None:
                return np.nan, np.nan
            sig = _trim(sig, n)
            if sig.size == 0:
                return np.nan, np.nan
                
            # HF Hilbert（rawをRMS化に使う）
            _, hf_raw = compute_hilbert_envelope(sig, sfreq, hf_band, order=4, smooth_sec=0.0)
            rms_vals, rms_t = _rms_series_from_env(hf_raw, win_sec=1.0, step_sec=0.25)

            if len(rms_vals) == 0:
                return np.nan, np.nan
            area = float(trapezoid(np.asarray(rms_vals, float), x=np.asarray(rms_t, float))) if len(rms_t) > 1 else np.nan
            aps  = float(area / analysis_duration_sec) if (analysis_duration_sec and np.isfinite(analysis_duration_sec) and analysis_duration_sec > 0) else np.nan
            return area, aps

        sig_dict_map = {"Cz": eeg_cz_dict, "CAR": eeg_car_dict, "CSD": eeg_csd_dict}
        rows = []
        for ch in channels:
            for ref in ("Cz","CAR","CSD"):
                area, aps = _compute_rms_area(sig_dict_map[ref].get(ch))
                rows.append({
                    "Channel": ch,
                    "Reference": ref,
                    "Analysis_duration_sec": analysis_duration_sec,
                    "RMS_area": area,
                    "RMS_area_per_sec": aps,
                })
        # EMG 2行
        # （plot_emg_figure_dual_column を呼ばず、同一手順でRMS_areaを算出）
        if emg_left is not None and emg_right is not None:
            for sig, name in ((emg_left,"EMG_Left"), (emg_right,"EMG_Right")):
                _, hf_raw = compute_hilbert_envelope(_trim(sig, n), sfreq, hf_band, order=4, smooth_sec=0.0)
                hf_raw = _trim(hf_raw, n)
                rms_vals, rms_t = _rms_series_from_env(hf_raw, win_sec=1.0, step_sec=0.25)
                if len(rms_vals) > 0 and len(rms_t) > 1 and analysis_duration_sec and np.isfinite(analysis_duration_sec) and analysis_duration_sec > 0:
                    area = float(trapezoid(np.asarray(rms_vals,float), x=np.asarray(rms_t,float)))
                    aps  = float(area / analysis_duration_sec)
                else:
                    area = np.nan; aps = np.nan
                rows.append({
                    "Channel": name,
                    "Reference": "EMG",
                    "Analysis_duration_sec": analysis_duration_sec,
                    "RMS_area": area,
                    "RMS_area_per_sec": aps,
                })

        df = pd.DataFrame(rows)
        for col in ("Analysis_duration_sec","RMS_area","RMS_area_per_sec"):
            df[col] = pd.to_numeric(df[col], errors='coerce').round(1)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[CSV saved] {csv_path}")

    return result


# T3/T4 の辞書
sigs_T3 = {"Cz": eeg_cz_dict.get("T3"), "CAR": eeg_car_dict.get("T3"), "CSD": eeg_csd_dict.get("T3")}
sigs_T4 = {"Cz": eeg_cz_dict.get("T4"), "CAR": eeg_car_dict.get("T4"), "CSD": eeg_csd_dict.get("T4")}

csv1_dir = os.path.expanduser("~/Desktop/EMG/Jupyter/plots/CSV1")
os.makedirs(csv1_dir, exist_ok=True)
csv_path = os.path.join(csv1_dir, f"RMS_area_{subject_id}.csv")

result_t3t4 = plot_figure2_triple_with_emg_median_freq(
    sigs_T3, sigs_T4,
    analysis_times=analysis_times, sfreq=sfreq,
    title="T3 vs T4 (Raw+DSA+Envelopes)",
    show=True,
    order=("EMG","Cz","CAR","CSD"),
    emg_left=emg_left, emg_right=emg_right,
    # ★ここから追加
    save_csv=True, csv_path=csv_path,
    eeg_cz_dict=eeg_cz_dict, eeg_car_dict=eeg_car_dict, eeg_csd_dict=eeg_csd_dict,
    channels=('Fp1','Fp2','F7','F8','T3','T4','T5','T6','O1','O2'),
    png_dir=save_dir, png_dpi=1000,
    box_aspect=0.5
)
