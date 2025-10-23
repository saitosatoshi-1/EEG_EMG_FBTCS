def plot_triple_minimal_T3T4(
    sigs_T3, sigs_T4, head_analysis_times, sfreq, title,
    save_dir=True,
    show=False, order=("Cz","CAR","CSD"),
    dsa_win_sec=0.25, dsa_overlap=0.9,
    rms_win_sec=0.5, rms_step_sec=0.25,
    constrained_layout=True,
    dsa_scale_mode="column",  # "column" = T3+T4で共通 / "axis" = T3/T4で独立
    min_rms_ylim=1e-3,        # 極小値のときの最低上限（見やすさ用）
    raw_decimate_threshold=20000,  # Rawの点数が多い時だけ間引く
    t3t4_gap_ratio=0.30,
):
    """
    T3 と T4 を列内で比較する 3段図（A:Raw, B:DSA, C:RMS）。
    - A, B, C の X 軸幅は A をマスターにして完全共有
    - C（RMS）は T3/T4 の y 軸スケールを共通化し，下限は 0 に固定
    - B（DSA）のカラーバーは列ごとに独立（dsa_scale_modeで設定可）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10        # 図全体の基本サイズ
    plt.rcParams["axes.titlesize"] = 12   # タイトル
    plt.rcParams["axes.labelsize"] = 11   # 軸ラベル
    plt.rcParams["legend.fontsize"] = 9   # 凡例
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    from scipy.signal import spectrogram, butter, sosfiltfilt, hilbert
    from scipy.integrate import trapezoid
    from numpy.lib.stride_tricks import sliding_window_view

    # ---------- 基本準備 ----------
    t = np.asarray(head_analysis_times, dtype=float)
    n = len(t)
    assert n > 1, "head_analysis_times が短すぎます。"

    def _trim(x):
        x = np.asarray(x, dtype=np.float32)
        return x[:n] if x.size else x

    def _pair(lbl):
        return _trim(sigs_T3.get(lbl, [])), _trim(sigs_T4.get(lbl, []))

    def _unit(lbl):
        return "μV/cm²" if lbl == "CSD" else "μV"

    def _bp_hilbert_env(x, band, order=4):
        if x is None or len(x) == 0:
            return np.array([], dtype=np.float32)
        low, high = band
        nyq = 0.5 * float(sfreq)
        sos = butter(order, [low/nyq, high/nyq], btype="band", output="sos")
        xf  = sosfiltfilt(sos, x)
        return np.abs(hilbert(xf)).astype(np.float32)

    dur = float(t[-1] - t[0])
    fmax = 300
    nperseg  = max(8, int(dsa_win_sec * sfreq))
    noverlap = int(nperseg * dsa_overlap)
    hf_band  = (64, 256)
    win  = max(1, int(round(rms_win_sec  * sfreq)))
    step = max(1, int(round(rms_step_sec * sfreq)))

    # DSA を 1回定義（ループ外）
    def _dsa(sig):
        if sig.size >= nperseg:
            f, tt, Sxx = spectrogram(sig, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                                     scaling="density", mode="psd")
            msk = f <= fmax; f = f[msk]
            Sxx_db = 10.0 * np.log10(Sxx[msk] + 1e-12).astype(np.float32)
            dt = (tt[1] - tt[0]) if len(tt) > 1 else 1.0
            df = (f[1]  - f[0])  if len(f)  > 1 else 1.0
            t_edges = np.concatenate([tt - dt/2, [tt[-1] + dt/2]])
            f_edges = np.concatenate([f  - df/2, [f[-1] + df/2]])
            # A の時間範囲に合わせて端を揃える（見かけの幅ズレ防止）
            t_edges[0]  = t[0]; t_edges[-1] = t[-1]
            return Sxx_db, t_edges, f_edges
        return None, None, None


    
    # ====== 追加：Raw段の共有y範囲（Cz & CAR 共通）を事前計算 ======
    share_for_raw = {"Cz", "CAR"}  # ここに含めたラベル同士で共通スケール
    raw_abs_max = {}
    for lbl in order:
        xL, xR = _pair(lbl)
        m = float(max(np.nanmax(np.abs(xL)) if xL.size else 0.0,
                      np.nanmax(np.abs(xR)) if xR.size else 0.0))
        raw_abs_max[lbl] = m
    shared_raw_max = max([raw_abs_max.get(lbl, 0.0) for lbl in share_for_raw]) if any(
        lbl in raw_abs_max for lbl in share_for_raw) else 0.0




    
    # ---------- Figure とレイアウト ----------
    fig = plt.figure(figsize=(6.0 * len(order), 10), constrained_layout=constrained_layout)
    fig.suptitle(title, fontsize=8)
    
    gs_outer = fig.add_gridspec(3, len(order), hspace=0.3, wspace=0.3)

    if constrained_layout:
        fig.set_constrained_layout_pads(w_pad=1.2, h_pad=1.2, wspace=0.7, hspace=0.7)

    #カラーバー幅
    CAX_RATIO = 0.05
    result = {"Analysis_duration_sec": dur, "by_side": {}}
    for lbl in order:
        result[lbl] = {}
        result["by_side"][lbl] = {}

    # X 軸マスター（A 行）
    masters = {}


    
    
    # ---------- A: Raw ----------
    for j, lbl in enumerate(order):
        gs = gs_outer[0, j].subgridspec(1, 4, wspace=0.2, width_ratios=[1, t3t4_gap_ratio, 1, CAX_RATIO])
        axL_A = fig.add_subplot(gs[0, 0])                 # T3
        fig.add_subplot(gs[0, 1]).axis("off")             # ★ GAP（非表示）
        axR_A = fig.add_subplot(gs[0, 2], sharey=axL_A)   # T4
        fig.add_subplot(gs[0, 3]).axis("off")             # CAXダミー（Rawは使わない）

        xL, xR = _pair(lbl)

        # 軽量化：点数が極端に多いときは等間引き
        if xL.size > raw_decimate_threshold:
            stride = int(np.ceil(xL.size / raw_decimate_threshold))
            axL_A.plot(t[::stride], xL[::stride], lw=0.7)
        else:
            axL_A.plot(t, xL, lw=0.7)

        if xR.size > raw_decimate_threshold:
            stride = int(np.ceil(xR.size / raw_decimate_threshold))
            axR_A.plot(t[::stride], xR[::stride], lw=0.7)
        else:
            axR_A.plot(t, xR, lw=0.7)

        axL_A.set_title(f"A: Raw — {lbl} (T3)")
        axR_A.set_title(f"A: Raw — {lbl} (T4)")
        axL_A.set_ylabel(f"Amplitude ({_unit(lbl)})")


        
        m_individual = raw_abs_max.get(lbl, 0.0)
        m = shared_raw_max if (lbl in share_for_raw) else m_individual
        if m > 0:
            axL_A.set_ylim(-1.1*m, 1.1*m)
            axR_A.set_ylim(-1.1*m, 1.1*m)

        axL_A.set_xlim(t[0], t[-1]); axR_A.set_xlim(t[0], t[-1])
        axL_A.margins(x=0); axR_A.margins(x=0)
        masters[lbl] = (axL_A, axR_A)

    # ---------- B: DSA（列ごと独立カラーバー） ----------
    for j, lbl in enumerate(order):
        gs = gs_outer[1, j].subgridspec(1, 4, wspace=0.2, width_ratios=[1, t3t4_gap_ratio, 1, CAX_RATIO])
        axL = fig.add_subplot(gs[0, 0], sharex=masters[lbl][0])   # T3
        fig.add_subplot(gs[0, 1]).axis("off")                     # ★ GAP
        axR = fig.add_subplot(gs[0, 2], sharex=masters[lbl][1])   # T4
        cax = fig.add_subplot(gs[0, 3])                           # カラーバー


        xL, xR = _pair(lbl)
        dL, tLe, fLe = _dsa(xL)
        dR, tRe, fRe = _dsa(xR)

        # vmin/vmax の決め方（列単位 or 軸単位）
        def _scale(d):
            if d is None: return None, None
            flat = d.ravel()
            return float(np.nanpercentile(flat, 5)), float(np.nanpercentile(flat, 95))

        if dsa_scale_mode == "axis":
            vminL, vmaxL = _scale(dL)
            vminR, vmaxR = _scale(dR)
        else:  # "column"
            vmin = vmax = None
            pool = []
            if dL is not None: pool.append(dL.ravel())
            if dR is not None: pool.append(dR.ravel())
            if pool:
                pool = np.concatenate(pool)
                vmin = float(np.nanpercentile(pool, 5))
                vmax = float(np.nanpercentile(pool, 95))
            vminL = vminR = vmin; vmaxL = vmaxR = vmax

        rep = None
        if dL is not None:
            rep = axL.pcolormesh(tLe, fLe, dL, shading="flat", cmap="jet",
                                 vmin=vminL, vmax=vmaxL, rasterized=True)
            axL.set_ylim(0, fmax)
        else:
            axL.text(0.5, 0.5, "Data too short", ha="center", va="center", transform=axL.transAxes)

        if dR is not None:
            rep = axR.pcolormesh(tRe, fRe, dR, shading="flat", cmap="jet",
                                 vmin=vminR, vmax=vmaxR, rasterized=True)
            axR.set_ylim(0, fmax)
        else:
            axR.text(0.5, 0.5, "Data too short", ha="center", va="center", transform=axR.transAxes)

        axL.set_xlim(t[0], t[-1]); axR.set_xlim(t[0], t[-1])
        axL.margins(x=0); axR.margins(x=0)
        axL.set_title(f"B: DSA — {lbl} (T3)")
        axR.set_title(f"B: DSA — {lbl} (T4)")
        axL.set_ylabel("Frequency (Hz)")

        if rep is not None:
            cb = fig.colorbar(rep, cax=cax)
            cb.set_label("Power (dB)", fontsize=9, labelpad=4)
            cb.ax.tick_params(labelsize=9, pad=2)

    # ---------- C: RMS(HF Hilbert) ----------
    def _rms_from_env(env_raw):
        if env_raw is None or len(env_raw) < win:
            return np.array([]), np.array([])
        W = sliding_window_view(env_raw, window_shape=win)[::step]
        if W.shape[0] < 2:
            return np.array([]), np.array([])
        rms  = np.sqrt(np.mean(W*W, axis=1)).astype(float)
    
        # ★ 追加：RMS系列を0.25 s 平滑（rms_step_sec 基準で秒→ポイント換算）
        smooth_sec = 0.25
        smooth_pts = max(1, int(round(smooth_sec / rms_step_sec)))  # 例: 0.25/0.25=1
        if smooth_pts > 1:
            k = np.ones(smooth_pts) / smooth_pts
            rms_s = np.convolve(rms, k, mode="same")
        else:
            rms_s = rms
    
        centers_idx = np.arange(win//2, win//2 + step*len(W), step, dtype=int)
        t_centers = t[centers_idx]
        offset = (len(rms) - len(rms_s)) // 2
        t_rs = t_centers[offset : offset + len(rms_s)]
        if (len(t_rs) != len(rms_s)) or (len(rms_s) < 2):
            t_rs = t_centers[:len(rms)]; rms_s = rms[:len(t_rs)]
        return rms_s, t_rs


    def _summ(rms_s, tt):
        if len(rms_s) < 2:
            return dict(RMS_area=0.0, RMS_area_per_sec=0.0, Rise_time=0.0, Rise_slope=0.0), (0, 0)
        try:
            area = float(trapezoid(rms_s, x=tt)); dur  = float(tt[-1] - tt[0])
        except ValueError:
            dx = np.median(np.diff(tt)) if len(tt) > 1 else (step/sfreq)
            area = float(trapezoid(rms_s, dx=dx)); dur = dx * (len(rms_s)-1)
        aps = area / max(dur, 1e-12)
        rmin, rmax = float(np.min(rms_s)), float(np.max(rms_s))
        if rmax > rmin:
            th10 = rmin + 0.1*(rmax-rmin); th90 = rmin + 0.9*(rmax-rmin)
            i10 = int(np.argmax(rms_s >= th10))
            i90 = max(int(np.argmax(rms_s >= th90)), i10+1)
            rise_time  = float(tt[i90] - tt[i10])
            rise_slope = float((rms_s[i90] - rms_s[i10]) / max(rise_time, 1e-12))
        else:
            i10 = i90 = 0; rise_time = rise_slope = 0.0
        return dict(RMS_area=area, RMS_area_per_sec=aps,
                    Rise_time=rise_time, Rise_slope=rise_slope), (i10, i90)


    # ====== 追加：RMS段の共有y範囲（Cz & CAR 共通）を事前計算 ======
    share_for_rms = {"Cz", "CAR"}  # ここに含めたラベル同士で共通スケール
    rms_abs_max = {}
    for lbl in order:
        xL, xR = _pair(lbl)
        envL = _bp_hilbert_env(xL, hf_band); envR = _bp_hilbert_env(xR, hf_band)
        rmsL, _tL = _rms_from_env(envL);     rmsR, _tR = _rms_from_env(envR)
        m = 0.0
        if len(rmsL):
            m = max(m, float(np.nanmax(rmsL)))
        if len(rmsR):
            m = max(m, float(np.nanmax(rmsR)))
        rms_abs_max[lbl] = m

    shared_rms_max = max([rms_abs_max.get(lbl, 0.0) for lbl in share_for_rms]) if any(
        lbl in rms_abs_max for lbl in share_for_rms) else 0.0
    # ===============================================================






    
    bottom_axes = []
    for j, lbl in enumerate(order):
        gs = gs_outer[2, j].subgridspec(1, 4, wspace=0.2, width_ratios=[1, t3t4_gap_ratio, 1, CAX_RATIO])
        axL = fig.add_subplot(gs[0, 0], sharex=masters[lbl][0])   # T3
        fig.add_subplot(gs[0, 1]).axis("off")                     # ★ GAP
        axR = fig.add_subplot(gs[0, 2], sharex=masters[lbl][1])   # T4
        fig.add_subplot(gs[0, 3]).axis("off")                     # CAXダミー


        xL, xR = _pair(lbl)
        envL = _bp_hilbert_env(xL, hf_band); envR = _bp_hilbert_env(xR, hf_band)
        rmsL, tL = _rms_from_env(envL);      rmsR, tR = _rms_from_env(envR)

        summL, (i10L, i90L) = _summ(rmsL, tL)
        summR, (i10R, i90R) = _summ(rmsR, tR)
        result["by_side"][lbl]["T3"] = summL
        result["by_side"][lbl]["T4"] = summR
        for k in ("RMS_area","RMS_area_per_sec","Rise_time","Rise_slope"):
            vL, vR = summL.get(k, np.nan), summR.get(k, np.nan)
            vs = [v for v in (vL, vR) if np.isfinite(v)]
            result[lbl][k] = float(np.mean(vs)) if vs else np.nan

        # プロット
        if len(rmsL) >= 2:
            axL.plot(tL, rmsL, lw=1.0, label = "off")
            if len(rmsL) > i90L and len(tL) > i90L:
                axL.axvline(tL[i10L], ls='--', alpha=0.7)
                axL.axvline(tL[i90L], ls='--', alpha=0.7)
                axL.plot(tL[i10L:i90L+1], rmsL[i10L:i90L+1], lw=3)
        else:
            axL.text(0.5, 0.5, "Data too short", ha="center", va="center", transform=axL.transAxes)

        if len(rmsR) >= 2:
            axR.plot(tR, rmsR, lw=1.0)
            if len(rmsR) > i90R and len(tR) > i90R:
                axR.axvline(tR[i10R], ls='--', alpha=0.7)
                axR.axvline(tR[i90R], ls='--', alpha=0.7)
                axR.plot(tR[i10R:i90R+1], rmsR[i10R:i90R+1], lw=3)
        else:
            axR.text(0.5, 0.5, "Data too short", ha="center", va="center", transform=axR.transAxes)

        #axL.legend(loc="upper right", fontsize=9)
        axL.set_ylabel("RMS (μV/cm²)" if lbl=="CSD" else "RMS (μV)")
        axL.set_title(f"C: RMS(HF Hilbert) — {lbl} (T3)")
        axR.set_title(f"C: RMS(HF Hilbert) — {lbl} (T4)")
        axL.set_xlim(t[0], t[-1]); axR.set_xlim(t[0], t[-1])
        axL.margins(x=0); axR.margins(x=0)
        bottom_axes.extend([axL, axR])

        # --- 共通 y 軸スケール設定（T3/T4 共通・下限=0・最小上限ガード） ---
        ymax = 0.0
        for v in (rmsL, rmsR):
            if len(v):
                vmax = float(np.nanmax(v))
                if np.isfinite(vmax):
                    ymax = max(ymax, vmax)
        m_individual = rms_abs_max.get(lbl, 0.0)
        m = shared_rms_max if (lbl in share_for_rms) else m_individual
        ymax = max(m, float(min_rms_ylim))
        axL.set_ylim(0, 1.1*ymax)
        axR.set_ylim(0, 1.1*ymax)

    for ax in bottom_axes:
        ax.set_xlabel("Time (s)")


    if show:
        import re, os
        os.makedirs(save_dir, exist_ok=True)

        # タイトルを安全なファイル名に変換
        safe_title = re.sub(r'[\\/:*?"<>|]+', '_', str(title))
        safe_title = re.sub(r'\s+', '_', safe_title).strip('_')

        # subject_id を先頭につける
        fname = f"{subject_id}_{safe_title}.png"
        out_path = os.path.join(save_dir, fname)

        fig.savefig(out_path, dpi=600, bbox_inches="tight",
                    facecolor="w", edgecolor="w")
        print(f"[Saved PNG] {out_path}")

        plt.show(block=True)

    plt.close(fig)
    return result










# ========= 呼び出し例（T3/T4可視化 & 全Electrode集計） ======================
# head区間の用意
# head区間
head_tonic_end_idx = int(head_tonic_duration_sec * sfreq)
head_analysis_times = analysis_times[:head_tonic_end_idx]

# 可視化（例：T3/T4）
# 可視化（T3/T4 を1枚で比較）
sigs_T3 = {
    "Cz":  eeg_cz_dict["T3"] [:head_tonic_end_idx],
    "CAR": eeg_car_dict["T3"][:head_tonic_end_idx],
    "CSD": eeg_csd_dict["T3"][:head_tonic_end_idx],
}
sigs_T4 = {
    "Cz":  eeg_cz_dict["T4"] [:head_tonic_end_idx],
    "CAR": eeg_car_dict["T4"][:head_tonic_end_idx],
    "CSD": eeg_csd_dict["T4"][:head_tonic_end_idx],
}

res = plot_triple_minimal_T3T4(
    sigs_T3=sigs_T3,
    sigs_T4=sigs_T4,
    head_analysis_times=head_analysis_times,
    sfreq=sfreq,
    rms_win_sec=1.0, rms_step_sec=0.25,
    show=True,
    title=None,
    save_dir=save_dir,
    order=("Cz","CAR","CSD"),
)
print("=== T3/T4 stats (per-side in res['by_side']) ===")








# 集計CSV（そのまま使えます：戻り値のキーは従来と同じ）
channels = ['Fp1','Fp2','F7','F8','T3','T4','T5','T6','O1','O2']
results = []

for ch in channels:
    sig_car = eeg_car_dict.get(ch)
    sig_csd = eeg_csd_dict.get(ch)
    sig_cz  = eeg_cz_dict.get(ch)
    have_all = (sig_car is not None) and (sig_csd is not None) and (sig_cz is not None)

    if have_all:
        n = head_tonic_end_idx  # 解析長
        # 左右（T3/T4）同一にして T3T4関数へ委譲（図はshow=Falseなので左右同一でOK）
        sigs_T3 = {
            "Cz":  np.asarray(sig_cz [:n], dtype=np.float32),
            "CAR": np.asarray(sig_car[:n], dtype=np.float32),
            "CSD": np.asarray(sig_csd[:n], dtype=np.float32),
        }
        sigs_T4 = {k: v.copy() for k, v in sigs_T3.items()}

        res = plot_triple_minimal_T3T4(
            sigs_T3=sigs_T3,
            sigs_T4=sigs_T4,
            head_analysis_times=head_analysis_times,
            sfreq=sfreq,
            title=f"{ch} (raw + DSA + HF RMS w/ rise)",
            show=False,
            save_dir=save_dir,
            order=("Cz","CAR","CSD")
        )
        dur = res.get('Analysis_duration_sec', np.nan)
    else:
        # 欠損チャネルもCSVに必ず含める
        res = {}
        dur = np.nan

    for ref in ['CAR','CSD','Cz']:
        metrics = res.get(ref, {})
        results.append({
            "Channel": ch,
            "Reference": ref,
            "Analysis_duration_sec": dur,
            "RMS_area": metrics.get('RMS_area', np.nan),
            "RMS_area_per_sec": metrics.get('RMS_area_per_sec', np.nan),
            "Rise_time": metrics.get('Rise_time', np.nan),
            "Rise_slope": metrics.get('Rise_slope', np.nan),
        })






df_head = pd.DataFrame(results)
for col in ["Analysis_duration_sec","RMS_area","RMS_area_per_sec","Rise_time","Rise_slope"]:
    if col in df_head.columns:
        df_head[col] = pd.to_numeric(df_head[col], errors='coerce').round(3)



base_dir = os.path.expanduser("~/Desktop/EMG/Jupyter/plots")
csv3_dir = os.path.join(base_dir, "CSV3")
os.makedirs(csv3_dir, exist_ok=True)

csv_path = os.path.join(csv3_dir, f"head_{subject_id}.csv")
df_head.to_csv(csv_path, index=False)

print(f"Head結果をCSVに保存しました: {csv_path}")
