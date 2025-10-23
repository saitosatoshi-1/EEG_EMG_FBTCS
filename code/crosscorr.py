# === RMS クロスコリレーション（CSV行を返す版） =========================
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10        # 図全体の基本サイズ
plt.rcParams["axes.titlesize"] = 12   # タイトル
plt.rcParams["axes.labelsize"] = 11   # 軸ラベル
plt.rcParams["legend.fontsize"] = 9   # 凡例
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
from scipy.stats import pearsonr

def plot_rms_crosscorr_4x3_to_pdf(
    eeg_car_dict, eeg_csd_dict, eeg_cz_dict,
    emg_left, emg_right, sfreq,
    show=False, save_dir=True,
    order=('Cz','CAR','CSD'),
    figsize=(11, 10),
    width_ratios=(1.0, 1.0, 1.0),
    height_ratios=(1.0, 1.0, 1.0, 1.0),
    box_aspect=0.65,   # 例: 1.0 で各Axesを正方形っぽく 
):
    win_sec=1.0; step_sec=0.25
    win_samples=int(win_sec*sfreq); step_samples=int(step_sec*sfreq)
    smooth_win=int(6.0/step_sec)
    max_lag_sec=30.0; max_lag=int(max_lag_sec/step_sec)

    chs=['T3','T4']
    refs=list(order)
    eeg_refs={'CAR':eeg_car_dict,'CSD':eeg_csd_dict,'Cz':eeg_cz_dict}
    emg_dict={'T3':emg_left,'T4':emg_right}

    def smooth(x,w):
        if w<=1: return x
        return np.convolve(x, np.ones(w)/w, mode='same')

    def running_pearsonr(x,y,maxlag):
        lags=np.arange(-maxlag,maxlag+1); rs=[]
        for lag in lags:
            if lag<0: xx=x[:lag]; yy=y[-lag:]
            elif lag>0: xx=x[lag:]; yy=y[:-lag]
            else: xx=x; yy=y
            rs.append(np.corrcoef(xx,yy)[0,1] if (len(xx)==len(yy) and len(xx)>2) else np.nan)
        return np.array(rs), lags

    def compute_hf_rms(sig):
        band = (64, 256)
        # ★ envelope平滑はしない（Methodsに合わせる）
        _, env = compute_hilbert_envelope(sig, sfreq, band, order=4, smooth_sec=0.0)
    
        rms = []
        for i in range(0, len(env) - win_samples, step_samples):
            w = env[i:i + win_samples]
            rms.append(float(np.sqrt(np.mean(w * w))))
        rms = np.array(rms, dtype=float)
        if rms.size == 0:
            return rms
    
        # 0–1正規化 → 6秒平滑は維持（Methods準拠）
        rms_min, rms_max = float(np.min(rms)), float(np.max(rms))
        rms = (rms - rms_min) / (rms_max - rms_min + 1e-8)
        return smooth(rms, smooth_win)


    
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs  = fig.add_gridspec(4, 3,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios)
    axs = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(4)])
    if box_aspect is not None:
        for ax in axs.ravel():
            ax.set_box_aspect(box_aspect)
    
    rows_xcorr = []  # ★ 追加：返却データをためる

    for col,ref in enumerate(refs):
        for row,ch in zip([0,2], chs):
            eeg = eeg_refs[ref].get(ch, None); emg = emg_dict[ch]
            if eeg is None or emg is None:
                axs[row,col].text(0.5,0.5,'No data',ha='center',va='center')
                continue
            x = compute_hf_rms(eeg); y = compute_hf_rms(emg)
            if x.size==0 or y.size==0:
                axs[row,col].text(0.5,0.5,'Data too short',ha='center',va='center')
                continue
            t = np.arange(len(x))*step_sec
            ax = axs[row,col]
            ax.plot(t, x)
            ax.plot(t, y)
            ax.set_xlim(0, t[-1] if t.size else 1.0)
            ymin, ymax = min(x.min(), y.min()), max(x.max(), y.max())
            ax.set_ylim(ymin*1.1, ymax*1.1)
            ax.set_ylabel('Normalized RMS (0-1)')
            ax.set_title(f'{ref} - {ch} Waveform')
            #if col==0: ax.legend(loc='upper right')

        for row,ch in zip([1,3], chs):
            eeg = eeg_refs[ref].get(ch, None); emg = emg_dict[ch]
            ax = axs[row,col]
            if eeg is None or emg is None:
                ax.text(0.5,0.5,'No data',ha='center',va='center')
                # 返す行も入れておく
                rows_xcorr.append({
                    'metric':'XCorr_HF_RMS','ref':ref,'ch':ch,
                    'r_max':np.nan,'lag_sec':np.nan,'lead':np.nan,
                    'win_sec':win_sec,'step_sec':step_sec
                })
                continue

            x = compute_hf_rms(eeg); y = compute_hf_rms(emg)
            if x.size==0 or y.size==0:
                ax.text(0.5,0.5,'Data too short',ha='center',va='center')
                rows_xcorr.append({
                    'metric':'XCorr_HF_RMS','ref':ref,'ch':ch,
                    'r_max':np.nan,'lag_sec':np.nan,'lead':np.nan,
                    'win_sec':win_sec,'step_sec':step_sec
                })
                continue

            cs, lags = running_pearsonr(x, y, max_lag)
            k = np.nanargmax(cs); r_max = cs[k]; lag_sec = lags[k]*step_sec
            # p値用にラグ整列
            if lags[k] > 0: xx, yy = x[lags[k]:], y[:-lags[k]]
            elif lags[k] < 0: xx, yy = x[:lags[k]], y[-lags[k]:]
            else: xx, yy = x, y
            p_val = pearsonr(xx, yy)[1] if (len(xx)>2 and len(xx)==len(yy)) else np.nan
            lead = "EMG→EEG" if lag_sec > 0 else ("EEG→EMG" if lag_sec < 0 else "同期")

            ax.plot(lags*step_sec, cs, 'k-')
            ax.axvline(0, color='gray', linestyle=':')
            ax.axvline(lag_sec, color='r', linestyle='--')
            ax.set_xlim(-max_lag_sec, max_lag_sec)
            ax.set_ylim(-1,1)
            ax.set_ylabel('Pearson r'); ax.set_xlabel('Lag (s)')
            p_str = "p<0.001" if (not np.isnan(p_val) and p_val<0.001) else f"p={p_val:.3f}"
            ax.set_title(f'{ref} - {ch} CrossCorr\nlag={lag_sec:.1f}s, r={r_max:.2f}, {p_str} ({lead})')

            # ★ 返却用に1行追加
            rows_xcorr.append({
                'metric':'XCorr_HF_RMS','ref':ref,'ch':ch,
                'r_max':float(r_max) if np.isfinite(r_max) else np.nan,
                'p':float(p_val) if np.isfinite(p_val) else np.nan,
                'lag_sec':float(lag_sec) if np.isfinite(lag_sec) else np.nan,
                'lead':lead,
                'win_sec':win_sec,'step_sec':step_sec
            })


    if show:
        import re, os
        os.makedirs(save_dir, exist_ok=True)

        # ファイル名: subject_id + "XCorr_RMS"
        safe_title = "XCorr_RMS"
        fname = f"{subject_id}_{safe_title}.png"
        out_path = os.path.join(save_dir, fname)

        fig.savefig(out_path, dpi=600, bbox_inches="tight",
                    facecolor="w", edgecolor="w")
        print(f"[Saved PNG] {out_path}")

        plt.show(block=True)
    else:
        plt.close(fig)

    return rows_xcorr




res_corr = plot_rms_crosscorr_4x3_to_pdf(
    eeg_car_dict, eeg_csd_dict, eeg_cz_dict,
    emg_left, emg_right, sfreq,
    show=True, save_dir=save_dir,
    order=('Cz','CAR','CSD'),
    figsize=(11,10),
    width_ratios=(1.0, 1.0, 1.0),# 1列目の箱を少し広く
    height_ratios=(1.0, 1.0, 1.0, 1.0),  # 行ごとに微調整も可能
    box_aspect=0.65,  # 例: 1.0 で各Axesを正方形に固定
)



# === CSV保存（CSV2/Corr フォルダに、症例ごとに分けて保存） ===
df_corr = pd.DataFrame(res_corr)

base_dir = os.path.expanduser("~/Desktop/EMG/Jupyter/plots")
csv2_corr_dir = os.path.join(base_dir, "CSV2", "Corr")
os.makedirs(csv2_corr_dir, exist_ok=True)

csv_path = os.path.join(csv2_corr_dir, f"Corr_{subject_id}.csv")
df_corr.to_csv(csv_path, index=False)

print(f"Corr結果をCSVに保存しました: {csv_path}")
