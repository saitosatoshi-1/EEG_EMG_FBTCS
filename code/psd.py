#=======PSD==============
from scipy.signal import welch
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10        # 図全体の基本サイズ
plt.rcParams["axes.titlesize"] = 12   # タイトル
plt.rcParams["axes.labelsize"] = 11   # 軸ラベル
plt.rcParams["legend.fontsize"] = 9   # 凡例
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

def plot_eeg_emg_psd_loglog_full(
    eeg_car_dict, eeg_csd_dict, eeg_cz_dict,
    emg_left, emg_right, analysis_times, sfreq,
    hf_band=(64, 256), show=False, save_dir=True,
    figsize=(10.5,5), width_ratios=(1,1,1), box_aspect=0.65
):
    ref_data_by_name = {'CAR': eeg_car_dict, 'CSD': eeg_csd_dict, 'Cz': eeg_cz_dict}
    ref_order = ['Cz', 'CAR', 'CSD']
    chs = ['T3', 'T4']
    emg_dict = {'T3': emg_left, 'T4': emg_right}

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=width_ratios, wspace=0.25, hspace=0.3)
    axs = np.array([[fig.add_subplot(gs[0, i]) for i in range(3)],
                    [fig.add_subplot(gs[1, i]) for i in range(3)]])

    # ★ box_aspect を全Axesに適用（縦:横）
    if box_aspect is not None:
        for ax in axs.ravel():
            ax.set_box_aspect(box_aspect)

    # ★ 枠線を細く（例: 0.6pt）
    for ax in axs.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            

    rows_psd = []

    for col, ref_name in enumerate(ref_order):
        ref_data = ref_data_by_name[ref_name]
        for row_ch, ch in enumerate(chs):
            ax = axs[row_ch, col]
            if (ref_data is None) or (ch not in ref_data) or (emg_dict[ch] is None):
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_axis_off()
                rows_psd.append({
                    'metric':'PSD_shape_corr','ref':ref_name,'ch':ch,
                    'r':np.nan,'p':np.nan,
                    'band_low':float(hf_band[0]),'band_high':float(hf_band[1])
                })
                continue

            eeg = ref_data[ch]; emg = emg_dict[ch]
            if len(eeg) < int(2*sfreq) or len(emg) < int(2*sfreq):
                ax.text(0.5, 0.5, "Data too short", ha="center", va="center")
                ax.set_axis_off()
                rows_psd.append({
                    'metric':'PSD_shape_corr','ref':ref_name,'ch':ch,
                    'r':np.nan,'p':np.nan,
                    'band_low':float(hf_band[0]),'band_high':float(hf_band[1])
                })
                continue

            nperseg  = int(0.5 * sfreq); noverlap = nperseg // 2
            f_eeg, Pxx_eeg = welch(eeg, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
            f_emg, Pxx_emg = welch(emg, fs=sfreq, nperseg=nperseg, noverlap=noverlap)

            m_eeg = (f_eeg >= hf_band[0]) & (f_eeg <= hf_band[1])
            m_emg = (f_emg >= hf_band[0]) & (f_emg <= hf_band[1])
            m = min(m_eeg.sum(), m_emg.sum())
            if m >= 3:
                log_eeg = np.log10(Pxx_eeg[m_eeg][:m] + 1e-12)
                log_emg = np.log10(Pxx_emg[m_emg][:m] + 1e-12)
                r, p = pearsonr(log_eeg, log_emg)
            else:
                r, p = np.nan, np.nan

                
                
            # 描画
            ax.loglog(f_eeg, Pxx_eeg)
            ax.loglog(f_emg, Pxx_emg)
            ax.axvspan(hf_band[0], hf_band[1], color='gray', alpha=0.15, lw=0)
            ax.set_xlim(2, 300)
            p_str = ('p value < 0.001' if (not np.isnan(p) and p < 0.001)
                     else ('p value = N/A' if np.isnan(p) else f'p value = {p:.3f}'))
            ax.set_title(f'{ch} ({ref_name}) - r={r:.2f}, {p_str}')
            #ax.legend()

            # 行を追加
            rows_psd.append({
                'metric':'PSD_shape_corr','ref':ref_name,'ch':ch,
                'r':float(r) if np.isfinite(r) else np.nan,
                'p':float(p) if np.isfinite(p) else np.nan,
                'band_low':float(hf_band[0]),'band_high':float(hf_band[1])
            })

            
            
    # 軸ラベル
    for col, ref_name in enumerate(ref_order):
        ylabel = 'PSD (μV²/cm⁴·Hz)' if ref_name == "CSD" else 'PSD (μV²/Hz)'
        axs[0, col].set_ylabel(ylabel); axs[1, col].set_ylabel(ylabel)
    for ax in axs[1, :]:
        ax.set_xlabel('Frequency (Hz)')

    plt.tight_layout()

    

    if show:
        import re, os
        os.makedirs(save_dir, exist_ok=True)

        # ファイル名: subject_id + "PSD_loglog"
        safe_title = "PSD_loglog"
        fname = f"{subject_id}_{safe_title}.png"
        out_path = os.path.join(save_dir, fname)

        fig.savefig(out_path, dpi=600, bbox_inches="tight",
                    facecolor="w", edgecolor="w")
        print(f"[Saved PNG] {out_path}")

        plt.show(block=True)

    plt.close(fig)

    return rows_psd





res = plot_eeg_emg_psd_loglog_full(
    eeg_car_dict, eeg_csd_dict, eeg_cz_dict,
    emg_left, emg_right, analysis_times,
    sfreq=sfreq,
    # start_idx/clonic_end_idx は渡さない → 全区間
    hf_band=(64, 256),
    show=True,
    save_dir=save_dir,
    figsize=(12,6), width_ratios=(1,1,1), box_aspect=0.65
)

print(res)  # {('T3','Cz'):{'r':..., 'p':...}, ...}

print("PSD波形の解析が全て終了しました。")




# === CSV保存（CSV2フォルダに） ===
df_psd = pd.DataFrame(res)

base_dir = os.path.expanduser("~/Desktop/EMG/Jupyter/plots")
csv2_psd_dir = os.path.join(base_dir, "CSV2", "PSD")
os.makedirs(csv2_psd_dir, exist_ok=True)

csv_path = os.path.join(csv2_psd_dir, f"PSD_{subject_id}.csv")
df_psd.to_csv(csv_path, index=False)

print(f"PSD結果をCSVに保存しました: {csv_path}")
