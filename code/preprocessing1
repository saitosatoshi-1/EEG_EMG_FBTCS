from pathlib import Path
import os, re
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from matplotlib.backends.backend_pdf import PdfPages
from functools import lru_cache
from scipy.signal import butter, sosfiltfilt, hilbert, spectrogram, welch

# --- デバッグ表示（注入確認） ---
print("[PARAMS]", subject_id, data_dir, plots_root, edf_path, save_png, png_dpi, enable_pdf, show_plots)

# --- 受け取ったパス整形 ---
data_dir_path   = Path(os.path.expanduser(data_dir))
plots_root_path = Path(os.path.expanduser(plots_root))

# edf_path が渡されていれば優先、空なら subject_id から組み立て
if edf_path and str(edf_path).strip():
    edf_path = Path(os.path.expanduser(str(edf_path)))
else:
    edf_path = data_dir_path / f"{subject_id}.edf"




# ===== Load EDF =====
if not edf_path.exists():
    raise FileNotFoundError(f"EDFがありません: {edf_path}")

raw = mne.io.read_raw_edf(str(edf_path), preload=True, encoding="latin1")

#edf_path = os.path.expanduser("~/Desktop/EMG/Jupyter/data/HK23 Rt.edf")
#save_root = os.path.expanduser("~/Desktop/EMG/Jupyter/plots/Case")
#os.makedirs(save_root, exist_ok=True)
#subject_id = "HK23 Rt"  # ★ここに症例名をセットする（変数から渡してもOK）
#save_dir = os.path.join(save_root, subject_id)  # 発作名などでサブフォルダ作成可


# 1) チャンネル名正規化（明示マップ推奨）
rename_map = {}
for ch in raw.ch_names:
    new = ch.replace('EEG ', '').replace('-Ref','')
    if new != ch:
        rename_map[ch] = new
if rename_map:
    raw.rename_channels(rename_map)

# === チャンネル一覧の確認 ===
print("=== EDFファイルに含まれるチャンネル ===")
for ch in raw.ch_names:
    print(ch)
    
# 2) EMG型付け（存在確認）
to_emg = {ch: 'emg' for ch in ['POL X5','POL X7'] if ch in raw.ch_names}
if to_emg:
    raw.set_channel_types(to_emg)


sfreq = float(raw.info['sfreq'])
print("サンプリング周波数（Hz）：", sfreq)


# ===== 解析区間 =====
start_sec = 0.0
raw.crop(tmin=start_sec, tmax=None)  # 全ch同期でcrop
times = raw.times
analysis_times = times

# subject_id から強直頭部偏向の秒数を取得（無ければ全長を使う）
try:
    head_tonic_duration_sec = parse_duration_from_subject_id(subject_id)
except ValueError:
    head_tonic_duration_sec = times[-1] if len(times) else 0.0
    print(f"[WARN] subject_id から秒数が取得できませんでした。記録全長 {head_tonic_duration_sec:.2f}s を使用します。")

head_tonic_end_idx = int(head_tonic_duration_sec * sfreq)
head_tonic_end_idx = max(1, min(head_tonic_end_idx, len(times)))  # 範囲ガード
head_tonic_times = times[:head_tonic_end_idx]

print(f"head_tonic (s): {head_tonic_duration_sec:.3f}")
print(f"head_tonic_end_idx: {head_tonic_end_idx} / {len(times)}")




# ========== フィルタ設定 ==========
# picks を一度だけ取得
picks_eeg_emg = mne.pick_types(raw.info, eeg=True, emg=True, exclude=[])

# 先にノッチ（高調波込み）、次に高域フィルタ（ここでは2Hz HPのみ）
# notch は必要な倍音をまとめて指定（50Hz系）
notch_freqs = 50
raw.notch_filter(freqs=notch_freqs, picks=picks_eeg_emg)
raw.filter(l_freq=2.0, h_freq=None, picks=picks_eeg_emg, fir_design='firwin')




# ========== データ取得（必要chのみµV化） ==========
def pick_uv_multi(raw, ch_names):
    """指定したチャンネルをまとめて µV 化して dict で返す"""
    out = {}
    for name in ch_names:
        if name in raw.ch_names:
            out[name] = (raw.get_data(picks=[name])[0] * 1e6).astype(np.float32)
    return out

# EMG
emg_dict = pick_uv_multi(raw, ["POL X5", "POL X7"])

# EEG（解析対象の10電極）
eeg_targets = ["Fp1","Fp2","F7","F8","T3","T4","T5","T6","O1","O2"]
eeg_dict = pick_uv_multi(raw, eeg_targets)

print("EMG available:", list(emg_dict.keys()))
print("EEG available:", list(eeg_dict.keys()))


# === EMG左右割り当て ===
# ここでは X5 を左、X7 を右として扱う（装着系で逆なら入れ替えてください）
emg_left  = emg_dict.get("POL X5", None)
emg_right = emg_dict.get("POL X7", None)

# 念のためのチェック
if emg_left is None or emg_right is None:
    raise ValueError(f"必要なEMGチャンネルが見つかりません。emg_dict keys={list(emg_dict.keys())}")

# numpy配列に整形（型はfloat32で統一）
emg_left  = np.asarray(emg_left,  dtype=np.float32)
emg_right = np.asarray(emg_right, dtype=np.float32)

# 長さが一致しない場合は短い方に合わせる
min_len = min(len(emg_left), len(emg_right))
emg_left  = emg_left[:min_len]
emg_right = emg_right[:min_len]




# 参考：EEGチャンネル一覧
eeg_chs = mne.pick_types(raw.info, eeg=True, exclude=[])
eeg_ch_names = [raw.ch_names[i] for i in eeg_chs]
print("EEG channels (count):", len(eeg_ch_names))


