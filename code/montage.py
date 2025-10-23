# === CAR, CSD, Czモンタージュ ========================================================
import numpy as np
import mne
from mne.preprocessing import compute_current_source_density

def prepare_car_csd(
    raw,
    exclude_chs=('A1','A2','T1','T2','M1','M2'),   # 10-20ではA1/A2, 10-10ではT1/T2/M1/M2など座標未定義が混ざりやすい
    montage='standard_1020',
    verbose=True,
    csd_kwargs=None
):
    """
    raw: mne.io.Raw
    返り値:
      raw_car  : 平均基準（CAR）を適用したRaw（EEGのみ）
      raw_csd  : CSD変換済みRaw（EEGのみ）
      used_chs : CAR/CSDの計算に最終的に使われたチャンネル名リスト
    """
    csd_kwargs = csd_kwargs or dict(lambda2=1e-5, stiffness=4, n_legendre_terms=50)
    
    # EEGだけ取り出し & 名称正規化
    raw_car = raw.copy().pick_types(eeg=True, eog=False, emg=False, meg=False)
    bads = set(getattr(raw, "info", {}).get("bads", []))
    raw_car.drop_channels([ch for ch in raw_car.ch_names if ch in bads])
    name_map = {"T7":"T3", "T8":"T4", "P7":"T5", "P8":"T6"}  # 必要に応じて
    raw_car.rename_channels(lambda n: name_map.get(n.replace('-Ref','').replace('EEG ',''),
                                                   n.replace('-Ref','').replace('EEG ','')))
    raw_car.set_channel_types({ch: 'eeg' for ch in raw_car.ch_names})

    # モンタージュ設定（座標が無いchはignore）
    raw_car.set_montage(montage, on_missing='ignore')

    # 座標が実在するchだけを残す（ゼロ/NaN除去）
    valid = []
    for ch in raw_car.ch_names:
        loc = raw_car.info['chs'][raw_car.ch_names.index(ch)]['loc'][:3]
        if (not np.allclose(loc, 0)) and (not np.isnan(loc).any()):
            valid.append(ch)

    # 除外リストも反映（存在するものだけ）
    valid = [ch for ch in valid if ch not in exclude_chs and ch in raw_car.ch_names]

    if len(valid) < 16:
        # CSDは空間隣接が必要なので、chが少なすぎると不安定
        if verbose:
            print(f"[prepare_car_csd] 警告: 座標有りチャンネルが少ないためCSDが不安定になる可能性があります: {len(valid)} ch")

    # 実際に使うチャンネルに絞る
    raw_car.pick_channels(valid)

    # 平均基準（CAR）
    raw_car.set_eeg_reference(ref_channels='average', projection=False)

    if verbose:
        print("[prepare_car_csd] CARに含まれるチャンネル:", raw_car.ch_names)

        
        
    # CSD（球面スプライン）
    # 十分な隣接関係が無いとエラーになる可能性 → try/except でフォールバック
    try:
        raw_csd = compute_current_source_density(raw_car.copy())
    except Exception as e:
        if verbose:
            print("[prepare_car_csd] CSD計算に失敗:", repr(e))
            print("  → モンタージュ/チャンネルセット/除外ch を見直してください。")
        raw_csd = None

    return raw_car, raw_csd, tuple(raw_car.ch_names)



def _get_uv_from_raw(raw_like, ch_name):
    """Raw系オブジェクトから単一chをμVで返す。存在しなければNone"""
    if (raw_like is None) or (ch_name not in raw_like.ch_names):
        return None
    return raw_like.get_data(picks=[ch_name])[0] * 1e6



def get_car_signal(raw_car, ch_name):
    """CAR後の電位 [μV]"""
    sig = _get_uv_from_raw(raw_car, ch_name)
    if sig is None:
        raise ValueError(f"[get_car_signal] チャンネルがありません: {ch_name}")
    return sig



def get_csd_signal(raw_csd, ch_name, scale=1.0):
    """
    CSDの疑似単位（V/m^2相当）をグラフ用に任意スケーリング。
    ここでは [μV] と同等の見栄えに合わせたい場合 scale を調整。
    デフォルトは 1.0（無変換）。以前の /10000 を再現したいなら scale=1/10000 を指定。
    """
    if raw_csd is None:
        raise ValueError("[get_csd_signal] raw_csd が None（CSD未計算）です。")
    sig = _get_uv_from_raw(raw_csd, ch_name)  # ここでは一度μV換算（見栄え優先）。本来の単位は別。
    if sig is None:
        raise ValueError(f"[get_csd_signal] チャンネルがありません: {ch_name}")
    return sig * float(scale)



def get_cz_bipolar(raw_car, ch_name, cz_name='Cz'):
    """
    CAR空間で Cz との差分（擬似双極: ch - Cz）[μV] を返す。
    Czが無ければ ch のみ（差分なし）を返す。
    """
    sig = get_car_signal(raw_car, ch_name)  # μV
    if cz_name in raw_car.ch_names:
        cz = get_car_signal(raw_car, cz_name)
        # 長さ安全化（万一）
        n = min(len(sig), len(cz))
        return sig[:n] - cz[:n]
    else:
        return sig


    
def make_eeg_dict(raw_obj, target_chs, getter):
    """
    raw_obj と getter（関数）を使って {ch: 波形} の辞書を作る。
    存在しないchはスキップ。空辞書になったら ValueError。
    """
    out = {}
    for ch in target_chs:
        if raw_obj is None:
            break
        if ch in raw_obj.ch_names:
            out[ch] = getter(raw_obj, ch)
    if not out:
        raise ValueError("[make_eeg_dict] 有効なチャンネルが見つかりません。target_chs と raw_obj.ch_names を確認してください。")
    return out


# ---- 使い方例 ----
# 必要なチャンネル（座標のある10-20名で書く）
raw_car, raw_csd, used_chs = prepare_car_csd(
    raw,
    exclude_chs=('A1','A2','T1','T2','M1','M2'),
    montage='standard_1020',
    verbose=True,
    # csd_kwargs=dict(lambda2=1e-5, stiffness=4, n_legendre_terms=50)  # 明示可
)

target_chs = ['Fp1','Fp2','F7','F8','T3','T4','T5','T6','O1','O2']
eeg_car_dict = make_eeg_dict(raw_car, target_chs, get_car_signal)
eeg_csd_dict = make_eeg_dict(raw_csd, target_chs, lambda r, ch: get_csd_signal(r, ch, scale=1/10000))
eeg_cz_dict  = make_eeg_dict(raw_car, target_chs, get_cz_bipolar) if 'Cz' in used_chs else {}


# EMGは既存の emg_left / emg_right をそのまま辞書へ
emg_dict = {'left': emg_left, 'right': emg_right}


sigs_T3 = {"Cz": eeg_cz_dict["T3"], "CAR": eeg_car_dict["T3"], "CSD": eeg_csd_dict["T3"]}
sigs_T4 = {"Cz": eeg_cz_dict["T4"], "CAR": eeg_car_dict["T4"], "CSD": eeg_csd_dict["T4"]}
