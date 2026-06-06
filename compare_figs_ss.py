"""SpatialScaper 版 compare 図（SELD-Data-Generator の compare_figs1.py と同形式）。
 (A) mixture0 で実際に使われたクリップ一覧（クラス/ファイル/時刻/DOA）を表示
 (B) 数クラス分、ORIGINAL(dry mono) vs SYNTHESIZED(4ch FOA) の波形+スペクトログラム図

SpatialScaper はイベント情報を保存しないので、generate_compare.py と同一シード・同一手順で
mixture0 を再現して ssc.fg_events を取得（mixture0 は最初に生成されるので保存済み mix000 と一致）。
"""
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spatialscaper as ss
import spatialscaper.core as ss_core

# ---- generate_compare.py と完全一致させる条件 ----
CLASSES = ["Bird", "BirdVocalization", "Car", "Gunshot", "Rain",
           "Subway", "Thunder", "Train", "Walk"]
ss_core.__DCASE_SOUND_EVENT_CLASSES__ = {c: i for i, c in enumerate(CLASSES)}

FOREGROUND_DIR = "datasets/sound_event_datasets/seld_match"
RIR_DIR = "datasets/rir_datasets"
ROOM, FORMAT = "metu", "foa"
DURATION, SR = 60.0, 24000
MAX_OVERLAP, REF_DB = 1, -65
SEED, N_EVENTS = 2024, 6
OUTPUT_DIR = "output_compare"
FIG_DIR = Path(OUTPUT_DIR) / "figs"
N_COMPARE = 4
CHN = ["W", "Y", "Z", "X"]


def doa_from_xyz(positions):
    """event_position(=xyz の list) から代表DOA(az,el)と移動フラグを返す。"""
    p = np.asarray(positions, dtype=float).reshape(-1, 3)
    moving = len(p) > 1
    x, y, z = p.mean(axis=0)
    az = int(round(np.degrees(np.arctan2(y, x))))
    el = int(round(np.degrees(np.arctan2(z, np.hypot(x, y)))))
    return az, el, moving


def rebuild_mixture0():
    """mixture0 を再現して ssc を返す（保存済み mix000 と同一内容を再生成）。"""
    np.random.seed(SEED)
    random.seed(SEED)
    ssc = ss.Scaper(duration=DURATION, foreground_dir=FOREGROUND_DIR, rir_dir=RIR_DIR,
                    fmt=FORMAT, room=ROOM, use_room_ambient_noise=False,
                    max_event_overlap=MAX_OVERLAP, sr=SR, speed_limit=2.0)
    ssc.ref_db = REF_DB
    ssc.add_background()
    for _ in range(N_EVENTS):
        ssc.add_event()
    audiofile = os.path.join(OUTPUT_DIR, FORMAT, "fold0_room0_mix000")
    labelfile = os.path.join(OUTPUT_DIR, "labels", "fold0_room0_mix000")
    ssc.generate(audiofile, labelfile)  # mix000 を再生成（同一）
    return ssc


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ssc = rebuild_mixture0()
    events = ssc.fg_events  # generate() 後は onset 昇順
    foa, _ = sf.read(os.path.join(OUTPUT_DIR, FORMAT, "fold0_room0_mix000.wav"))

    # ---------- (A) 使用クリップ一覧 ----------
    print("=" * 78)
    print("USED CLIPS (SpatialScaper mixture0, room=metu/実測SRIR)")
    print("=" * 78)
    for i, e in enumerate(events):
        az, el, moving = doa_from_xyz(e.event_position)
        st, du = float(e.event_time), float(e.event_duration)
        tag = "moving" if moving else "static"
        print(f"  [{e.label:<16}] {Path(e.source_file).name:<14} "
              f"t={st:5.1f}-{st+du:5.1f}s  az={az:>4} el={el:>4}  ({tag})")

    # ---------- (B) ORIGINAL vs SYNTHESIZED 図 ----------
    # Car を優先（test_car.m と同じ例）→ 残りを異クラスで埋める
    chosen, seen = [], set()
    order = sorted(range(len(events)), key=lambda i: (events[i].label != "Car", i))
    for i in order:
        if events[i].label in seen:
            continue
        seen.add(events[i].label)
        chosen.append(i)
        if len(chosen) >= N_COMPARE:
            break
    chosen.sort()

    print("\n" + "=" * 78)
    print(f"Comparison figures for {len(chosen)} distinct classes (mixture0):")
    for i in chosen:
        e = events[i]
        az, el, moving = doa_from_xyz(e.event_position)
        st, du = float(e.event_time), float(e.event_duration)
        # ORIGINAL (dry mono)
        orig, _ = librosa.load(e.source_file, sr=SR, mono=True,
                               offset=float(e.source_time), duration=du)
        # SYNTHESIZED FOA slice
        s0 = int(round(st * SR))
        s1 = min(int(round((st + du) * SR)), foa.shape[0])
        seg = foa[s0:s1, :]
        w = seg[:, 0]

        fig, ax = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
        mv = " (moving)" if moving else ""
        fig.suptitle(f"{e.label}    DOA az={az}deg el={el}deg{mv}    "
                     f"(METU measured SRIR, t={st:.1f}-{st+du:.1f}s)", fontsize=13)
        to = np.arange(len(orig)) / SR
        ax[0, 0].plot(to, orig, lw=0.5, color="tab:blue")
        ax[0, 0].set_title("ORIGINAL source (dry, mono) - waveform"); ax[0, 0].set_xlabel("s")
        So = librosa.amplitude_to_db(np.abs(librosa.stft(orig, n_fft=1024, hop_length=256)), ref=np.max)
        librosa.display.specshow(So, sr=SR, hop_length=256, x_axis="time", y_axis="hz",
                                 ax=ax[0, 1], cmap="magma"); ax[0, 1].set_title("ORIGINAL - spectrogram")
        tw = np.arange(len(w)) / SR
        ax[1, 0].plot(tw, w, lw=0.5, color="tab:red")
        ax[1, 0].set_title("SYNTHESIZED FOA W(omni) - waveform"); ax[1, 0].set_xlabel("s")
        if len(w) > 16:
            Sw = librosa.amplitude_to_db(np.abs(librosa.stft(w, n_fft=1024, hop_length=256)), ref=np.max)
            librosa.display.specshow(Sw, sr=SR, hop_length=256, x_axis="time", y_axis="hz",
                                     ax=ax[1, 1], cmap="magma")
        ax[1, 1].set_title("SYNTHESIZED FOA W - spectrogram")
        for c in range(min(4, seg.shape[1])):
            ax[2, 0].plot(tw, seg[:, c] + c * 0.15, lw=0.4, label=CHN[c])
        ax[2, 0].set_title("SYNTHESIZED FOA 4ch (W,Y,Z,X offset)")
        ax[2, 0].legend(fontsize=8, loc="upper right"); ax[2, 0].set_xlabel("s")
        ax[2, 1].axis("off")
        ax[2, 1].text(0.02, 0.5,
                      f"Source clip : {Path(e.source_file).name}\nClass        : {e.label}\n"
                      f"DOA azimuth  : {az} deg\nDOA elevation: {el} deg\n"
                      f"Duration     : {du:.2f} s\nTrajectory   : {'moving' if moving else 'static'}\n\n"
                      f"FOA = W,Y,Z,X (ACN/SN3D).\nReverb: METU em32 measured SRIR.",
                      fontsize=11, family="monospace", va="center")
        safe = e.label
        png = FIG_DIR / f"compare_{i:02d}_{safe}.png"
        fig.savefig(png, dpi=120); plt.close(fig)
        print(f"  saved: {png}   [{e.label}]")
    print("DONE")


if __name__ == "__main__":
    main()
