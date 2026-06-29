"""fig_car.py の siren 版（音源と出力名だけ差し替え＝コードはほぼ同一）。
緊急車両サイレン(合成 siren.wav) 1音源 → FOA を SpatialScaper(実測SRIR) で作る。
car_demo と同じ DOA・同じ手順で生成し、car と全く同じに比較できるようにする。
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

CLASSES = ["Bird", "BirdVocalization", "Car", "Gunshot", "Rain",
           "Subway", "Thunder", "Train", "Walk"]
ss_core.__DCASE_SOUND_EVENT_CLASSES__ = {c: i for i, c in enumerate(CLASSES)}

FOREGROUND_DIR = "datasets/sound_event_datasets/seld_match"
RIR_DIR = "datasets/rir_datasets"
SR = 24000
SIREN_CLIP = "../SELD-Data-Generator/database/FSD50K/FSD50K.dev_audio/siren.wav"  # 緊急車両サイレン(合成, ピーポー)
OUT_TRACK = "output_compare/foa/siren_demo"
OUT_LABEL = "output_compare/labels/siren_demo"
FIG = Path("output_compare/figs/compare_siren_demo.png")
CHN = ["W", "Y", "Z", "X"]


def label_doa(label_csv):
    labels = np.loadtxt(label_csv, delimiter=",")
    labels = np.atleast_2d(labels)
    doa_cols = labels[:, 3:6]
    values, counts = np.unique(doa_cols, axis=0, return_counts=True)
    az, el, radius = values[np.argmax(counts)]
    return int(round(az)), int(round(el)), float(radius), labels


def doa(positions):
    p = np.asarray(positions, float).reshape(-1, 3)
    x, y, z = p.mean(axis=0)
    return (int(round(np.degrees(np.arctan2(y, x)))),
            int(round(np.degrees(np.arctan2(z, np.hypot(x, y))))),
            len(p) > 1)


def main():
    random.seed(7); np.random.seed(7)
    ssc = ss.Scaper(duration=30.0, foreground_dir=FOREGROUND_DIR, rir_dir=RIR_DIR,
                    fmt="foa", room="metu", use_room_ambient_noise=False,
                    max_event_overlap=1, sr=SR, max_event_dur=30.0, speed_limit=2.0)
    ssc.ref_db = -65
    # サイレンを1つ強制配置（静止、t=1s 開始、クリップ全長を使う）。class は car と同じ枠を流用(id=2)
    ssc.add_event(label=("const", "Car"), source_file=("choose", [SIREN_CLIP]),
                  event_time=("const", 1.0), event_position=("static", None))
    ssc.generate(OUT_TRACK, OUT_LABEL)

    e = ssc.fg_events[0]
    raw_az, raw_el, moving = doa(e.event_position)
    az, el, radius, labels = label_doa(OUT_LABEL + ".csv")
    st, du = float(e.event_time), float(e.event_duration)
    foa, _ = sf.read(OUT_TRACK + ".wav")
    s0 = int(round(st * SR)); s1 = min(int(round((st + du) * SR)), foa.shape[0])
    seg = foa[s0:s1, :]; w = seg[:, 0]
    orig, _ = librosa.load(e.source_file, sr=SR, mono=True,
                           offset=float(e.source_time), duration=du)

    print(f"Siren event: {Path(e.source_file).name}  t={st:.1f}-{st+du:.1f}s  "
          f"label_az={az} label_el={el} r={radius:.3f}m  "
          f"raw_position_az={raw_az} raw_position_el={raw_el}  "
          f"({'moving' if moving else 'static'})")

    fig, ax = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(f"Siren (SpatialScaper, METU measured SRIR)   "
                 f"CSV label DOA az={az}deg el={el}deg r={radius:.3f}m   "
                 f"t={st:.1f}-{st+du:.1f}s", fontsize=13)
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
                  f"Source clip : {Path(e.source_file).name}\nClass        : Siren (synthetic ambulance)\n"
                  f"CSV class id : 2 (car枠を流用)\n"
                  f"CSV frames   : {int(labels[:,0].min())}-{int(labels[:,0].max())}"
                  f" ({len(labels)} frames @10Hz)\n"
                  f"CSV azimuth  : {az} deg\nCSV elevation: {el} deg\n"
                  f"CSV distance : {radius:.3f} m\n"
                  f"Event time   : {st:.1f}-{st+du:.1f} s\nDuration     : {du:.2f} s\n"
                  f"Trajectory   : {'moving' if moving else 'static'}\n"
                  f"Generator    : SpatialScaper, room=metu, fmt=foa, sr=24000\n"
                  f"Background   : white noise only, ambient room noise disabled\n\n"
                  f"FOA = W,Y,Z,X (ACN/SN3D).\nReverb: METU em32 measured SRIR.\n"
                  f"siren.wav = ピーポー 960/770Hz (合成).",
                  fontsize=11, family="monospace", va="center")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120); plt.close(fig)
    print(f"saved: {FIG}")


if __name__ == "__main__":
    main()
