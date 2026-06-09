"""両ツールの“実出力”car FOA から、スライド用の図＋試聴音声を作る。
入力: output_compare/foa/car_demo.wav      (SpatialScaper 実出力)
      output_compare/foa/car_seld_foa.wav  (SELD-Data-Generator mode1 実出力)
出力: output_compare/figs/car_compare_tools.png  (2ツール横並び比較図)
      output_compare/listen/car_seld_stereo.wav / car_seld_mono_omni.wav
"""
import os
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 24000
CHN = ["W", "Y", "Z", "X"]
LISTEN = "output_compare/listen"
FIGDIR = "output_compare/figs"
os.makedirs(LISTEN, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# (見出し, FOAパス, 切り出し[s], 残響の出所)
TOOLS = [
    ("SpatialScaper (measured METU SRIR)", "output_compare/foa/car_demo.wav", (1.0, 27.5), "measured SRIR (METU)"),
    ("SELD-Data-Generator mode1 (simulated room)", "output_compare/foa/car_seld_foa.wav", (0.0, 26.0), "simulated room (pyroomacoustics)"),
]


def load(path, t0=None, t1=None):
    x, sr = sf.read(path)
    if x.ndim == 1:
        x = x[:, None]
    if sr != SR:
        x = librosa.resample(x.T.astype(float), orig_sr=sr, target_sr=SR).T
        sr = SR
    if t0 is not None:
        x = x[int(t0 * sr):int(t1 * sr)]
    return x, sr


def main():
    # --- SELD 側の試聴デコード（SpatialScaper側は既存） ---
    foa, sr = load("output_compare/foa/car_seld_foa.wav")
    W, Y = foa[:, 0], foa[:, 1]
    st = np.stack([W + Y, W - Y], 1); st = st / np.max(np.abs(st))
    sf.write(f"{LISTEN}/car_seld_stereo.wav", st.astype("float32"), sr)
    sf.write(f"{LISTEN}/car_seld_mono_omni.wav", (W / np.max(np.abs(W))).astype("float32"), sr)
    print("listen: car_seld_stereo.wav / car_seld_mono_omni.wav")

    # --- 2ツール横並び比較図 ---
    fig, ax = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
    for j, (title, path, (t0, t1), rev) in enumerate(TOOLS):
        f, sr = load(path, t0, t1)
        t = np.arange(len(f)) / sr
        for c in range(4):
            ax[0, j].plot(t, f[:, c] + c * 0.15, lw=0.4, label=CHN[c])
        ax[0, j].set_title(title, fontsize=11)
        ax[0, j].legend(fontsize=7, loc="upper right"); ax[0, j].set_xlabel("time [s]")
        Wj = f[:, 0]
        S = librosa.amplitude_to_db(np.abs(librosa.stft(Wj, n_fft=1024, hop_length=256)), ref=np.max)
        librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time", y_axis="hz",
                                 ax=ax[1, j], cmap="magma")
        ax[1, j].set_title("FOA W(omni) spectrogram")
        ax[2, j].axis("off")
        ax[2, j].text(0.02, 0.5,
                      f"Source   : 233472.wav (Car passing by)\n"
                      f"DOA      : az=-18 el=-32 r=1.871 m\n"
                      f"Format   : FOA 4ch (W,Y,Z,X), 24 kHz\n"
                      f"Reverb   : {rev}\n"
                      f"= 実ツールの出力", family="monospace", fontsize=11, va="center")
    fig.suptitle("Car FOA: SpatialScaper vs SELD-Data-Generator  (both real tool outputs, same source & DOA)",
                 fontsize=13)
    out = f"{FIGDIR}/car_compare_tools.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
