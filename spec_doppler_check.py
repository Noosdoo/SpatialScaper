"""ステップ3：ドップラー判定。静止(car_demo) vs 移動(car_moving) の W-ch スペクトログラムを並べ、
移動側で周波数が「高→低」へ滑らかに遷移する曲線（ドップラー）が出るかを見る。
"""
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 24000
FILES = [
    ("STATIC  (car_demo.wav)", "output_compare/foa/car_demo.wav"),
    ("MOVING  (car_moving.wav)", "output_compare/foa/car_moving.wav"),
]
FMAX = 4000  # エンジン倍音が見える低域に拡大


def main():
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    for ax, (name, path) in zip(axes, FILES):
        x, sr = sf.read(path)
        W = x[:, 0] if x.ndim > 1 else x
        S = librosa.amplitude_to_db(
            np.abs(librosa.stft(W, n_fft=2048, hop_length=256)), ref=np.max)
        librosa.display.specshow(S, sr=SR, hop_length=256, x_axis="time",
                                 y_axis="hz", ax=ax, cmap="magma", vmin=-80, vmax=0)
        ax.set_ylim(0, FMAX)
        ax.set_title(f"{name}   W-ch spectrogram (0-{FMAX/1000:.0f} kHz)")
    fig.suptitle("Doppler check: STATIC vs MOVING  (same source 233472)\n"
                 "ドップラーがあれば移動側で周波数が時間と共に滑らかに上下する筈", fontsize=12)
    out = "output_compare/figs/doppler_check.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
