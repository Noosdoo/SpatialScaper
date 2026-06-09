"""FOA・移動軌跡＋ドップラーの最小パイプライン（自由音場・屋外モデルのプロト）。
mono実証(doppler_demo.py)を 4ch FOA・実通過軌跡に拡張。
処理：軌跡 position(t) -> 距離r(t)/方位az(t)/仰角el(t)
      -> 到着時刻 tau=t+r/c で全量リサンプル（=ドップラー）
      -> 距離減衰(1/r) + 時変の自由音場FOAエンコード(ACN/SN3D)
出力：output_compare/foa/car_doppler_foa.wav(4ch) + labels/car_doppler_foa.csv
      + listen/car_doppler_foa_stereo.wav + figs/doppler_foa_check.png
※屋内SRIR(metu±1.5m)は速い通過を表現できないので自由音場で実証。SpatialScaperはシーン/軌跡/FOA形式の土台。
"""
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 24000
C = 343.0
CLIP = "../SELD-Data-Generator/database/FSD50K/FSD50K.dev_audio/233472.wav"


def main():
    dur, d0, v, height = 4.0, 5.0, 30.0, -2.0     # 最接近5m, 30m/s, 車は耳より下(z=-2)
    car, _ = librosa.load(CLIP, sr=SR, mono=True, duration=dur)
    N = len(car); t = np.arange(N) / SR

    # 通過軌跡：y=d0, z=height, x方向に速さv（中央で最接近）
    x = v * (t - t[-1] / 2.0)
    y = np.full_like(t, d0)
    z = np.full_like(t, height)
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)                          # rad (ACN: az=atan2(y,x))
    el = np.arctan2(z, np.hypot(x, y))

    # 到着時刻でリサンプル＝ドップラー（信号も方向も距離も到着グリッドへ）
    tau = t + r / C
    to = np.arange(tau[0], tau[-1], 1.0 / SR)
    s   = np.interp(to, tau, car)
    az_a = np.interp(to, tau, az)
    el_a = np.interp(to, tau, el)
    r_a  = np.interp(to, tau, r)

    # 距離減衰(1/r) + 自由音場FOAエンコード（方向が時変）
    amp = s / np.maximum(r_a, 0.5)
    W = amp
    Y = amp * np.sin(az_a) * np.cos(el_a)
    Z = amp * np.sin(el_a)
    X = amp * np.cos(az_a) * np.cos(el_a)
    foa = np.stack([W, Y, Z, X], axis=1)
    foa = foa / np.max(np.abs(foa))
    sf.write("output_compare/foa/car_doppler_foa.wav", foa.astype("float32"), SR)

    # 試聴ステレオ
    st = np.stack([foa[:, 0] + foa[:, 1], foa[:, 0] - foa[:, 1]], axis=1); st /= np.max(np.abs(st))
    sf.write("output_compare/listen/car_doppler_foa_stereo.wav", st.astype("float32"), SR)

    # DCASEラベル(10fps): frame,class,track,az,el,r
    nfr = int(len(foa) / SR * 10)
    azd = np.degrees(np.interp(np.linspace(to[0], to[-1], nfr), to, az_a))
    eld = np.degrees(np.interp(np.linspace(to[0], to[-1], nfr), to, el_a))
    rd  = np.interp(np.linspace(to[0], to[-1], nfr), to, r_a)
    with open("output_compare/labels/car_doppler_foa.csv", "w") as f:
        for k in range(nfr):
            f.write(f"{k},2,0,{int(round(azd[k]))},{int(round(eld[k]))},{rd[k]:.3f}\n")

    print(f"saved car_doppler_foa.wav {foa.shape}")
    print(f"az: {np.degrees(az_a[0]):.0f}->{np.degrees(az_a[-1]):.0f} deg, "
          f"el closest {np.degrees(el_a).min():.0f}, r: {r_a.min():.1f}->{r_a.max():.1f} m")

    # W-ch スペクトログラム（FOAでもドップラーが乗るか）
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(W, n_fft=2048, hop_length=128)), ref=np.max)
    librosa.display.specshow(S, sr=SR, hop_length=128, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    ax.set_ylim(0, 4000)
    ax.set_title("car_doppler_foa  W-ch spectrogram (free-field, 30 m/s pass-by + Doppler)")
    fig.savefig("output_compare/figs/doppler_foa_check.png", dpi=120)
    print("saved: output_compare/figs/doppler_foa_check.png")


if __name__ == "__main__":
    main()
