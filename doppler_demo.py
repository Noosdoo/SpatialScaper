"""ドップラー最小実装＋動作証明（mono・スタンドアロン）。
原理：音源の各サンプルは「放射時刻 t」に出て「到着時刻 tau = t + r(t)/c」にマイクへ届く。
到着時刻が一様グリッドになるよう補間し直す＝距離変化（視線速度）から自然にピッチ変化が出る。
静止(r一定)なら tau は一様にずれるだけ＝周波数変化なし（=全音声には影響しない実証）。
出力: output_compare/foa/{tone,car}_{doppler,nodoppler}.wav, figs/doppler_proof.png
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


def apply_doppler(x, r_t, sr, c=C):
    """x: 音源(N,), r_t: 各サンプル時刻での音源-マイク距離[m](N,). 戻り: ドップラー反映後の音。"""
    N = len(x)
    t = np.arange(N) / sr
    tau = t + r_t / c                       # 各放射サンプルの到着時刻
    tau_out = np.arange(tau[0], tau[-1], 1.0 / sr)
    return np.interp(tau_out, tau, x).astype("float32")   # 到着時刻を一様化＝ドップラー


def passby_distance(N, sr, d0=5.0, v=30.0):
    """直線通過の距離。y=d0 で x方向に速さv。中央で最接近。"""
    t = np.arange(N) / sr
    x_pos = v * (t - t[-1] / 2.0)            # 中央で x=0（最接近）
    return np.sqrt(x_pos**2 + d0**2)


def main():
    dur, d0, v = 4.0, 5.0, 30.0
    print(f"pass-by: closest {d0} m, speed {v} m/s | Doppler factor approach={C/(C-v):.3f}, recede={C/(C+v):.3f}")

    # --- 1) 純音(1kHz)で証明 ---
    N = int(dur * SR)
    r = passby_distance(N, SR, d0, v)
    tone = np.sin(2*np.pi*1000*np.arange(N)/SR).astype("float32")
    tone_dop = apply_doppler(tone, r, SR)
    sf.write("output_compare/foa/tone_nodoppler.wav", tone, SR)
    sf.write("output_compare/foa/tone_doppler.wav", tone_dop, SR)

    # --- 2) 実音源(233472)に適用 ---
    car, _ = librosa.load("../SELD-Data-Generator/database/FSD50K/FSD50K.dev_audio/233472.wav",
                          sr=SR, mono=True, duration=dur)
    rc = passby_distance(len(car), SR, d0, v)
    sf.write("output_compare/foa/car_nodoppler.wav", car.astype("float32"), SR)
    sf.write("output_compare/foa/car_doppler.wav", apply_doppler(car, rc, SR), SR)

    # --- 3) 証明スペクトログラム（純音：平坦 vs 滑らかに上下）---
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for a, (nm, sig) in zip(ax, [("NO Doppler: flat 1 kHz", tone),
                                 ("WITH Doppler: 1 kHz slides high->low", tone_dop)]):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=2048, hop_length=128)), ref=np.max)
        librosa.display.specshow(S, sr=SR, hop_length=128, x_axis="time", y_axis="hz", ax=a, cmap="magma")
        a.set_ylim(700, 1300); a.set_title(nm)
    fig.suptitle(f"Doppler implementation proof  (1 kHz tone, {v:.0f} m/s pass-by, closest {d0:.0f} m)")
    fig.savefig("output_compare/figs/doppler_proof.png", dpi=120)
    print("saved: output_compare/figs/doppler_proof.png")
    print("wavs: tone_{nodoppler,doppler}.wav, car_{nodoppler,doppler}.wav (in output_compare/foa/)")


if __name__ == "__main__":
    main()
