"""
gen_doppler_air_foa.py — 新doppler: ドップラー + 1/r + 大気吸収(ISO 9613-1) の屋外FOA生成。
前作 gen_doppler_foa.py（ドップラー+1/rのみ）と違い、大気吸収を「距離依存の時変STFTフィルタ」で
追加し、遠ざかるほど高域が落ちる（こもる）を再現する。
A/B = 大気吸収OFF(noair, =前作相当) vs ON(air)。どちらもドップラー+1/rは入っている。

出力（CWD=SpatialScaper で実行）:
  foa/car_doppler_{noair,air}_foa.wav   (4ch, 24kHz, ACN/SN3D)
  listen/car_doppler_{noair,air}_foa_stereo.wav
  labels/car_doppler_{noair,air}_foa.csv  (DCASE 10fps: frame,class,track,az,el,r)
"""
import numpy as np
import soundfile as sf
import librosa

SR = 24000
C = 343.0
R_MIN = 0.5
CLASS_ID = 2
CLIP = "../SELD-Data-Generator/database/FSD50K/FSD50K.dev_audio/233472.wav"

DUR, D0, V, HEIGHT = 4.0, 5.0, 30.0, -2.0   # 4s, 最接近5m, 30m/s, 耳より2m下
T_C, RH = 20.0, 50.0                         # 気温[℃]/湿度[%]（大気吸収の条件）


def passby_traj(t, d0=D0, v=V, height=HEIGHT):
    x = v * (t - t[-1] / 2.0)
    y = np.full_like(t, d0)
    z = np.full_like(t, height)
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.hypot(x, y))
    return r, az, el


def alpha_iso9613(f, Tc=T_C, RH=RH, pa=101.325):
    """ISO 9613-1 大気吸収係数 α(f) [dB/m]。f は numpy 配列可（コードで検証済みの式）。"""
    T = Tc + 273.15
    T0 = 293.15
    pr = 101.325
    paR = pa / pr
    psat = pr * 10**(-6.8346 * (273.16 / T)**1.261 + 4.6151)
    h = RH * (psat / pr) / paR
    frO = paR * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = paR * (T / T0)**(-0.5) * (9 + 280 * h * np.exp(-4.170 * ((T / T0)**(-1/3) - 1)))
    return 8.686 * f * f * (
        1.84e-11 * (1 / paR) * (T / T0)**0.5
        + (T / T0)**(-2.5) * (
            0.01275 * np.exp(-2239.1 / T) / (frO + f * f / frO)
          + 0.1068 * np.exp(-3352.0 / T) / (frN + f * f / frN)))


def apply_air_absorption(s, r_a, nfft=1024, hop=256):
    """距離依存の大気吸収を時変STFTで適用。各フレーム中心の距離で H(f)=10^(-α(f)·r/20)。"""
    f = np.fft.rfftfreq(nfft, 1 / SR)
    a = alpha_iso9613(f)                       # [dB/m] per freq-bin
    win = np.hanning(nfft)
    out = np.zeros(len(s) + nfft)
    nrm = np.zeros(len(s) + nfft)
    for st in range(0, max(1, len(s) - nfft), hop):
        c = min(st + nfft // 2, len(r_a) - 1)
        H = 10**(-a * r_a[c] / 20.0)           # フレーム距離での周波数別減衰
        fr = np.fft.irfft(np.fft.rfft(s[st:st + nfft] * win) * H, n=nfft)
        out[st:st + nfft] += fr * win
        nrm[st:st + nfft] += win**2
    nrm[nrm < 1e-8] = 1.0
    return (out / nrm)[:len(s)]


def render(sig, air=False):
    N = len(sig)
    t = np.arange(N) / SR
    r, az, el = passby_traj(t)
    tau = t + r / C                            # 到着時刻（ドップラー）
    to = np.arange(tau[0], tau[-1], 1 / SR)
    s = np.interp(to, tau, sig)
    az_a = np.interp(to, tau, az)
    el_a = np.interp(to, tau, el)
    r_a = np.interp(to, tau, r)
    if air:
        s = apply_air_absorption(s, r_a)       # ★大気吸収（時変・距離依存）
    amp = s / np.maximum(r_a, R_MIN)           # 1/r 距離減衰
    foa = np.stack([amp,
                    amp * np.sin(az_a) * np.cos(el_a),
                    amp * np.sin(el_a),
                    amp * np.cos(az_a) * np.cos(el_a)], axis=1)
    return foa, az_a, el_a, r_a


def write_label(path, az_a, el_a, r_a, n, fps=10):
    nfr = int(n / SR * fps)
    idx = np.linspace(0, len(az_a) - 1, nfr)
    azd = np.degrees(np.interp(idx, np.arange(len(az_a)), az_a))
    eld = np.degrees(np.interp(idx, np.arange(len(el_a)), el_a))
    rd = np.interp(idx, np.arange(len(r_a)), r_a)
    with open(path, "w") as fp:
        for k in range(nfr):
            fp.write(f"{k},{CLASS_ID},0,{int(round(azd[k]))},{int(round(eld[k]))},{rd[k]:.3f}\n")


def hf_energy(W, lo=4000, hi=12000):
    """高域(lo-hi Hz)のエネルギー（検証用）。"""
    F = np.fft.rfftfreq(len(W), 1 / SR)
    P = np.abs(np.fft.rfft(W * np.hanning(len(W))))**2
    return P[(F >= lo) & (F < hi)].sum()


def main():
    print(f"=== 大気吸収 ISO 9613-1 @ {T_C}C {RH}%RH ===")
    for fhz in [1000, 2000, 4000, 8000]:
        a = alpha_iso9613(np.array([float(fhz)]))[0]
        print(f"  {fhz:>5}Hz: alpha={a:.4f} dB/m  -> 5m:{a*5:.2f}dB  60m:{a*60:.1f}dB")

    car = librosa.load(CLIP, sr=SR, mono=True, duration=DUR)[0]
    fo_off, az0, el0, r0 = render(car, air=False)
    fo_on,  az1, el1, r1 = render(car, air=True)
    sc = max(np.max(np.abs(fo_off)), np.max(np.abs(fo_on))) + 1e-12
    for tag, fo, az_a, el_a, r_a in [("noair", fo_off, az0, el0, r0),
                                     ("air",   fo_on,  az1, el1, r1)]:
        fo = (fo / sc).astype(np.float32)
        sf.write(f"output_compare/foa/car_doppler_{tag}_foa.wav", fo, SR)
        st = np.stack([fo[:, 0] + fo[:, 1], fo[:, 0] - fo[:, 1]], axis=1)
        st = (st / (np.max(np.abs(st)) + 1e-12)).astype(np.float32)
        sf.write(f"output_compare/listen/car_doppler_{tag}_foa_stereo.wav", st, SR)
        write_label(f"output_compare/labels/car_doppler_{tag}_foa.csv", az_a, el_a, r_a, len(fo))

    # 検証: 遠ざかる後半(far)で air は高域が落ちているはず（near では差が小さい）
    n = fo_off.shape[0]
    near = slice(0, SR // 2)               # 最初0.5s（接近・近距離）
    far = slice(n - SR // 2, n)            # 最後0.5s（遠ざかり・遠距離）
    for nm, sl in [("near(~5-8m)", near), ("far(~55-60m)", far)]:
        ratio = hf_energy(fo_on[sl, 0]) / (hf_energy(fo_off[sl, 0]) + 1e-20)
        print(f"  HF(4-12k) air/noair @{nm}: {ratio:.3f}  ({10*np.log10(ratio+1e-20):+.1f} dB)")
    print("done: car_doppler_{noair,air}_foa.wav (+ stereo, labels)")


if __name__ == "__main__":
    main()
