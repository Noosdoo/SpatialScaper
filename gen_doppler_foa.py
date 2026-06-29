"""
ドップラー比較生成（自由音場・移動音源・FOA）— 軸4の「ドップラー on/off」モジュール試作。
同一の通過軌跡で ドップラーOFF（移動＋距離減衰のみ）と ドップラーON を生成し、
MATLAB(car_doppler_foa_spec.m) で W-ch スペクトログラムを上下比較できるようにする。

物理（コードで検証済み）:
  音速 c=343 m/s、到着時刻リサンプル tau=t+r(t)/c（視線速度→ピッチ変化）、
  距離減衰 1/r（r は R_MIN=0.5 m でクランプ）、自由音場 FOA(ACN/SN3D, 時変方向)。
  ※大気吸収は今回は未適用（次段の別モジュール）。

出力（CWD=SpatialScaper で実行）:
  foa/{tone,car}_nodoppler_foa.wav, foa/{tone,car}_doppler_foa.wav  (4ch, 24kHz, ACN/SN3D)
  listen/{tone,car}_{nodoppler,doppler}_foa_stereo.wav             (試聴ステレオ)
  labels/{tone,car}_{nodoppler,doppler}_foa.csv  (DCASE 10fps: frame,class,track,az,el,r)
"""
import numpy as np
import soundfile as sf
import librosa

SR = 24000
C = 343.0           # 音速[m/s]（20℃相当）
R_MIN = 0.5         # 1/r のクランプ（r->0 の発散防止）
CLASS_ID = 2        # DCASEクラス(車)。tone も便宜上同じIDで出力
CLIP  = "../SELD-Data-Generator/database/FSD50K/FSD50K.dev_audio/233472.wav"  # car
SIREN = "../SELD-Data-Generator/database/FSD50K/FSD50K.dev_audio/siren.wav"   # 緊急車両サイレン(合成)

# 通過シナリオ（必要ならここを変える）
D0, V, HEIGHT = 5.0, 8.0, -2.0   # 最接近5m, 8m/s(≒30km/h, 住宅街想定), 耳より2m下。長さは元音源に合わせる


def passby_traj(t, d0=D0, v=V, height=HEIGHT):
    """直線通過: y=d0一定, z=height一定, x=v*(t-中央)。中央で最接近。
    戻り: r(距離), az=atan2(y,x), el=atan2(z,hypot(x,y))  [rad]"""
    x = v * (t - t[-1] / 2.0)
    y = np.full_like(t, d0)
    z = np.full_like(t, height)
    r = np.sqrt(x**2 + y**2 + z**2)
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.hypot(x, y))
    return r, az, el


def render_foa(sig, doppler=True, dist_atten=True):
    """移動音源を FOA(W,Y,Z,X / ACN-SN3D) に。doppler=False はピッチ不変(=移動・減衰のみ)。
    戻り: foa[N,4], az_a, el_a, r_a（ラベル用に到着グリッドの方向・距離）"""
    N = len(sig)
    t = np.arange(N) / SR
    r, az, el = passby_traj(t)
    if doppler:
        tau = t + r / C                       # 各放射サンプルの到着時刻
        to = np.arange(tau[0], tau[-1], 1.0 / SR)
        s    = np.interp(to, tau, sig)        # 到着時刻を一様化＝ドップラー
        az_a = np.interp(to, tau, az)
        el_a = np.interp(to, tau, el)
        r_a  = np.interp(to, tau, r)
    else:
        s, az_a, el_a, r_a = sig, az, el, r   # 到着時刻リサンプルなし＝ドップラーOFF
    amp = s / np.maximum(r_a, R_MIN) if dist_atten else s
    foa = np.stack([amp,                                  # W
                    amp * np.sin(az_a) * np.cos(el_a),    # Y
                    amp * np.sin(el_a),                   # Z
                    amp * np.cos(az_a) * np.cos(el_a)],   # X
                   axis=1)
    return foa, az_a, el_a, r_a


def write_label(path, az_a, el_a, r_a, n_samples, fps=10):
    """到着グリッドの方向・距離を 10fps の DCASE ラベルに落とす。"""
    nfr = int(n_samples / SR * fps)
    idx = np.linspace(0, len(az_a) - 1, nfr)
    azd = np.degrees(np.interp(idx, np.arange(len(az_a)), az_a))
    eld = np.degrees(np.interp(idx, np.arange(len(el_a)), el_a))
    rd  = np.interp(idx, np.arange(len(r_a)), r_a)
    with open(path, "w") as f:
        for k in range(nfr):
            f.write(f"{k},{CLASS_ID},0,{int(round(azd[k]))},{int(round(eld[k]))},{rd[k]:.3f}\n")


def main():
    car   = librosa.load(CLIP,  sr=SR, mono=True)[0]   # 元音源フル長（doppler出力を音源長に合わせる）
    siren = librosa.load(SIREN, sr=SR, mono=True)[0]
    sources = {
        "tone":  np.sin(2 * np.pi * 1000 * np.arange(len(car)) / SR).astype(np.float32),
        "car":   car,
        "siren": siren,
    }
    print(f"pass-by: closest {D0} m, speed {V} m/s ({V*3.6:.0f} km/h), height {HEIGHT} m, "
          f"len {len(car)/SR:.1f}s (元音源に一致)")
    print(f"Doppler factor: approach x{C/(C-V):.3f} (+{(C/(C-V)-1)*100:.1f}%), "
          f"recede x{C/(C+V):.3f} ({(C/(C+V)-1)*100:.1f}%)")
    for name, sig in sources.items():
        foa_off, az0, el0, r0 = render_foa(sig, doppler=False)
        foa_on,  az1, el1, r1 = render_foa(sig, doppler=True)
        scale = max(np.max(np.abs(foa_off)), np.max(np.abs(foa_on))) + 1e-12  # OFF/ON 共通スケール
        for tag, foa, az_a, el_a, r_a in [("nodoppler", foa_off, az0, el0, r0),
                                          ("doppler",   foa_on,  az1, el1, r1)]:
            foa = (foa / scale).astype(np.float32)
            sf.write(f"output_compare/foa/{name}_{tag}_foa.wav", foa, SR)
            st = np.stack([foa[:, 0] + foa[:, 1], foa[:, 0] - foa[:, 1]], axis=1)
            st = (st / (np.max(np.abs(st)) + 1e-12)).astype(np.float32)
            sf.write(f"output_compare/listen/{name}_{tag}_foa_stereo.wav", st, SR)
            write_label(f"output_compare/labels/{name}_{tag}_foa.csv", az_a, el_a, r_a, len(foa))
        print(f"  {name}: OFF {foa_off.shape[0]} smp / ON {foa_on.shape[0]} smp  "
              f"az {np.degrees(az1[0]):.0f}->{np.degrees(az1[-1]):.0f}deg  "
              f"r {r1.min():.1f}->{r1.max():.1f}m  el_min {np.degrees(el1).min():.0f}deg")
    print("done: foa/{tone,car}_{nodoppler,doppler}_foa.wav (+ stereo, labels)")


if __name__ == "__main__":
    main()
