"""ドップラー検証用：同じ音源233472を SpatialScaper で“移動”させて1本生成（最小構成）。
左(-x)→右(+x)へ、マイク近傍(y=0.4m)を通過する直線軌跡。fig_car.py の静止版を流用。
出力: output_compare/foa/car_moving.wav  (+ labels/car_moving.csv)
"""
import random
import numpy as np
import soundfile as sf
import spatialscaper as ss
import spatialscaper.core as ss_core

CLASSES = ["Bird", "BirdVocalization", "Car", "Gunshot", "Rain",
           "Subway", "Thunder", "Train", "Walk"]
ss_core.__DCASE_SOUND_EVENT_CLASSES__ = {c: i for i, c in enumerate(CLASSES)}

FOREGROUND_DIR = "datasets/sound_event_datasets/seld_match"
RIR_DIR = "datasets/rir_datasets"
SR = 24000
CAR_CLIP = "../SELD-Data-Generator/database/FSD50K/FSD50K.dev_audio/233472.wav"
OUT_TRACK = "output_compare/foa/car_moving"
OUT_LABEL = "output_compare/labels/car_moving"


def main():
    random.seed(11); np.random.seed(11)
    ssc = ss.Scaper(duration=30.0, foreground_dir=FOREGROUND_DIR, rir_dir=RIR_DIR,
                    fmt="foa", room="metu", use_room_ambient_noise=False,
                    max_event_overlap=1, sr=SR, max_event_dur=30.0, speed_limit=30.0)
    ssc.ref_db = -65
    # まず静止で1イベント追加（source/duration/label等を確定）→ 軌跡を上書きして“移動”に
    ssc.add_event(label=("const", "Car"), source_file=("choose", [CAR_CLIP]),
                  event_time=("const", 1.0), event_position=("static", None))
    e = ssc.fg_events[0]
    dur = float(e.event_duration)
    N = max(2, int(round(dur * 10)))                      # 10fps相当の軌跡点数
    # 静止点と同じ look-direction (az=-18, el=-32) を“最接近”で通過する pass-by。
    # 距離はドップラー検証のため可変（遠->近(d_min)->遠）。距離一定だと半径速度0でドップラー出ない。
    az0, el0, d_min = np.deg2rad(-18.0), np.deg2rad(-32.0), 1.0
    c = np.array([d_min*np.cos(el0)*np.cos(az0),
                  d_min*np.cos(el0)*np.sin(az0),
                  d_min*np.sin(el0)])                     # 最接近点(az=-18,el=-32,r=1.0m)
    dirv = np.array([-np.sin(az0), np.cos(az0), 0.0])     # 方位接線（c に垂直＝c が最接近）
    s = np.linspace(-1.25, 1.25, N)                       # 室内(±1.5m)に収まる範囲
    traj = [(c + si*dirv).tolist() for si in s]           # 静止と同じ向きを最接近で通過・距離可変
    ssc.fg_events[0] = e._replace(event_position=traj)
    ssc.generate(OUT_TRACK, OUT_LABEL)

    # ---- 報告用の確認 ----
    foa, _ = sf.read(OUT_TRACK + ".wav")
    print(f"saved {OUT_TRACK}.wav  shape={foa.shape}")
    lab = np.loadtxt(OUT_LABEL + ".csv", delimiter=",", ndmin=2)
    print(f"az: {lab[0,3]:.0f}->{lab[-1,3]:.0f}  (min {lab[:,3].min():.0f}, max {lab[:,3].max():.0f})")
    print(f"el: min {lab[:,4].min():.0f}, max {lab[:,4].max():.0f}  (closest ~ -32)")
    print(f"r : min {lab[:,5].min():.2f}, max {lab[:,5].max():.2f} m  (varies -> radial velocity = Doppler source)")
    print(f"frames={len(lab)}, event_dur={dur:.1f}s  (moving: az/el/r all change)")


if __name__ == "__main__":
    main()
