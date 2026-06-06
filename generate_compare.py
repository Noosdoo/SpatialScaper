"""
比較用 SpatialScaper 生成スクリプト（SELD-Data-Generator との対照実験）。
最小構成: metu / FOA 4ch / 24kHz / 60s / max_event_overlap=1 / 屋外物理なし。
背景は合成ホワイトノイズ（use_room_ambient_noise=False）で TAU-SNoise_DB 不要。

使い方:
  python generate_compare.py [NSCAPES] [N_EVENTS]
"""
import os
import sys
import numpy as np
import spatialscaper as ss
import spatialscaper.core as ss_core

# SELD-Data-Generator と同一の9クラス（同じ音源を使用）。
# SpatialScaper は DCASE13クラス固定なので、ライブラリ非改変でクラス辞書を差し替える。
CLASSES = ["Bird", "BirdVocalization", "Car", "Gunshot", "Rain",
           "Subway", "Thunder", "Train", "Walk"]
ss_core.__DCASE_SOUND_EVENT_CLASSES__ = {c: i for i, c in enumerate(CLASSES)}

# ---- 条件（比較で揃えるパラメータ） ----
NSCAPES = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N_EVENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 6  # 1 scape あたりのイベント数（固定）
FOREGROUND_DIR = "datasets/sound_event_datasets/seld_match"
RIR_DIR = "datasets/rir_datasets"
ROOM = "metu"
FORMAT = "foa"          # FOA 4ch (ACN/SN3D)
DURATION = 60.0
SR = 24000
MAX_OVERLAP = 1         # polyphony 1（SELD-Data-Generator mode1 と同条件）
REF_DB = -65
OUTPUT_DIR = "output_compare"
SEED = 2024


def generate_soundscape(index):
    track_name = f"fold0_room0_mix{index:03d}"
    ssc = ss.Scaper(
        duration=DURATION,
        foreground_dir=FOREGROUND_DIR,
        rir_dir=RIR_DIR,
        fmt=FORMAT,
        room=ROOM,
        use_room_ambient_noise=False,   # TAU-SNoise 不要・合成ノイズ背景
        max_event_overlap=MAX_OVERLAP,
        sr=SR,
        speed_limit=2.0,
    )
    ssc.ref_db = REF_DB
    ssc.add_background()                 # 合成ホワイトノイズ背景
    for _ in range(N_EVENTS):
        ssc.add_event()
    audiofile = os.path.join(OUTPUT_DIR, FORMAT, track_name)
    labelfile = os.path.join(OUTPUT_DIR, "labels", track_name)
    ssc.generate(audiofile, labelfile)


if __name__ == "__main__":
    np.random.seed(SEED)
    import random
    random.seed(SEED)
    print(f"SpatialScaper compare-gen: NSCAPES={NSCAPES}, N_EVENTS={N_EVENTS}, "
          f"room={ROOM}, fmt={FORMAT}, dur={DURATION}, overlap={MAX_OVERLAP}")
    for i in range(NSCAPES):
        print(f"--- soundscape {i+1}/{NSCAPES} ---")
        generate_soundscape(i)
    print("DONE")
