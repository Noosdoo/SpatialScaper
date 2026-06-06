"""SELD-Data-Generator が task1 で実際に使う31クリップを、正しいクラスごとに
SpatialScaper の foreground レイアウトへコピーする。
入力: ../SELD-Data-Generator/clip_class_map.json (map_clips.py が生成)
出力: datasets/sound_event_datasets/seld_match/<SafeClass>/<clip>.wav
"""
import json, shutil
from pathlib import Path

SELD = Path("../SELD-Data-Generator")
SRC_AUDIO = SELD / "database/FSD50K/FSD50K.dev_audio"
DST = Path("datasets/sound_event_datasets/seld_match")

# SELD-Data-Generator のクラス名 -> ファイルシステム安全なフォルダ名（=SpatialScaperラベル名）
SAFE = {
    "Bird": "Bird",
    "Bird vocalization, bird call, bird song": "BirdVocalization",
    "Car": "Car",
    "Gunshot, gunfire": "Gunshot",
    "Rain": "Rain",
    "Subway, metro, underground": "Subway",
    "Thunder": "Thunder",
    "Train": "Train",
    "Walk, footsteps": "Walk",
}


def main():
    data = json.load(open(SELD / "clip_class_map.json", encoding="utf-8"))
    per_class = data["per_class"]
    DST.mkdir(parents=True, exist_ok=True)
    total = 0
    safe_classes = []
    for cls, clips in per_class.items():
        safe = SAFE.get(cls, cls.split(",")[0].replace(" ", "_"))
        safe_classes.append(safe)
        outdir = DST / safe
        outdir.mkdir(parents=True, exist_ok=True)
        for clip in clips:
            src = SRC_AUDIO / clip
            if not src.exists():
                print(f"  WARN missing source: {src}")
                continue
            shutil.copyfile(src, outdir / clip)
            total += 1
        print(f"  {safe:<18} {len(clips)} clip(s)")
    print(f"\ncopied {total} clips into {DST}")
    print("CLASSES =", sorted(safe_classes))


if __name__ == "__main__":
    main()
