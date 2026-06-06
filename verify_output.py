"""生成された SpatialScaper FOA 出力の健全性チェック（比較表の定量項目用）。
- チャンネル数/SR/長さ
- 出現クラス（ラベル）
- DOA 範囲（az/el）, 距離の有無
"""
import glob
import numpy as np
import soundfile as sf

CLASSES = ["Bird", "BirdVocalization", "Car", "Gunshot", "Rain",
           "Subway", "Thunder", "Train", "Walk"]

def main():
    wavs = sorted(glob.glob("output_compare/foa/*.wav"))
    print(f"=== {len(wavs)} FOA files ===")
    for w in wavs:
        x, sr = sf.read(w)
        print(f"  {w.split('/')[-1]:<28} shape={x.shape} sr={sr} dur={x.shape[0]/sr:.1f}s "
              f"peak={np.max(np.abs(x)):.3f}")

    labs = sorted(glob.glob("output_compare/labels/*.csv"))
    all_cls, az_all, el_all, dist_all = set(), [], [], []
    ncols = None
    nrows = 0
    for lp in labs:
        arr = np.loadtxt(lp, delimiter=",", ndmin=2)
        if arr.size == 0:
            continue
        ncols = arr.shape[1]
        nrows += arr.shape[0]
        all_cls.update(int(c) for c in arr[:, 1])
        az_all.extend(arr[:, 3].tolist())
        el_all.extend(arr[:, 4].tolist())
        if arr.shape[1] >= 6:
            dist_all.extend(arr[:, 5].tolist())
    print(f"\n=== labels: {len(labs)} files, {nrows} rows, {ncols} cols ===")
    print("  classes present (id->name):",
          {c: (CLASSES[c] if c < len(CLASSES) else '?') for c in sorted(all_cls)})
    if az_all:
        print(f"  azimuth   range: [{min(az_all):.0f}, {max(az_all):.0f}] deg")
        print(f"  elevation range: [{min(el_all):.0f}, {max(el_all):.0f}] deg")
    if dist_all:
        print(f"  distance  range: [{min(dist_all):.2f}, {max(dist_all):.2f}] m  (距離ラベル有り)")
    else:
        print("  distance: なし")

if __name__ == "__main__":
    main()
