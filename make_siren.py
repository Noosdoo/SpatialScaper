"""
make_siren.py — 緊急車両サイレン（日本の救急車「ピーポー」）を合成して siren.wav を作る。
car の音源 233472.wav と同じ長さ・同じ場所(FSD50K.dev_audio)に置き、car と全く同じに扱えるようにする。
方式: 960Hz(ピー)/770Hz(ポー)を 0.65s 毎に交互。位相連続でクリック無し。+2倍音で少しサイレンらしく。
本物の録音があれば siren.wav を差し替えるだけで全デモがそのまま使える。
"""
import numpy as np
import soundfile as sf
import librosa

SR = 24000
CAR   = "../SELD-Data-Generator/database/FSD50K/FSD50K.dev_audio/233472.wav"
SIREN = "../SELD-Data-Generator/database/FSD50K/FSD50K.dev_audio/siren.wav"

F_HI, F_LO, SEG = 960.0, 770.0, 0.65   # ピー/ポーの周波数[Hz]・各長さ[s]


def main():
    ref = librosa.load(CAR, sr=SR, mono=True)[0]
    dur = len(ref) / SR                     # car と同じ長さに合わせる
    t = np.arange(int(dur * SR)) / SR

    f = np.where((t % (2 * SEG)) < SEG, F_HI, F_LO)   # ピーポー交互
    phase = 2 * np.pi * np.cumsum(f) / SR             # 位相連続（切替時のクリック無し）
    s = np.sin(phase) + 0.25 * np.sin(2 * phase)      # +2倍音で少し厚み
    s = 0.9 * s / np.max(np.abs(s))

    sf.write(SIREN, s.astype(np.float32), SR)
    print(f"siren.wav: {dur:.1f}s, ピーポー {F_HI:.0f}/{F_LO:.0f}Hz each {SEG}s @ {SR}Hz")
    print(f"saved -> {SIREN}")


if __name__ == "__main__":
    main()
