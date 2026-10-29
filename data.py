# Fetch a small set of preset clips into data/presets/ (only the files we need, no full archives).
import json
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 22050
MAX_SEC = 6.0
ROOT = Path(__file__).parent / "data"
RAW = ROOT / "raw"
PRESETS = ROOT / "presets"

ARCTIC = "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_{spk}_arctic/wav/{utt}.wav"
ESC_CSV = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC_WAV = "https://github.com/karolpiczak/ESC-50/raw/master/audio/{fn}"

# speaker -> (description, utterances)
VOICES = {
    "bdl": ("US male", ["arctic_a0001", "arctic_a0002"]),
    "slt": ("US female", ["arctic_a0001", "arctic_a0002"]),
    "rms": ("US male, deeper", ["arctic_a0003"]),
    "clb": ("US female", ["arctic_a0003"]),
    "awb": ("Scottish male", ["arctic_a0004"]),
}
TEXTURES = ["rain", "crackling_fire", "wind", "crickets", "sea_waves", "engine", "church_bells"]


def fetch(url, dst):
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dst)
    return dst


def load(path, sr=SR, max_sec=MAX_SEC):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=40)
    y = y[: int(max_sec * sr)]
    return (0.9 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def active_frac(y):
    # ESC-50 clips are zero-padded to 5 s, prefer the ones that are actually full of sound
    rms = librosa.feature.rms(y=y)[0]
    return float((rms > 0.1 * rms.max()).mean())


def main():
    PRESETS.mkdir(parents=True, exist_ok=True)
    meta = []

    for spk, (desc, utts) in VOICES.items():
        for utt in utts:
            raw = fetch(ARCTIC.format(spk=spk, utt=utt), RAW / "arctic" / spk / f"{utt}.wav")
            name = f"{spk}_{utt.split('_')[1]}"
            sf.write(PRESETS / f"{name}.wav", load(raw), SR)
            meta.append(dict(name=name, category="voice", desc=f"CMU ARCTIC {spk} ({desc})",
                             source=ARCTIC.format(spk=spk, utt=utt), license="CMU ARCTIC (free use, attribution)"))

    csv = fetch(ESC_CSV, RAW / "esc50" / "esc50.csv").read_text().splitlines()[1:]
    for cat in TEXTURES:
        cands = [ln.split(",")[0] for ln in csv if ln.split(",")[3] == cat][:5]
        best = None
        for fn in cands:
            y = load(fetch(ESC_WAV.format(fn=fn), RAW / "esc50" / fn))
            score = active_frac(y) * len(y)
            if best is None or score > best[0]:
                best = (score, fn, y)
        _, fn, y = best
        sf.write(PRESETS / f"{cat}.wav", y, SR)
        meta.append(dict(name=cat, category="texture", desc=f"ESC-50 {cat} ({fn})",
                         source=ESC_WAV.format(fn=fn), license="CC BY-NC 3.0 (ESC-50 / Freesound)"))

    (PRESETS / "presets.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(meta)} presets to {PRESETS}")


if __name__ == "__main__":
    main()
