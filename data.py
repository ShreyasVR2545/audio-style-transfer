# Fetch a small set of preset clips into data/presets/ (only the files we need, no full archives).
import json
import urllib.parse
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
IOWA = "https://theremin.music.uiowa.edu/sound files/"

SENTENCES = {
    "a0001": "Author of the danger trail, Philip Steels, etc.",
    "a0002": "Not at this particular case, Tom, apologized Whittemore.",
    "a0003": "For the twentieth time that evening the two men shook hands.",
}
# speaker -> (description, utterances); a0002/a0003 are 3-5 s for everyone, bdl a0001 is used by exp.py
VOICES = {
    "bdl": ("US male", ["a0001", "a0002", "a0003"]),
    "slt": ("US female", ["a0002", "a0003"]),
    "rms": ("US male, deep", ["a0002", "a0003"]),
    "clb": ("US female", ["a0002", "a0003"]),
    "awb": ("Scottish male", ["a0002", "a0003"]),
    "jmk": ("Canadian male", ["a0002", "a0003"]),
    "ksp": ("Indian English male", ["a0002", "a0003"]),
}
TEXTURES = ["rain", "crackling_fire", "wind", "crickets", "sea_waves", "engine", "church_bells",
            "thunderstorm", "chirping_birds", "clock_tick", "helicopter", "vacuum_cleaner", "keyboard_typing",
            "clapping", "train", "pouring_water"]
# name -> (label, path under the Iowa MIS "sound files" folder)
INSTRUMENTS = {
    "violin": ("violin, bowed A4-B4", "MIS/Strings/violin2012/Violin.arco.mf.sulA.A4B4.stereo.aif"),
    "cello": ("cello, bowed G2-Gb3", "MIS/Strings/cello2012/Cello.arco.mf.sulG.G2Gb3.stereo.aif"),
    "flute": ("flute, A4", "MIS Pitches - 2014/Woodwinds/Flute/Flute.vib.ff.A4.stereo.aif"),
    "clarinet": ("clarinet, G3", "MIS Pitches - 2014/Woodwinds/Bb Clarinet/BbClarinet.ff.G3.stereo.aif"),
    "alto_sax": ("alto sax, A3", "MIS Pitches - 2014/Woodwinds/Eb Alto Saxophone/AltoSax.vib.ff.A3.stereo.aif"),
    "trumpet": ("trumpet, C4", "MIS Pitches - 2014/Brass/BbTrumpet/Trumpet.vib.ff.C4.stereo.aif"),
    "horn": ("french horn, F3", "MIS Pitches - 2014/Brass/Horn/Horn.ff.F3.stereo.aif"),
    "piano": ("piano, C4", "MIS/Piano_Other/piano/Piano.mf.C4.aiff"),
}


def fetch(url, dst):
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(urllib.parse.quote(url, safe=":/"), dst)
    return dst


def trim(y, sr=SR, max_sec=MAX_SEC, top_db=None):
    # a fixed 40 dB trim does nothing when there is a noise floor (mic recordings), so by default cut at
    # 6 dB above the floor (the quietest frame); steady sounds with no quiet part keep 40 dB
    if top_db is None:
        floor = librosa.amplitude_to_db(librosa.feature.rms(y=y)[0], ref=np.max).min()
        top_db = 40 if floor > -15 else float(np.clip(-floor - 6, 12, 40))
        a, b = librosa.effects.trim(y, top_db=top_db)[1]
        pad = int(0.1 * sr)  # the higher threshold clips soft word onsets, give some back
        y = y[max(0, a - pad): b + pad]
    else:
        y, _ = librosa.effects.trim(y, top_db=top_db)
    y = y[: int(max_sec * sr)]
    return (0.9 * y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def load(path, sr=SR, max_sec=MAX_SEC, top_db=40):
    return trim(librosa.load(path, sr=sr, mono=True)[0], sr, max_sec, top_db)


def active_frac(y):
    # ESC-50 clips are zero-padded to 5 s, prefer the ones that are actually full of sound
    rms = librosa.feature.rms(y=y)[0]
    return float((rms > 0.1 * rms.max()).mean())


def main():
    PRESETS.mkdir(parents=True, exist_ok=True)
    meta = []

    for spk, (desc, utts) in VOICES.items():
        for utt in utts:
            url = ARCTIC.format(spk=spk, utt=f"arctic_{utt}")
            name = f"{spk}_{utt}"
            # awb was recorded with an audible noise floor, a 40 dB trim leaves ~1 s of hiss at the ends
            y = load(fetch(url, RAW / "arctic" / spk / f"arctic_{utt}.wav"), top_db=25 if spk == "awb" else 40)
            sf.write(PRESETS / f"{name}.wav", y, SR)
            words = " ".join(SENTENCES[utt].split()[:4])
            meta.append(dict(name=name, category="voice", label=f"Voice: {desc} ({spk}), \"{words}...\"",
                             source=url, license="CMU ARCTIC (free use, attribution)"))

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
        meta.append(dict(name=cat, category="texture", label=f"Texture: {cat.replace('_', ' ')}",
                         source=ESC_WAV.format(fn=fn), license="CC BY-NC 3.0 (ESC-50 / Freesound)"))

    for name, (label, path) in INSTRUMENTS.items():
        y = load(fetch(IOWA + path, RAW / "iowa" / Path(path).name))
        sf.write(PRESETS / f"{name}.wav", y, SR)
        meta.append(dict(name=name, category="instrument", label=f"Instrument: {label}", source=IOWA + path,
                         license="University of Iowa MIS (free to use for any project, no restrictions)"))

    (PRESETS / "presets.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(meta)} presets to {PRESETS}")


if __name__ == "__main__":
    main()
