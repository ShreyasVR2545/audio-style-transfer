# Small experiment: which properties of the output come from the content clip and which from the style clip.
import json
import re
import time
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nst

PRESETS = Path(__file__).parent / "data" / "presets"
OUT = Path(__file__).parent / "results"

CONTENTS = ["bdl_a0001", "slt_a0002", "church_bells"]
STYLES = ["rms_a0003", "clb_a0003", "rain", "crackling_fire", "wind", "crickets"]
METHODS = ["envelope", "audio", "image"]
SWEEP = [1, 3, 10, 30, 100, 300, 1000, 10000]


def load(name, sr=nst.SR):
    return librosa.load(PRESETS / f"{name}.wav", sr=sr)[0]


_asr = None


def transcribe(y, sr=nst.SR):
    global _asr
    if _asr is None:
        from transformers import pipeline
        _asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en",
                        device=0 if torch.cuda.is_available() else -1)
    return _asr({"raw": y.astype(np.float32), "sampling_rate": sr})["text"]


def wer(ref, hyp):
    norm = lambda s: re.sub(r"[^a-z' ]", " ", s.lower()).split()
    r, h = norm(ref), norm(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1] / max(len(r), 1)


def describe(y, sr=nst.SR, pitch=True):
    S = np.abs(librosa.stft(y, n_fft=nst.N_FFT, hop_length=nst.HOP))
    d = dict(
        centroid=librosa.feature.spectral_centroid(S=S, sr=sr).mean(),
        bandwidth=librosa.feature.spectral_bandwidth(S=S, sr=sr).mean(),
        flatness=librosa.feature.spectral_flatness(S=S).mean(),
        # floor at -60 dB below peak, otherwise faint noise in the silences between words dominates the MFCCs
        mfcc=librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=64),
                                                          ref=np.max, top_db=60), n_mfcc=20)[1:].mean(1),
        onset=librosa.onset.onset_strength(S=librosa.amplitude_to_db(S), sr=sr),
        rms=librosa.feature.rms(S=S, frame_length=nst.N_FFT)[0],
    )
    if pitch:
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr, frame_length=2048)
        d["f0"] = np.nanmedian(f0) if voiced.mean() > 0.1 else np.nan
    return d


def corr(a, b):
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def toward(o, c, s):
    # 0 = output looks like content, 1 = looks like style (on the line between them)
    return float((o - c) / (s - c)) if abs(s - c) > 1e-9 else np.nan


def metrics(c, s, o, c_txt=None, out_y=None):
    row = {}
    for k in ["centroid", "bandwidth", "flatness"]:
        row[k] = float(o[k])
        row[f"{k}_toward_style"] = toward(o[k], c[k], s[k])
    dc, ds = np.linalg.norm(o["mfcc"] - c["mfcc"]), np.linalg.norm(o["mfcc"] - s["mfcc"])
    row["mfcc_d_content"], row["mfcc_d_style"] = float(dc), float(ds)
    row["mfcc_toward_style"] = float(dc / (dc + ds))
    if "f0" in o:
        row["f0_content"], row["f0_style"], row["f0_out"] = float(c["f0"]), float(s["f0"]), float(o["f0"])
    row["onset_corr"] = corr(o["onset"], c["onset"])
    row["rms_corr"] = corr(o["rms"], c["rms"])
    if c_txt is not None:
        row["transcript"] = transcribe(out_y)
        row["wer"] = wer(c_txt, row["transcript"])
    return row


def show(spec, ax, title=""):
    ax.imshow(np.asarray(spec), origin="lower", aspect="auto", cmap="magma", vmin=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def run_one(c_y, s_y, method, init, s_wt, steps=300):
    t = time.time()
    y, spec, losses = nst.transfer(c_y, s_y, method, init, s_wt, steps=steps)
    return y, spec, losses, time.time() - t


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    clips = {n: load(n) for n in CONTENTS + STYLES}
    desc = {n: describe(y) for n, y in clips.items()}
    txt = {n: transcribe(clips[n]) for n in CONTENTS if not n.startswith("church")}
    print("content transcripts:", json.dumps(txt, indent=1))

    rows, specs = [], {}

    def record(cn, sn, method, init, s_wt, tag="grid"):
        y, spec, losses, dt = run_one(clips[cn], clips[sn], method, init, s_wt)
        assert np.isfinite(y).all() and np.abs(y).max() > 1e-3, (cn, sn, method)
        name = f"{cn}__{sn}__{method}_{init}_{s_wt:g}"
        sf.write(OUT / "runs" / f"{name}.wav", y, nst.SR)
        row = dict(tag=tag, content=cn, style=sn, method=method, init=init, s_wt=s_wt, secs=round(dt, 1),
                   **{f"loss_{k}": v for k, v in losses.items()})
        row.update(metrics(desc[cn], desc[sn], describe(y), txt.get(cn), y))
        rows.append(row)
        specs[name] = spec
        print(name, {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()
                     if k in ["mfcc_toward_style", "onset_corr", "f0_out", "wer", "secs"]})

    for cn in CONTENTS:
        for sn in STYLES:
            for m in METHODS:
                record(cn, sn, m, "content", 100)
    for sn in STYLES:
        for m in ["audio", "image"]:
            record("bdl_a0001", sn, m, "noise", 100, tag="noise")
    for w in SWEEP:
        for m in ["audio", "image"]:
            record("bdl_a0001", "rain", m, "content", w, tag="sweep")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics.csv", index=False, float_format="%.4f")
    plots(df, clips, specs)


def plots(df, clips, specs):
    cspec = {n: nst.to_spec(y)[0].cpu().numpy() for n, y in clips.items()}

    # one spectrogram grid per content clip: rows are styles
    for cn in CONTENTS:
        fig, axs = plt.subplots(len(STYLES), 5, figsize=(15, 2 * len(STYLES)))
        for r, sn in enumerate(STYLES):
            show(cspec[cn], axs[r, 0], f"content {cn}")
            show(cspec[sn], axs[r, 1], f"style {sn}")
            for j, m in enumerate(METHODS):
                show(specs[f"{cn}__{sn}__{m}_content_100"], axs[r, j + 2], m)
        fig.tight_layout()
        fig.savefig(OUT / f"grid_{cn}.png", dpi=60)
        plt.close(fig)

    fig, axs = plt.subplots(2, len(STYLES), figsize=(3 * len(STYLES), 4.5))
    for j, sn in enumerate(STYLES):
        for i, m in enumerate(["audio", "image"]):
            show(specs[f"bdl_a0001__{sn}__{m}_noise_100"], axs[i, j], f"{m} noise-init, {sn}")
    fig.tight_layout()
    fig.savefig(OUT / "noise_init.png", dpi=60)
    plt.close(fig)

    # summary: per method, averaged over the content-init grid, split by voice vs texture style
    g = df[df.tag == "grid"].copy()
    g["style_kind"] = np.where(g["style"].str.contains("_a0"), "voice style", "texture style")
    cols = ["mfcc_toward_style", "centroid_toward_style", "onset_corr", "rms_corr", "wer"]
    summ = g.groupby(["style_kind", "method"])[cols].mean().round(3)
    summ.to_csv(OUT / "summary.csv")
    print(summ)

    fig, axs = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.2))
    for ax, col in zip(axs, cols):
        summ[col].unstack(0).loc[METHODS].plot.bar(ax=ax, rot=0, legend=ax is axs[0])
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=80)
    plt.close(fig)

    pitch_plot(clips)

    sw = df[df.tag == "sweep"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
    for m, d in sw.groupby("method"):
        for ax, col in zip(axs, ["mfcc_toward_style", "flatness_toward_style", "onset_corr", "wer"]):
            ax.plot(d.s_wt, d[col], "o-", label=m)
            ax.set_xscale("log")
            ax.set_title(col, fontsize=10)
            ax.set_xlabel("style weight")
    axs[0].legend()
    fig.suptitle("bdl_a0001 + rain, style-weight sweep", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "sweep.png", dpi=80)
    plt.close(fig)


def pitch_plot(clips):
    # does the output follow the content's intonation or the style speaker's register?
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.2), sharey=True)
    for ax, (cn, sn) in zip(axs, [("bdl_a0001", "clb_a0003"), ("slt_a0002", "rms_a0003")]):
        tracks = [(cn, clips[cn], "k"), (sn, clips[sn], "0.6")]
        tracks += [(m, librosa.load(OUT / "runs" / f"{cn}__{sn}__{m}_content_100.wav", sr=nst.SR)[0], col)
                   for m, col in [("envelope", "tab:green"), ("audio", "tab:blue"), ("image", "tab:orange")]]
        for label, y, col in tracks:
            f0 = librosa.pyin(y, fmin=60, fmax=500, sr=nst.SR, frame_length=2048)[0]
            ax.plot(librosa.times_like(f0, sr=nst.SR, hop_length=512), f0, ".", ms=3, color=col, label=label)
        ax.set_title(f"content {cn}, style {sn}", fontsize=10)
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8, markerscale=3)
    axs[0].set_ylabel("pyin f0 (Hz)")
    fig.tight_layout()
    fig.savefig(OUT / "pitch.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()
