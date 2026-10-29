# Local demo: record or upload a content clip, pick a style, watch the spectrogram change, listen.
import argparse
import json
from pathlib import Path

import gradio as gr
import librosa
import matplotlib
import numpy as np
import pandas as pd

import exp
import nst

PRESETS = Path(__file__).parent / "data" / "presets"
META = json.loads((PRESETS / "presets.json").read_text())
CUSTOM = "custom (record / upload below)"
CHOICES = [f"{m['category']}: {m['name']}" for m in sorted(META, key=lambda m: m["category"])] + [CUSTOM]
CMAP = matplotlib.colormaps["magma"]
MAX_SEC = 6.0


def to_mono(audio):
    sr, y = audio
    y = y.astype(np.float32)
    if np.issubdtype(audio[1].dtype, np.integer):
        y /= np.iinfo(audio[1].dtype).max
    if y.ndim > 1:
        y = y.mean(1)
    y = librosa.resample(y, orig_sr=sr, target_sr=nst.SR)[: int(MAX_SEC * nst.SR)]
    return 0.9 * y / (np.abs(y).max() + 1e-9)


def preset_audio(choice):
    if choice == CUSTOM:
        return None
    return exp.load(choice.split(": ")[1])


def img(spec, vmax):
    # render straight to an rgb array, much cheaper than a matplotlib figure per frame
    a = np.asarray(spec.cpu() if hasattr(spec, "cpu") else spec)
    a = np.clip(a / vmax, 0, 1)[::-1]
    return (CMAP(a)[..., :3] * 255).astype(np.uint8)


def table(c, s, o, txt=None):
    f0 = lambda d: round(float(d["f0"]), 1) if np.isfinite(d.get("f0", np.nan)) else "-"
    rows = [
        ["spectral centroid (Hz)", c["centroid"], s["centroid"], o["centroid"]],
        ["spectral bandwidth (Hz)", c["bandwidth"], s["bandwidth"], o["bandwidth"]],
        ["spectral flatness", c["flatness"], s["flatness"], o["flatness"]],
        ["median f0 (Hz)", f0(c), f0(s), f0(o)],
        ["MFCC distance to content", 0.0, np.linalg.norm(s["mfcc"] - c["mfcc"]), np.linalg.norm(o["mfcc"] - c["mfcc"])],
        ["MFCC distance to style", np.linalg.norm(c["mfcc"] - s["mfcc"]), 0.0, np.linalg.norm(o["mfcc"] - s["mfcc"])],
        ["onset envelope corr. with content", 1.0, "-", exp.corr(o["onset"], c["onset"])],
        ["RMS envelope corr. with content", 1.0, "-", exp.corr(o["rms"], c["rms"])],
    ]
    if txt:
        rows.append(["transcript (Whisper)", txt[0], "-", txt[1]])
        rows.append(["word error rate vs content", 0.0, "-", exp.wer(txt[0], txt[1])])
    fmt = lambda v: (f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}") if isinstance(v, (float, np.floating)) else v
    return pd.DataFrame([[fmt(v) for v in r] for r in rows], columns=["metric", "content", "style", "output"])


def go(c_audio, choice, s_audio, method, init, s_wt, steps, do_asr):
    if c_audio is None:
        raise gr.Error("Record or upload a content clip first.")
    c = to_mono(c_audio)
    if choice == CUSTOM:
        if s_audio is None:
            raise gr.Error("Pick a preset style or record / upload one.")
        s = to_mono(s_audio)
    else:
        s = preset_audio(choice)

    c_spec, c_phase = nst.to_spec(c)
    s_spec, _ = nst.to_spec(s)
    vmax = max(c_spec.max(), s_spec.max()).item()
    c_img, s_img = img(c_spec, vmax), img(s_spec, vmax)
    yield c_img, s_img, None, "starting...", None, None

    every = max(5, int(steps) // 30)
    for i, spec, losses in nst.run(c_spec, s_spec, method, init, float(s_wt), steps=int(steps), every=every):
        msg = f"step {i} / {int(steps)}"
        if losses:
            msg += f" &nbsp; style loss {losses['style']:.4f} &nbsp; content loss {losses['content']:.4f}"
        yield c_img, s_img, img(spec, vmax), msg, None, None

    y = nst.griffin_lim(spec, c_phase, length=len(c))
    yield c_img, s_img, img(spec, vmax), msg + " &nbsp; (computing metrics...)", (nst.SR, y), None

    txt = (exp.transcribe(c), exp.transcribe(y)) if do_asr else None
    tab = table(exp.describe(c), exp.describe(s), exp.describe(y), txt)
    yield c_img, s_img, img(spec, vmax), msg + " &nbsp; done", (nst.SR, y), tab


def preview(choice):
    y = preset_audio(choice)
    return gr.update(value=None if y is None else (nst.SR, y), visible=choice != CUSTOM), gr.update(visible=choice == CUSTOM)


with gr.Blocks(title="Spectrogram style transfer") as demo:
    gr.Markdown("## Spectrogram style transfer\n"
                "Gatys-style Gram matching on log-STFT spectrograms, inverted with Griffin-Lim. "
                "*Audio view*: random 1D conv over time, frequency bins as channels. "
                "*Image view*: pretrained VGG-19 on the spectrogram as a picture. "
                "*Envelope*: plain DSP baseline that swaps the long-term spectral envelope.")
    with gr.Row():
        with gr.Column():
            c_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="content (max 6 s)",
                            value=str(PRESETS / "bdl_a0001.wav"))
        with gr.Column():
            choice = gr.Dropdown(CHOICES, value="texture: rain", label="style")
            s_play = gr.Audio(value=(nst.SR, preset_audio("texture: rain")), label="style clip", interactive=False)
            s_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="custom style", visible=False)
    with gr.Row():
        method = gr.Radio(["audio", "image", "envelope"], value="audio", label="method")
        init = gr.Radio(["content", "noise"], value="content", label="init")
        s_wt = gr.Slider(1, 10000, value=100, step=1, label="style weight (content weight is 1)")
        steps = gr.Slider(50, 1000, value=300, step=50, label="steps")
        do_asr = gr.Checkbox(False, label="Whisper transcript / WER (slow first time)")
    btn = gr.Button("Run", variant="primary")
    status = gr.Markdown()
    with gr.Row():
        c_img = gr.Image(label="content spectrogram", show_label=True)
        s_img = gr.Image(label="style spectrogram")
        o_img = gr.Image(label="output spectrogram (live)")
    out = gr.Audio(label="output", type="numpy")
    tab = gr.Dataframe(label="what changed", interactive=False)

    choice.change(preview, choice, [s_play, s_in])
    btn.click(go, [c_in, choice, s_in, method, init, s_wt, steps, do_asr], [c_img, s_img, o_img, status, out, tab])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public gradio link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo.queue().launch(share=args.share, server_port=args.port)
