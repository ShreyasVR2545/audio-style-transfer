# Local demo: pick or record a content clip and a style clip, then watch audio -> image -> style transfer -> audio.
import argparse
import json
import time

import gradio as gr
import librosa
import numpy as np
import scipy.signal as ss

import data
import nst
import viz

META = json.loads((data.PRESETS / "presets.json").read_text())
OWN = "own"
CHOICES = [(m["label"], m["name"]) for m in META] + [("Record or upload my own clip...", OWN)]
METHODS = [("Audio view (random 1D conv)", "audio"), ("Image view (VGG-19, like Gatys)", "image"),
           ("No network (EQ match baseline)", "envelope")]
STRENGTH = {"light": 10, "medium": 50, "strong": 150, "extreme": 1000}
ANIM_FPS = 20


def prep(audio):
    # browser mic: usually 48 kHz, int16, sometimes stereo, quiet, with DC offset, hum and silence at both ends
    sr, y = audio
    y = y.astype(np.float32)
    if np.issubdtype(audio[1].dtype, np.integer):
        y /= np.iinfo(audio[1].dtype).max
    if y.ndim > 1:
        y = y.mean(1)
    y = librosa.resample(y, orig_sr=sr, target_sr=nst.SR)
    # high-pass removes DC and hum, which would otherwise dominate the lowest spectrogram rows after normalising
    y = ss.sosfiltfilt(ss.butter(4, 70, "highpass", fs=nst.SR, output="sos"), y)
    return data.trim(y)


def clip(choice, audio):
    if choice != OWN:
        return data.load(data.PRESETS / f"{choice}.wav")
    return None if audio is None else prep(audio)


def show_clip(choice, audio):
    y = clip(choice, audio)
    return gr.update(visible=choice == OWN), None if y is None else (nst.SR, y)


def pitch(y):
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=nst.SR, frame_length=2048)
    return np.nanmedian(f0) if voiced.mean() > 0.2 else None


def summary(c, y):
    cen = lambda v: librosa.feature.spectral_centroid(y=v, sr=nst.SR).mean()
    note = f"Brightness (spectral centroid) {cen(c):.0f} Hz -> {cen(y):.0f} Hz."
    pc, po = pitch(c), pitch(y)
    if pc and po:
        note += f"  Pitch {pc:.0f} Hz -> {po:.0f} Hz."
    elif pc:
        note += f"  Pitch {pc:.0f} Hz -> no clear pitch left."
    return note


def go(c_choice, c_audio, s_choice, s_audio, method, init, strength, steps):
    c, s = clip(c_choice, c_audio), clip(s_choice, s_audio)
    if c is None or s is None:
        raise gr.Error("Pick a preset or record / upload a clip for both content and style.")
    if len(c) < nst.SR // 2 or len(s) < nst.SR // 2:
        raise gr.Error("A clip is shorter than 0.5 s after trimming silence. Try recording again, a bit louder.")
    s_wt, steps = STRENGTH[strength], int(steps)
    frames = []

    def show(frame, caption, out=None, extra=(None, None, None, None)):
        frames.append(viz.jpeg(frame))
        return (frame, caption, out) + extra

    c_spec, c_phase = nst.to_spec(c)
    s_spec, _ = nst.to_spec(s)
    cn, sn = c_spec.cpu().numpy(), s_spec.cpu().numpy()
    vmax = max(cn.max(), sn.max())

    n = 48
    for k in range(n + 1):
        yield show(viz.analysis_frame(c, s, cn, sn, vmax, k / n), "**1 / 3  Audio -> image** (STFT)")
        time.sleep(1 / ANIM_FPS)

    cap = f"**2 / 3  Style matching**, style weight {s_wt} (content weight 1)"
    every = max(5, steps // 40)
    hist = []
    if init == "content" and method != "envelope":
        yield show(viz.optim_frame(cn, vmax, 0, steps, {}, [], method), cap)
    for i, spec, losses in nst.run(c_spec, s_spec, method, init, s_wt, steps=steps, every=every, snap=64):
        if "style" in losses:
            hist.append(losses["style"])
        yield show(viz.optim_frame(spec.cpu().numpy(), vmax, i, steps, losses, hist, method), cap)
    o = spec.cpu().numpy()

    cap = "**3 / 3  Image -> audio** (Griffin-Lim)"
    gl_iter = 200
    for it, y in nst.griffin_lim_iter(spec, c_phase, gl_iter, len(c), every=5):
        yield show(viz.synth_frame(o, vmax, y, it / gl_iter, it, gl_iter), cap)
        time.sleep(1 / ANIM_FPS)

    note = summary(c, y)
    final = viz.done_frame(cn, sn, o, vmax, note)
    imgs = tuple(viz.spec_rgb(a, vmax, 400, 260) for a in (cn, sn, o))
    yield show(final, f"**Done.** {note}", (nst.SR, y), imgs + (frames,))


def video(frames, out):
    if not frames or out is None:
        raise gr.Error("Run a transfer first.")
    sr, y = out
    y = y.astype(np.float32) / (32768 if np.issubdtype(y.dtype, np.integer) else 1)
    return viz.save_video(frames, y, sr)


def default_steps(method):
    return 100 if method == "image" and nst.DEV.type == "cpu" else 300


with gr.Blocks(title="Spectrogram style transfer") as demo:
    gr.Markdown("## Spectrogram style transfer\n"
                "Turns a sound into a picture (spectrogram), runs Gatys-style neural style transfer on that "
                "picture, and turns it back into sound. Pick a content clip (what is said / played) and a style "
                "clip (whose sound it should take on), then press Run.")
    with gr.Row():
        for side in ["content", "style"]:
            with gr.Column():
                default = "bdl_a0001" if side == "content" else "crackling_fire"
                hint = ("what should be kept: words, rhythm" if side == "content"
                        else "what to borrow: colour and texture of the sound")
                dd = gr.Dropdown(CHOICES, value=default, label=side.capitalize(), info=hint)
                mic = gr.Audio(sources=["microphone", "upload"], type="numpy", visible=False,
                               label=f"Record or upload ({side}), up to 6 s is used")
                used = gr.Audio(value=(nst.SR, data.load(data.PRESETS / f"{default}.wav")),
                                label=f"{side} clip that will be used", interactive=False)
                dd.change(show_clip, [dd, mic], [mic, used])
                mic.change(show_clip, [dd, mic], [mic, used])
                if side == "content":
                    c_dd, c_mic = dd, mic
                else:
                    s_dd, s_mic = dd, mic
    gr.Markdown("Own clips: silence at the start and end is cut, hum is filtered out and the level is "
                "normalised; the players above let you hear exactly what will be used. The browser only allows "
                "the microphone on localhost or https (`--share`).")

    with gr.Accordion("Settings", open=False):
        method = gr.Radio(METHODS, value="audio", label="Method",
                          info="Audio view treats frequencies as channels (works best); image view treats "
                               "the spectrogram as a photo; the baseline only copies the style's average EQ.")
        init = gr.Radio([("Start from content", "content"), ("Start from noise", "noise")], value="content",
                        label="Starting point",
                        info="Noise rebuilds the sound from scratch: more of the style, fewer words survive.")
        strength = gr.Radio(list(STRENGTH), value="medium", label="Style strength",
                            info="Style weight relative to content: light 10, medium 50, strong 150, extreme 1000.")
        steps = gr.Slider(50, 600, value=default_steps("audio"), step=50, label="Steps",
                          info="More steps = closer match, slower. On a CPU 10 steps take ~2 s (audio view) or ~8 s (image view).")
        method.change(lambda m: default_steps(m), method, steps)

    btn = gr.Button("Run", variant="primary")
    caption = gr.Markdown()
    stage = gr.Image(label="pipeline", show_label=False, interactive=False, format="jpeg", height=540)
    out = gr.Audio(label="output", type="numpy", interactive=False)
    with gr.Row():
        c_img = gr.Image(label="content spectrogram", interactive=False)
        s_img = gr.Image(label="style spectrogram", interactive=False)
        o_img = gr.Image(label="output spectrogram", interactive=False)
    frames = gr.State([])
    with gr.Accordion("Save the animation as a video", open=False):
        vid_btn = gr.Button("Make mp4")
        vid = gr.Video(label="animation + output audio")

    btn.click(go, [c_dd, c_mic, s_dd, s_mic, method, init, strength, steps],
              [stage, caption, out, c_img, s_img, o_img, frames])
    vid_btn.click(video, [frames, out], vid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="temporary public link (also gives https, which mic needs)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo.queue().launch(share=args.share, server_port=args.port)
