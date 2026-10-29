# Frame drawing for the app's pipeline animation. Plain numpy + PIL so a frame takes a few ms.
import io
import subprocess
import tempfile

import matplotlib
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
BG = (17, 17, 22)
FG = (230, 230, 235)
DIM = (130, 130, 140)
ACCENT = (255, 190, 60)
WAVE = (120, 200, 255)
CMAP = matplotlib.colormaps["magma"]
FONT = ImageFont.load_default(size=20)
SMALL = ImageFont.load_default(size=15)


def spec_rgb(spec, vmax, w, h, upto=None):
    # low frequencies at the bottom; columns after `upto` (fraction) left blank; slight overdrive reads better
    a = np.clip(np.asarray(spec) / (0.8 * vmax), 0, 1)[::-1]
    img = (CMAP(a)[..., :3] * 255).astype(np.uint8)
    img = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR)).copy()
    if upto is not None:
        img[:, int(upto * w):] = BG
    return img


def wave_rgb(y, w, h, upto=1.0, color=WAVE):
    img = np.full((h, w, 3), BG, np.uint8)
    img[h // 2] = (50, 50, 60)
    n = int(upto * w)
    if n == 0 or len(y) == 0:
        return img
    # min/max envelope per pixel column, the way audio editors draw waveforms
    cols = np.array_split(y, w)[:n]
    peak = max(np.abs(y).max(), 1e-6)
    for x, c in enumerate(cols):
        lo, hi = c.min() / peak, c.max() / peak
        y0, y1 = int((1 - hi) * (h - 1) / 2), int((1 - lo) * (h - 1) / 2)
        img[y0:y1 + 1, x] = color
    return img


def gram_rgb(g, ref, size):
    # log scale, normalised by the style target so both heatmaps share one colour range
    lo, hi = np.log1p(np.maximum(ref, 0)).min(), np.log1p(np.maximum(ref, 0)).max()
    a = np.clip((np.log1p(np.maximum(g, 0)) - lo) / (hi - lo + 1e-9), 0, 1)
    img = (matplotlib.colormaps["viridis"](a)[..., :3] * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(img).resize((size, size), Image.NEAREST))


def canvas():
    return np.full((H, W, 3), BG, np.uint8)


def paste(cv, img, x, y):
    cv[y:y + img.shape[0], x:x + img.shape[1]] = img


def text(cv, items):
    im = Image.fromarray(cv)
    d = ImageDraw.Draw(im)
    for x, y, s, col, font in items:
        d.text((x, y), s, fill=col, font=font)
    return np.array(im)


def rect(cv, x0, y0, x1, y1, col, width=2):
    im = Image.fromarray(cv)
    ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=col, width=width)
    return np.array(im)


def line(cv, pts, col, width=2):
    im = Image.fromarray(cv)
    ImageDraw.Draw(im).line(pts, fill=col, width=width)
    return np.array(im)


def analysis_frame(c_y, s_y, c_spec, s_spec, vmax, frac):
    # stage 1: a window slides along each waveform and the spectrogram is filled in column by column
    cv = canvas()
    pw, wh, sh, top = 440, 110, 250, 95
    items = [(24, 16, "1 / 3   Audio -> image", FG, FONT),
             (24, 46, "A short window slides along the sound; each window's frequency content becomes one "
                      "column of the spectrogram.", DIM, SMALL)]
    for k, (y, spec, name) in enumerate([(c_y, c_spec, "content"), (s_y, s_spec, "style")]):
        x0 = 24 + k * (pw + 32)
        paste(cv, wave_rgb(y, pw, wh), x0, top + 24)
        paste(cv, spec_rgb(spec, vmax, pw, sh, upto=frac), x0, top + wh + 66)
        items += [(x0, top, f"{name} waveform  ({len(y) / 22050:.1f} s)", FG, SMALL),
                  (x0, top + wh + 42, f"{name} spectrogram (time ->, frequency ^)", FG, SMALL)]
        if frac < 1:
            wx = x0 + int(frac * pw)
            cv = rect(cv, wx - 4, top + 22, wx + 4, top + wh + 26, ACCENT)
            cv = line(cv, [(wx, top + wh + 26), (wx, top + wh + 66)], ACCENT)
    return text(cv, items)


def optim_frame(spec, vmax, step, steps, losses, hist, method):
    # stage 2: the output spectrogram moving from its start towards the result, with the Gram blocks
    cv = canvas()
    paste(cv, spec_rgb(spec, vmax, 600, 380), 24, 120)
    items = [(24, 16, "2 / 3   Style matching", FG, FONT),
             (24, 46, "The spectrogram itself is optimised: keep the content's features, match the style's "
                      "Gram matrix", DIM, SMALL),
             (24, 66, "(which feature channels are active together, averaged over time).", DIM, SMALL),
             (24, 96, f"output spectrogram   step {step} / {steps}", FG, SMALL)]
    if "gram" in losses:
        paste(cv, gram_rgb(losses["s_gram"], losses["s_gram"], 140), 650, 120)
        paste(cv, gram_rgb(losses["gram"], losses["s_gram"], 140), 800, 120)
        what = "filters" if method == "audio" else "relu1_1 channels"
        items += [(650, 96, "style Gram", FG, SMALL), (800, 96, "output Gram", FG, SMALL),
                  (650, 266, f"first 64 x 64 {what}", DIM, SMALL)]
    if hist:
        # log style-loss curve
        x0, y0, w, h = 650, 320, 290, 150
        cv = rect(cv, x0, y0, x0 + w, y0 + h, (60, 60, 70), 1)
        lg = np.log10(np.maximum(hist, 1e-8))
        lo, hi = lg.min(), max(lg.max(), lg.min() + 1e-6)
        xs = np.linspace(x0, x0 + w, max(len(lg), 2))[: len(lg)]
        pts = [(float(x), float(y0 + h - (v - lo) / (hi - lo) * h)) for x, v in zip(xs, lg)]
        if len(pts) > 1:
            cv = line(cv, pts, ACCENT)
        items += [(x0, y0 - 24, "style loss (log scale)", FG, SMALL),
                  (x0, y0 + h + 8, f"style {losses.get('style', 0):.4f}   content {losses.get('content', 0):.3f}",
                   DIM, SMALL)]
    return text(cv, items)


def synth_frame(spec, vmax, y, frac, it, n_iter):
    # stage 3: Griffin-Lim guesses the phase the spectrogram does not store; the waveform is drawn as it goes
    cv = canvas()
    paste(cv, spec_rgb(spec, vmax, 912, 220), 24, 110)
    paste(cv, wave_rgb(y, 912, 150, upto=frac), 24, 370)
    px = 24 + int(frac * 912)
    if frac < 1:
        cv = line(cv, [(px, 106), (px, 524)], ACCENT)
    items = [(24, 16, "3 / 3   Image -> audio", FG, FONT),
             (24, 46, "A spectrogram has no phase. Griffin-Lim guesses it by going back and forth between "
                      "waveform and spectrogram", DIM, SMALL),
             (24, 66, f"(starting from the content's phase).   iteration {it} / {n_iter}", DIM, SMALL),
             (24, 340, "output waveform", FG, SMALL)]
    return text(cv, items)


def done_frame(c_spec, s_spec, o_spec, vmax, note):
    cv = canvas()
    items = [(24, 16, "Result", FG, FONT), (24, 46, note, DIM, SMALL)]
    for k, (spec, name) in enumerate([(c_spec, "content"), (s_spec, "style"), (o_spec, "output")]):
        x0 = 24 + k * 312
        paste(cv, spec_rgb(spec, vmax, 290, 380), x0, 120)
        items.append((x0, 96, name, FG, SMALL))
    return text(cv, items)


def jpeg(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def save_video(frames, audio, sr, fps=15):
    # frames are jpeg bytes; the animation plays silently, then the last frame holds while the output plays
    import imageio.v2 as imageio
    import imageio_ffmpeg

    tmp = tempfile.mkdtemp()
    hold = int(np.ceil((len(audio) / sr + 0.5) * fps))
    with imageio.get_writer(f"{tmp}/v.mp4", fps=fps, codec="libx264", macro_block_size=4) as w:
        for f in frames + [frames[-1]] * hold:
            w.append_data(np.asarray(Image.open(io.BytesIO(f))))
    sf.write(f"{tmp}/a.wav", np.concatenate([np.zeros(int(len(frames) / fps * sr)), audio]), sr)
    out = f"{tmp}/style_transfer.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", f"{tmp}/v.mp4",
                    "-i", f"{tmp}/a.wav", "-c:v", "copy", "-c:a", "aac", "-shortest", out], check=True)
    return out
