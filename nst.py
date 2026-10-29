# Spectrogram style transfer: audio -> log-STFT -> Gram-matrix NST -> Griffin-Lim -> audio.
import numpy as np
import torch
import torch.nn.functional as F

SR = 22050
N_FFT = 1024
HOP = 256

if torch.cuda.is_available():
    DEV = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEV = torch.device("mps")
else:
    DEV = torch.device("cpu")


def stft(y):
    y = torch.as_tensor(y, dtype=torch.float32, device=DEV)
    return torch.stft(y, N_FFT, HOP, window=torch.hann_window(N_FFT, device=DEV), return_complex=True)


def istft(S, length=None):
    return torch.istft(S, N_FFT, HOP, window=torch.hann_window(N_FFT, device=DEV), length=length)


def to_spec(y):
    S = stft(y).contiguous()
    return torch.log1p(S.abs()), torch.angle(S)


def griffin_lim(spec, phase=None, n_iter=200, length=None):
    # fast Griffin-Lim (Perraudin et al. 2013); starting from the content phase sounds much less metallic
    mag = torch.expm1(spec.clamp(min=0))
    if phase is None:
        phase = 2 * np.pi * torch.rand_like(mag)
    S = mag * torch.exp(1j * phase)
    prev = torch.zeros_like(S)
    for _ in range(n_iter):
        proj = stft(istft(S, length))
        if proj.shape[-1] != mag.shape[-1]:
            proj = proj[..., : mag.shape[-1]]
        S_new = proj + 0.99 * (proj - prev)
        prev = proj
        S = mag * torch.exp(1j * torch.angle(S_new))
    y = istft(S, length).cpu().numpy()
    return y / max(1.0, np.abs(y).max() / 0.99)


# audio view: frequency bins are channels, one 1D conv over time with random filters (Ulyanov & Lebedev)
def audio_net(n_filt=4096, k=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n_filt, N_FFT // 2 + 1, k, generator=g) * np.sqrt(2.0 / ((N_FFT // 2 + 1) * k))
    w = w.to(DEV)

    def feats(spec):
        a = F.relu(F.conv1d(spec[None], w))[0]
        return [a], a

    return feats


# image view: spectrogram as a grayscale picture through pretrained VGG-19 (Gatys layers)
_vgg = None
STYLE_LAYERS = [1, 6, 11, 20, 29]  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
CONTENT_LAYER = 22  # relu4_2


def image_net(scale):
    global _vgg
    if _vgg is None:
        from torchvision.models import vgg19, VGG19_Weights
        _vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:30].eval().to(DEV)
        for p in _vgg.parameters():
            p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=DEV)[:, None, None]

    def feats(spec):
        # flip so low frequencies are at the bottom like a normal spectrogram picture
        x = (spec.flip(0) / scale).clamp(0, 1).repeat(3, 1, 1)
        x = ((x - mean) / std)[None]
        style, content = [], None
        for i, layer in enumerate(_vgg):
            x = layer(x)
            if i in STYLE_LAYERS:
                style.append(x[0].flatten(1))
            if i == CONTENT_LAYER:
                content = x[0].flatten(1)
        return style, content

    return feats


def gram(a):
    return a @ a.T / a.shape[1]


def envelope_swap(c_spec, s_spec, smooth=15):
    # DSP baseline: divide out the content's long-term spectral envelope, multiply in the style's
    k = torch.ones(1, 1, smooth, device=DEV) / smooth
    env = lambda spec: F.conv1d(torch.expm1(spec).mean(1)[None, None], k, padding=smooth // 2)[0, 0]
    gain = (env(s_spec) + 1e-3) / (env(c_spec) + 1e-3)
    return torch.log1p(torch.expm1(c_spec) * gain[:, None])


def run(c_spec, s_spec, method="audio", init="content", s_wt=100.0, c_wt=1.0, steps=300, every=10,
        seed=0, n_filt=4096):
    """Yields (step, spec, loss_dict) every `every` LBFGS iterations. The last yield is the final spectrogram."""
    if method == "envelope":
        yield steps, envelope_swap(c_spec, s_spec), {}
        return

    torch.manual_seed(seed)
    if method == "audio":
        feats = audio_net(n_filt=n_filt, seed=seed)
    else:
        feats = image_net(scale=max(c_spec.max(), s_spec.max()).item())

    with torch.no_grad():
        s_grams = [gram(a) for a in feats(s_spec)[0]]
        c_target = feats(c_spec)[1]

    if init == "content":
        x = c_spec.clone()
    else:
        x = torch.rand_like(c_spec) * s_spec.mean() * 2
    x.requires_grad_(True)
    # LBFGS rather than Adam: Adam moves every bin by ~lr per step even where the gradient is tiny,
    # which paints a faint noise floor over the silent parts of the spectrogram
    opt = torch.optim.LBFGS([x], max_iter=every, history_size=20, line_search_fn="strong_wolfe")
    losses = {}

    def closure():
        opt.zero_grad()
        style, content = feats(x.clamp(min=0))
        # losses normalised by target size so s_wt means roughly the same thing for both views
        ls = sum(((gram(a) - g) ** 2).sum() / (g ** 2).sum() for a, g in zip(style, s_grams)) / len(style)
        lc = ((content - c_target) ** 2).sum() / (c_target ** 2).sum()
        loss = s_wt * ls + c_wt * lc
        loss.backward()
        losses.update(style=ls.item(), content=lc.item())
        return loss

    for i in range(every, steps + every, every):
        opt.step(closure)
        yield min(i, steps), x.detach().clamp(min=0), dict(losses)


def transfer(c_y, s_y, method="audio", init="content", s_wt=100.0, steps=300, gl_iter=200, **kw):
    c_spec, c_phase = to_spec(c_y)
    s_spec, _ = to_spec(s_y)
    for _, spec, losses in run(c_spec, s_spec, method, init, s_wt, steps=steps, **kw):
        pass
    y = griffin_lim(spec, c_phase, n_iter=gl_iter, length=len(c_y))
    return y, spec.cpu().numpy(), losses
