"""
Many functions adapted from: https://github.com/magenta/ddsp/blob/761d61a5f13373c170f3a04d54bb28a1e2f06bab/ddsp/core.py
and more generally: https://github.com/magenta/ddsp.git 
"""

import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import librosa as li
import crepe
import math

from einops import rearrange

def ensure_4d_resnet(x):
  """Add extra dimensions to make sure tensor has height and width."""
  if len(x.shape) == 2:
      return x[:, None, None, :]
  elif len(x.shape) == 3:
      return x[:, :, None, :]
  else:
      return x

def inv_ensure_4d_resnet(x, n_dims):
  """Remove excess dims, inverse of ensure_4d() function."""
  if n_dims == 2:
      return x[:, 0, 0, :]
  if n_dims == 3:
      return x[:, :, 0, :]
  else:
      return x

def inv_ensure_4d(x, n_dims):
  """Remove excess dims, inverse of ensure_4d() function."""
  if n_dims == 2:
    return x[:, 0, 0, :]
  if n_dims == 3:
    return x[:, :, 0, :]
  else:
    return x

def ensure_4d(x):
    """Add extra dimensions to make sure tensor has height and width."""
    if len(x.shape) == 2:
      return x[:, :, None, None]
    elif len(x.shape) == 3:
      return x[:, :, None, :]
    else:
      return x

def inv_ensure_4d(x, n_dims):
    """Remove excess dims, inverse of ensure_4d() function."""
    if n_dims == 2:
      return x[:, :, 0, 0]
    if n_dims == 3:
      return x[:, :, 0, :]
    else:
      return x

def safe_log(x):
    return torch.log(x + 1e-7)

def calc_same_pad(i, k, s, d=1):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

@torch.no_grad()
def mean_std_loudness(dataset):
    mean = 0
    std = 0
    n = 0
    for _, _, l in dataset:
        n += 1
        mean += (l.mean().item() - mean) / n
        std += (l.std().item() - std) / n
    return mean, std


def multiscale_fft(signal, scales, overlap):
    stfts = []
    for s, o in zip(scales, overlap):
        S = torch.stft(
            signal,
            s,
            int(s * (1 - o)),
            s,
            torch.hann_window(s).to(signal),
            True,
            normalized=True,
            return_complex=True,
        ).abs()
        stfts.append(S)
    return stfts


def resample(x, factor: int):
    batch, frame, channel = x.shape
    x = x.permute(0, 2, 1).reshape(batch * channel, 1, frame)

    window = torch.hann_window(
        factor * 2,
        dtype=x.dtype,
        device=x.device,
    ).reshape(1, 1, -1)
    y = torch.zeros(x.shape[0], x.shape[1], factor * x.shape[2]).to(x)
    y[..., ::factor] = x
    y[..., -1:] = x[..., -1:]
    y = torch.nn.functional.pad(y, [factor, factor])
    y = torch.nn.functional.conv1d(y, window)[..., :-1]

    y = y.reshape(batch, channel, factor * frame).permute(0, 2, 1)

    return y


def upsample(signal, factor):
    if len(signal.shape) == 3:
        signal = signal.permute(0, 2, 1)
        signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor, mode="linear")
        return signal.permute(0, 2, 1)
    
    assert len(signal.shape) == 4, "signal must have 3 or 4 dims"

    signal = torch.einsum("btpa->bpat", signal)
    stack = []
    for num in range(signal.shape[0]):
       interp = nn.functional.interpolate(signal[num], size=signal.shape[-1] * factor, mode="linear")
       stack.append(interp)
    
    stack = torch.stack(stack)
    return torch.einsum("bpat->btpa", stack)
    

def remove_above_nyquist(amplitudes, pitch, sampling_rate, multi=False):
    if multi:
       return remove_above_nyquist_multi(amplitudes, pitch, sampling_rate)
    n_harm = amplitudes.shape[-1]
    pitches = pitch.repeat(1,1, n_harm).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa

def remove_above_nyquist_multi(amplitudes, pitch, sampling_rate):
    """
    aplitudes: tensor, shape [b t p a]
    pitch: tensor, shape [b t p]
    sampling_rate: int
    """
    aa = (pitch < sampling_rate / 2).float() + 1e-4
    aa = aa.unsqueeze(-1)
    aa = aa.repeat(1, 1, 1, amplitudes.shape[-1]).to(pitch)
    return aa * amplitudes


def scale_function(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7


def extract_loudness(signal, sampling_rate, block_size, n_fft=2048):
    S = li.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
    )
    S = np.log(abs(S) + 1e-7)
    f = li.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weight = li.A_weighting(f)

    S = S + a_weight.reshape(-1, 1)

    S = np.mean(S, 0)[..., :-1]

    return S


def extract_pitch(signal, sampling_rate, block_size):
    length = signal.shape[-1] // block_size
    f0 = crepe.predict(
        signal,
        sampling_rate,
        step_size=int(1000 * block_size / sampling_rate),
        verbose=1,
        center=True,
        viterbi=True,
    )
    f0 = f0[1].reshape(-1)[:-1]

    if f0.shape[-1] != length:
        f0 = np.interp(
            np.linspace(0, 1, length, endpoint=False),
            np.linspace(0, 1, f0.shape[-1], endpoint=False),
            f0,
        )

    return f0


def mlp(in_size, hidden_size, n_layers):
    channels = [in_size] + (n_layers) * [hidden_size]
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(channels[i], channels[i + 1]))
        net.append(nn.LayerNorm(channels[i + 1]))
        net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


def pitch_ss_loss(predicted, true_pitch):
   top_k = predicted.shape[-1]
   true_pitch = true_pitch.repeat(1, 1, top_k)
   vals, _ = torch.abs(predicted-true_pitch).min(dim=-1)
   vals = vals.mean()
   return vals

def gru(n_input, hidden_size):
    return nn.GRU(n_input * hidden_size, hidden_size, batch_first=True)

def unit_to_midi(unit, midi_min=0.0, midi_max=127.0):
  return midi_min + (midi_max - midi_min) * unit

def midi_to_hz(t):
   midi = torch.arange(0, t.shape[-1]).to(t)
   hz =  440.0*(2.0**((midi-69.0)/12.0))
   hz = hz.repeat(t.shape[0], t.shape[1], 1)
   return hz
   

def harmonic_synth(pitch, amplitudes, sampling_rate, multi=False):
    if multi:
       return harmonic_synth_multi(pitch, amplitudes, sampling_rate)
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal

def harmonic_synth_multi(pitch, amplitudes, sampling_rate):
    """
    pitch: tensor, shape [b upsampled_signal p]
    amplitudes: tensor, shape [b upsampled_signal p a]
    """
    omega = pitch * (2.0 * np.pi)
    omega = omega/float(sampling_rate)
    omega = torch.cumsum(omega, 1)
    omega = omega.repeat(1, 1, amplitudes.shape[-1])
    omega = rearrange(omega, "b s (p a) -> b s p a", a=amplitudes.shape[-1])
    omega = omega * torch.arange(1, amplitudes.shape[-1] + 1).to(omega)
    omega = torch.sin(omega)
    signal = torch.einsum("bspa->bs", omega*amplitudes)
    return signal.unsqueeze(-1)


def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output
