import torch
import torch.nn as nn
from .core import mlp, gru, scale_function, remove_above_nyquist, upsample
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve
from .core import resample
from .encoders import MfccTimeDistributedRnnEncoder, EncoderConfig
from .decoders import RnnFcDecoder, DecoderConfig
import math


class Reverb(nn.Module):
    def __init__(self, length, sampling_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sampling_rate = sampling_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sampling_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x

class Autoencoder(nn.Module):
    def __init__(self, encoder=MfccTimeDistributedRnnEncoder, encoder_config=EncoderConfig, 
                 decoder=RnnFcDecoder, decoder_config=DecoderConfig):
        super().__init__()
        if encoder is not None:
            encoder = encoder(**dict(encoder_config))   
        self.encoder = encoder
        self.decoder = decoder(**dict(decoder_config))
    
    def forward(self, p, l, s, encoder_out=True):
        if self.encoder is not None and encoder_out:
            z = self.encoder(s)
            return self.decoder(p, l, z)
        else:
            return self.decoder(p, l)


class DDSP(nn.Module):
    def __init__(self, hidden_size, sampling_rate,
                 block_size, autoencoder=Autoencoder
                 ):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # for real-time inference
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        self.autoencoder = autoencoder()
        self.reverb = Reverb(sampling_rate, sampling_rate)

        
    def forward(self, s, pitch, loudness, encoder_out=True):
        amp_param, noise_param = self.autoencoder(pitch, loudness, s, encoder_out=encoder_out)

        # harmonic part
        param = scale_function(amp_param)
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate)

        # noise part
        param = scale_function(noise_param - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        #reverb part
        signal = self.reverb(signal)

        return signal

    def realtime_forward(self, pitch, loudness):
        """
        TODO: update Decoder class to allow for caching
         hidden = torch.cat([
            self.in_mlps[0](pitch),
            self.in_mlps[1](loudness),
        ], -1)

        gru_out, cache = self.gru(hidden, self.cache_gru)
        self.cache_gru.copy_(cache)

        hidden = torch.cat([gru_out, pitch, loudness], -1)
        hidden = self.out_mlp(hidden)

        # harmonic part
        param = scale_function(self.proj_matrices[0](hidden))

        total_amp = param[..., :1]
        amplitudes = param[..., 1:]

        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes *= total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        n_harmonic = amplitudes.shape[-1]
        omega = torch.cumsum(2 * math.pi * pitch / self.sampling_rate, 1)

        omega = omega + self.phase
        self.phase.copy_(omega[0, -1, 0] % (2 * math.pi))

        omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)

        harmonic = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)

        # noise part
        param = scale_function(self.proj_matrices[1](hidden) - 5)

        impulse = amp_to_impulse_response(param, self.block_size)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
        ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        signal = harmonic + noise

        return signal
        """
        pass
       