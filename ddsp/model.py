import torch
import torch.nn as nn
from .core import scale_function, remove_above_nyquist, upsample, normalize_from_midi
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve
from .core import resample
from .encoders import MfccTimeDistributedRnnEncoder, EncoderConfig
from .resnet import ResNetAutoencoder, ResNetEncoderConfig
from .decoders import RnnFcDecoder, DecoderConfig
from einops import rearrange


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
    
    def forward(self, p, l, s):
        if self.encoder is not None:
            z = self.encoder(s)
            return self.decoder(p, l, z)
        else:
            return self.decoder(p, l)


class DDSP(nn.Module):
    def __init__(self, hidden_size=512, sampling_rate=16000,
                 block_size=256, n_midi=128, autoencoder=ResNetAutoencoder
                 ):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # for real-time inference
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        if autoencoder == ResNetAutoencoder:
            self.autoencoder = ResNetAutoencoder(**dict(ResNetEncoderConfig))
        else:
            self.autoencoder = autoencoder()
        self.reverb = Reverb(sampling_rate, sampling_rate)
        self.sigmoid = nn.Sigmoid()
    
        
    def forward(self, s, pitch=None, loudness=None, top_k_pitches=True):
        if isinstance(self.autoencoder, ResNetAutoencoder):
            pitch, amp_param, noise_param = self.autoencoder(s)
            #amp_param = rearrange(amp_param, "b t (a p) -> b t p a", p=pitch.shape[-1])
            #multi=True
            pitch_dist = self.sigmoid(pitch)
            pitch_mask = torch.where(pitch_dist >= 0.5, 1, 0)
            pitch = normalize_from_midi(pitch)
            # their method takes the expected value as f0
            # have tried multiple unsupervised methods
            # will implement the self-supervised method with synthetic
            # data that the magenta team uses
            pitch = (pitch*pitch_mask).sum(dim=-1).unsqueeze(-1)
            #if top_k_pitches:
                #_, top_k = torch.topk(pitch_dist, k=3, sorted=False)
               # pitch = pitch[:, :, top_k[-1][-1]]
                #amp_param = amp_param[:, :, top_k[-1][-1]]

        else:
            amp_param, noise_param = self.autoencoder(pitch, loudness, s)
        multi=False

        # harmonic part
        param = scale_function(amp_param)
        total_amp = param[..., :1]
        amplitudes = param[..., 1:]
        
        amplitudes = remove_above_nyquist(
            amplitudes,
            pitch,
            self.sampling_rate,
            multi=multi, 
        )
        amplitudes /= amplitudes.sum(-1, keepdim=True)
        amplitudes = amplitudes*total_amp

        amplitudes = upsample(amplitudes, self.block_size)
        pitch = upsample(pitch, self.block_size)

        harmonic = harmonic_synth(pitch, amplitudes, self.sampling_rate, multi=multi)

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

class DDSPInv(nn.Module):
    pass