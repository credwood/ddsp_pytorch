import torch
import torch.nn as nn
from .core import scale_function, remove_above_nyquist, upsample, midi_to_hz
from .core import harmonic_synth, amp_to_impulse_response, fft_convolve, pitch_ss_loss
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
                 block_size=256, n_midi=128, pitch_encoder=None,
                 autoencoder=Autoencoder
                 ):
        super().__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))

        # for real-time inference
        self.register_buffer("cache_gru", torch.zeros(1, 1, hidden_size))
        self.register_buffer("phase", torch.zeros(1))

        if pitch_encoder == ResNetAutoencoder:
            self.pitch_encoder = ResNetAutoencoder(**dict(ResNetEncoderConfig))
            self.pitch_sigmoid = nn.Sigmoid()
        else:
            self.pitch_encoder = None
        self.autoencoder = autoencoder()
        self.reverb = Reverb(sampling_rate, sampling_rate)
        
        
        
    def forward(self, s, pitch=None, loudness=None):
        if isinstance(self.pitch_encoder, ResNetAutoencoder):
            raise NotImplementedError("training for ResNet not implemented")
        else:
            assert pitch is not None, "must pass f0 value"

        multi=False
        amp_param, noise_param = self.autoencoder(pitch, loudness, s)

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
