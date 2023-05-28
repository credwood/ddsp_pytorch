import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import resample, ensure_4d, inv_ensure_4d
import torchaudio.transforms as T

class ZEncoder(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.time_steps = time_steps
    
    def expand_z(self, z):
        if len(z.shape) == 2:
            z = z[:, None, :]
        z_time_steps = int(z.shape[1])
        # TODO add other resampling methods from original ddsp core
        # this implements the hann window method
        if z_time_steps != self.time_steps:
            z = resample(z, self.time_steps)
        return z
    
    def forward(self, input):
        raise NotImplementedError

class MfccTimeDistributedRnnEncoder(ZEncoder):
    def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               z_dims=32,
               z_time_steps=250,
               mfcc_bins=13,
               sample_rate=16000,
               **kwargs):
        super().__init__(z_time_steps)
        if z_time_steps not in [63, 125, 250, 500, 1000]:
            raise ValueError(
            '`z_time_steps` currently limited to 63,125,250,500 and 1000')
        rnn_type = rnn_type.lower()
        assert rnn_type == "gru" or rnn_type == "lstm", "rnn must be either gru or lstm"
        self.z_audio_spec = {
            '63': {
                'fft_size': 2048,
                'overlap': 0.5
            },
            '125': {
                'fft_size': 1024,
                'overlap': 0.5
            },
            '250': {
                'fft_size': 1024,
                'overlap': 0.75
            },
            '500': {
                'fft_size': 512,
                'overlap': 0.75
            },
            '1000': {
                'fft_size': 256,
                'overlap': 0.75
            }
        }
        self.fft_size = self.z_audio_spec[str(z_time_steps)]['fft_size']
        self.overlap = self.z_audio_spec[str(z_time_steps)]['overlap']
        self.instance_norm = nn.InstanceNorm2d(mfcc_bins, affine=True)
        self.rnn = nn.GRU(mfcc_bins, rnn_channels, batch_first=True) if rnn_type == 'gru' else nn.LSTM(mfcc_bins, rnn_channels, batch_first=True)
        self.out = nn.Linear(rnn_channels, z_dims)
        # magenta team implemented utility methods for the mfcc calculation
        # and I've pieced together the parameters to use with the PyTorch
        # implementation, but I think their method of calculating mels slightly differs
        # using this: https://www.tensorflow.org/api_docs/python/tf/signal/linear_to_mel_weight_matrix
        self.mfcc = T.MFCC(
            sample_rate = sample_rate,
            n_mfcc=mfcc_bins,
            log_mels = True,
            melkwargs={"n_fft": self.fft_size, "n_mels": 128, "f_min": 20.0, "f_max": 8000.0, "hop_length": int(self.fft_size * (1.0 - self.overlap))},
        )
    
    def forward(self, audio):
        z = self.mfcc(audio)
        num_dims = len(z.shape)
        print(z.shape)
        z = ensure_4d(z)
        z = self.instance_norm(z)
        z = inv_ensure_4d(z, num_dims)
        z = torch.einsum("bct->btc", z)
        z, _ = self.rnn(z)
        z = self.out(z)
        print(z.shape)
        return z

