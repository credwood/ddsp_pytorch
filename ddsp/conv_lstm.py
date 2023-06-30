"""
Experimental model taking the Macaron-Net/Conformer
as insipration.
Conformer elemets adapted from:
https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py#L16
"""
import torch
from typing import Union, List
import torch.nn as nn
import torch.nn.functional as F
from .core import safe_log
import torchaudio.transforms as T
from effortless_config import Config
from einops import rearrange

#|--------------Helper Classes--------------|

class LayerNorm(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(dim)
    
    def forward(self, x):
        x = torch.einsum("bm...t->bt...m", x)
        x = super().forward(x)
        x = torch.einsum("bt...m->bm...t", x)
        return x

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(x)

class GLU(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


#|--------------LSTM--------------|

class LSTMConfig(Config):
    dim = 128
    num_layers = 3
    p_dropout = 0.0

class LSTM(nn.Module):
    def __init__(self, dim=128, num_layers=3, p_dropout=0.0):
        super().__init__()

        self.dim = dim
        self.lstm = nn.LSTM(dim, dim, num_layers=num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=p_dropout)
    
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x, _ = self.lstm(x)
        x = residual + self.dropout(x)
        return x

#|--------------MLP--------------|
class MLPConfig(Config):
    dim_in = 128
    hidden = 128 * 4
    dropout = 0.0

class MLP(nn.Module):
    def __init__(self, dim_in, hidden, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, hidden),
            Swish(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, dim_in),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        return x + self.mlp(x)

#|--------------Convolution Block--------------|

def padd_conv(kernel_size):
    pad = kernel_size//2
    return (pad, pad-(kernel_size+1)%2)

class ConvConfig(Config):
    dim_in = 128
    causal = True
    expansion_factor = 2
    kernel_size = 31
    dropout=0.
    
class DepthWiseConv1D(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(ch_in, ch_out, kernel_size, groups=ch_in)
    
    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, dim_in=128, causal = True, expansion_factor = 2, kernel_size = 31, dropout=0.):
        super().__init__()
        inner_dim = dim_in * expansion_factor
        padding = padd_conv(kernel_size) if not causal else (kernel_size-1, 0)

        self.block = nn.Sequential(
            LayerNorm(dim_in),
            nn.Conv1d(dim_in, inner_dim*2, 1),
            GLU(dim=1),
            DepthWiseConv1D(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding),
            LayerNorm(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim_in, 1),

        )
    
    def forward(self, x):
        return x + self.block(x)


#|--------------Model--------------|

class ConvLSTM(nn.Module):
    def __init__(self, out_dim =128, conv_config=ConvConfig, lstm_config=LSTMConfig, mlp_config=MLPConfig):
        super().__init__()
        self.lstm_confg = dict(lstm_config)
        self.conv_config = dict(conv_config)
        self.mlp_config = dict(mlp_config)
        assert self.lstm_confg["dim"] == self.conv_config["dim_in"] == self.mlp_config["dim_in"]
        
        self.lstm = LSTM(**(self.lstm_confg))
        self.conv_block = ConvBlock(**(self.conv_config))
        self.mlp = MLP(**(self.mlp_config))

        self.n_mels = self.lstm_confg["dim"]
        self.spectral_fn = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            n_mels=self.n_mels,
            hop_length=int(1024 * (1.0 - 0.75)),
            f_min=20.0,
            f_max=8000.0, 
        )

        dim = self.mlp_config["dim_in"]
        self.layer_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)
    
    def forward(self, x):
        x = self.spectral_fn(x)
        x = safe_log(x)
        x = torch.einsum("bmt->btm", x)
        x = self.lstm(x)
        x = torch.einsum("btm->bmt", x)
        x = self.conv_block(x)
        x = torch.einsum("bmt->btm", x)
        x = self.mlp(x)

        return self.out(self.layer_norm(x))
