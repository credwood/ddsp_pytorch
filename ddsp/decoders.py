import torch
import torch.nn as nn
from .core import mlp, gru
from effortless_config import Config

class DecoderConfig(Config):
    hidden_size=512
    mlp_ch=512
    layers_per_stack=3
    n_harmonics=100
    n_bands=65
    n_amp=1


class RnnFcDecoder(nn.Module):
    def __init__(self, hidden_size=512, mlp_ch=512, encoder_out=True, z_dims=16, 
                 layers_per_stack=3, n_harmonics=100, n_bands=65, n_amp=1):
        super().__init__()
        
        self.encoder_out = encoder_out
        if encoder_out:
            self.mlp_in = nn.ModuleList([
                mlp(1,mlp_ch, layers_per_stack),
                mlp(1,mlp_ch, layers_per_stack),
                mlp(z_dims,mlp_ch, layers_per_stack)]
            )
            self.rnn = gru(3, hidden_size)
            self.mlp_out = mlp(hidden_size*4, hidden_size, layers_per_stack)
            self.decoders = nn.ModuleList([
                nn.Linear(hidden_size, n_harmonics+n_amp), 
                nn.Linear(hidden_size, n_bands)]
            )
        else:
            self.mlp_in = nn.ModuleList([
                mlp(1,mlp_ch, layers_per_stack),
                mlp(1,mlp_ch, layers_per_stack),]
            )
            self.rnn = gru(2, hidden_size)
            self.mlp_out = mlp(hidden_size*3, hidden_size, layers_per_stack)
            self.decoders = nn.ModuleList([
                nn.Linear(hidden_size, n_harmonics+n_amp), 
                nn.Linear(hidden_size, n_bands)]
            )

    def forward(self, pitch, loudness, z):
        """
        z: latent z, shape [batch_size, seq_len, mcff_num]
        p: pitch, shape [batch_size, seq_len, 1]
        l: loudness, shape [batch_size, seq_len, 1]
        """
        if z is not None and self.encoder_out:
            inputs = torch.cat([
                self.mlp_in[0](pitch),
                self.mlp_in[1](loudness),
                self.mlp_in[2](z),
            ], -1)
        else:
            inputs = torch.cat([
                self.mlp_in[0](pitch),
                self.mlp_in[1](loudness),
            ], -1)
        hidden = self.rnn(inputs)[0]
        hidden = torch.cat([inputs, hidden], -1)
        hidden = self.mlp_out(hidden)
        return self.decoders[0](hidden), self.decoders[1](hidden)
    