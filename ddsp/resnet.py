
"""
Resnet adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
import torch
from typing import Union, List
import torch.nn as nn
import torch.nn.functional as F
from .core import safe_log
import torchaudio.transforms as T
from effortless_config import Config
from einops import rearrange


class ResNetEncoderConfig(Config):
    time_steps = 250
    n_mels = 229
    sample_rate = 16000
    pitch = 128
    amplitude = 100+1
    noise_mag = 60
    size = 'small'
    
# ------------------ ResNet ----------------------------------------------------

class LayerNorm(nn.LayerNorm):
    """
    Moves Channel dim to the back to apply LayerNorm
    from: https://github.com/facebookresearch/encodec/blob/main/encodec/modules/norm.py
    """
    def __init__(self, norm_shape: Union[int, List[int], torch.Size], **kwargs):
        super().__init__(norm_shape, **kwargs)
    
    def forward(self, x):
        x = torch.einsum("bm...t->bt...m", x)
        x = super().forward(x)
        x = torch.einsum("bt...m->bm...t", x)
        return x


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    #pad_in = calc_same_pad(in_planes, 3, stride, dilation)
    #pad_out = calc_same_pad(out_planes, 3, stride, dilation)
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        padding="same",
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    #pad_in = calc_same_pad(in_planes, 3, stride)
    #pad_out = calc_same_pad(out_planes, 3, stride)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, padding="same", stride=1, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = LayerNorm,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual = False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = LayerNorm,
        time_steps=250,
        n_mels=128
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        #pad_in = calc_same_pad(n_mels, 7, 2)
        #pad_out = calc_same_pad(time_steps, 7, 2)
        self.conv1 = nn.Conv2d(n_mels, self.inplanes, kernel_size=7, stride=1, padding="same", bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 1024, 3, stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride = 1,
        dilate = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)



class ResNetAutoencoder(nn.Module):
    "ResNet Autoencoder that maps from audio to synthesizer parameters"
    def __init__(self,
                 time_steps=250,
                 frequencies=128,
                 size='small',
                 n_mels=229,
                 factor_downsample=4,
                 **kwargs
                 ):
        super().__init__()
        self.spectral_fn = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            n_mels=n_mels,
            hop_length=int(1024 * (1.0 - 0.75)),
            f_min=20.0,
            f_max=8000.0, 
        )
        size_dict = {
        'small': (32, [2, 3, 4]),
        'medium': (32, [3, 4, 6]),
        'large': (64, [3, 4, 6]),
        }
        self.size = size
        self.time_steps = time_steps
        ch, num_layers = size_dict[size]
        self.resnet = ResNet(Bottleneck, num_layers, time_steps=time_steps, n_mels=n_mels)
        self.out = nn.Linear(1024*factor_downsample, frequencies)
        
    def forward(self, audio):
        assert len(audio.shape) == 2, "audio must have shape batch_size, samples"
        
        mels = self.spectral_fn(audio)
        mels = safe_log(mels)
        mels = mels[:, :, :self.time_steps]
        mels = mels[:, :, :, None]
        mels = rearrange(mels, "b m t c -> b m c t")
        x = self.resnet(mels)
        x = rearrange(x, "b m c t -> b t (c m)")
        
        return self.out(x)
        