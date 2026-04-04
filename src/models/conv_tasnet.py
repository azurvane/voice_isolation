from torch import nn
import torch.nn.functional as F
from .helper import _reshape


class Encoder(nn.Module):
    def __init__(
            self, 
            num_filters=512, 
            kernel_size=16, 
            stride=8,
    ):
        super().__init__()
        
        self.conv1d = nn.Conv1d(
            in_channels=1, 
            out_channels=num_filters, 
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = _reshape(x)
        x = self.conv1d(x)
        x = self.relu(x)
        return x


class TCN(nn.Module):
    def __init__(
            self, 
            in_channels=128, 
            kernel = 3,
            dilation=1
        ):
        super().__init__()
        
        self.kernel = kernel
        self.dilation = dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=self.kernel, 
            dilation=dilation, 
            groups=in_channels, 
            bias=False
            )
        self.prelu = nn.PReLU(in_channels)
        self.norm = nn.GroupNorm(1, in_channels)
        self.pointWise = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=1, 
            bias=False
            )
        
    def forward(self, x):
        residual = x
        
        pad_L = (self.kernel - 1) * self.dilation
        
        out = F.pad(x, (pad_L, 0))
        out = self.conv(out)
        out = self.prelu(out)
        out = self.norm(out)
        out = self.pointWise(out)
        return out+residual



'''
    contains three parts 
        LayerNorm 1x1 conv
        TCN blocks 
            just implemented the stacking of 8 layers to create 
            a block and stack 3 block
        1x1 conv Non Linear
    
    all three parts needs to be seperated as output of TCN blocks 
    is needed in diarization branch
'''
class Separator(nn.Module):
    def __init__(
            self,
            N: int = 512,    # encoder filters
            B: int = 128,    # bottleneck dimension
            num_blocks = 8,  # dilations per repeat [1, 2, 4, 8, 16, 32, 64, 128]
            num_repeats = 3, # how many times to repeat the 8-block stack
            num_speakers = 2 # number of masks to produce (C)
            ):
        super().__init__()
        
        self.in_channels = N
        self.Bottleneck = B
        self.dilations = [2**i for i in range(num_blocks)]
        self.num_repeats = num_repeats
        self.num_speakers = num_speakers
        
        self.groupNorm = nn.GroupNorm(1, self.in_channels)
        self.bottleneckCompression = nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.Bottleneck, 
            kernel_size=1, 
            bias = False
            )
        self.tcnStack = nn.ModuleList()
        
        for _ in range(self.num_repeats):
            for dilation in self.dilations:
                self.tcnStack.append(TCN(
                    in_channels=self.Bottleneck, 
                    dilation=dilation
                    ))
        
        self.bottleneckExpension = nn.Conv1d(
            in_channels=self.Bottleneck, 
            out_channels=self.in_channels*self.num_speakers, 
            kernel_size=1, 
            bias = False
            )
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        # LayerNorm 1x1 conv
        out = self.groupNorm(x)
        out = self.bottleneckCompression(out)
        
        # TCN blocks
        for tcn in self.tcnStack:
            out = tcn(out)
        
        # 1x1 conv Non Linear
        out = self.bottleneckExpension(out)
        out = self.activation(out)
        b, _, L = out.shape
        out = out.view(b, self.num_speakers, self.in_channels, L)
        return out
