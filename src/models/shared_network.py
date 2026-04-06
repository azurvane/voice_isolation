from torch import nn
import torch.nn.functional as F


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
        x = self.conv1d(x)
        x = self.relu(x)
        return x


class LayerNormConv1x1(nn.Module):
    def __init__(
            self,
            N: int = 512,    # encoder filters
            B: int = 128,    # bottleneck dimension
            ):
        super().__init__()
        
        self.in_channels = N
        self.Bottleneck = B
        
        self.groupNorm = nn.GroupNorm(1, self.in_channels)
        self.bottleneckCompression = nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.Bottleneck, 
            kernel_size=1, 
            bias = False
            )
    
    def forward(self, x):
        out = self.groupNorm(x)
        out = self.bottleneckCompression(out)
        
        return out


class _TCNBlock(nn.Module):
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


class TCNStack(nn.Module):
    def __init__(
            self,
            N: int = 512,    # encoder filters
            B: int = 128,    # bottleneck dimension
            num_blocks = 8,  # dilations per repeat [1, 2, 4, 8, 16, 32, 64, 128]
            num_repeats = 3, # how many times to repeat the 8-block stack
            ):
        super().__init__()
        
        self.in_channels = N
        self.Bottleneck = B
        self.dilations = [2**i for i in range(num_blocks)]
        self.num_repeats = num_repeats
        
        self.tcnStack = nn.ModuleList()
        
        for _ in range(self.num_repeats):
            for dilation in self.dilations:
                self.tcnStack.append(_TCNBlock(
                    in_channels=self.Bottleneck, 
                    dilation=dilation
                    ))
    
    def forward(self, x):
        out = x
        for tcn in self.tcnStack:
            out = tcn(out)
        
        return out