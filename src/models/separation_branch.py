from torch import nn

class NonLinearConv1x1(nn.Module):
    def __init__(
            self,
            N: int = 512,    # encoder filters
            B: int = 128,    # bottleneck dimension
            num_speakers = 2 # number of masks to produce (C)
            ):
        super().__init__()
        
        self.in_channels = N
        self.Bottleneck = B
        self.num_speakers = num_speakers
        
        self.bottleneckExpension = nn.Conv1d(
            in_channels=self.Bottleneck, 
            out_channels=self.in_channels*self.num_speakers, 
            kernel_size=1, 
            bias = False
            )
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        out = self.bottleneckExpension(x)
        out = self.activation(out)
        b, _, L = out.shape
        out = out.view(b, self.num_speakers, self.in_channels, L)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            num_filters=512, 
            kernel_size=16, 
            stride=8
            ):
        super().__init__()
        
        self.convTrans = nn.ConvTranspose1d(
            in_channels=num_filters, 
            out_channels=1, 
            kernel_size=kernel_size,
            stride=stride,
            bias=False
            )
    
    def forward(self, x):
        return self.convTrans(x)