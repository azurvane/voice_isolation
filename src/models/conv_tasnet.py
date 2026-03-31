from torch import nn

def reshape(batch):
    if batch.dim() == 2:
        return batch.unsqueeze(1)
    return batch


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
        x = reshape(x)
        x = self.conv1d(x)
        x = self.relu(x)
        return x