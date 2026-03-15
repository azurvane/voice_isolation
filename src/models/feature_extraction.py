import torch
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from pathlib import Path
import os
import sys
import json
import matplotlib.pyplot as plt




def LogMelFeatures(audio, n_fft, hop_length):
    torch_lmf = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=80
        )
    mel_spec = torch_lmf(audio)
    log = torch.log(mel_spec + 1e-8)
    return log

