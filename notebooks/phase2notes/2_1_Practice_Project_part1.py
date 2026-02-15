import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.audio_utils import load_audio

# Device configuration (for your MacBook)
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print("✅ Using Apple Silicon GPU")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ Using NVIDIA GPU")
else:
    device = torch.device('cpu')
    print("⚠️ Using CPU")

print(f"Device: {device}")



def pad_or_trim_audio(audio_tensor: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Pad with zeros or trim audio to target length.
    
    Args:
        audio_tensor: 1D tensor
        target_length: Desired length
    
    Returns:
        Tensor of length target_length
    """
    current_length = audio_tensor.shape[0]
    
    if current_length < target_length:
        # Pad
        padding = target_length - current_length
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
    elif current_length > target_length:
        # Trim
        audio_tensor = audio_tensor[:target_length]
    
    return audio_tensor


# ============================================
# PRACTICE PROJECT: Audio Batch Loader
# ============================================

def load_audio_batch(
    audio_files: List[Path],
    sr: int = 16000,
    target_length: int = 48000,  # 3 seconds at 16kHz
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load multiple audio files and prepare them as a batch for training.
    
    Args:
        audio_files: List of paths to audio files
        target_length: Target length in samples (pad/trim to this)
        device: Device to move tensors to ('cpu', 'cuda', 'mps')
    
    Returns:
        audio_batch: Tensor of shape (batch_size, 1, target_length)
        lengths: Original lengths before padding (for masking later)
    
    YOUR TASK:
    Implement this function step by step.
    """
    
    # TODO 1: Initialize empty lists to store tensors and lengths
    audio_tensors = []
    original_lengths = []
    
    # TODO 2: Load each audio file
    # For each file:
    #   - Load audio using load_audio()
    #   - Convert to tensor
    #   - Store original length
    #   - Pad or trim to target_length
    #   - Reshape to (1, 1, target_length)
    #   - Append to audio_tensors list
    
    # YOUR CODE HERE
    for file in audio_files:
        audio, _ = load_audio(file_path=file, sr_=sr)
        audio_tensor = torch.tensor(audio)
        original_lengths.append(audio.size)
        audio_pad_trim = pad_or_trim_audio(audio_tensor, target_length=target_length)
        audio_reshape = audio_pad_trim.view(1,1,-1)
        audio_tensors.append(audio_reshape)
    
    # TODO 3: Stack all tensors into a batch
    # Hint: Use torch.cat() along dim=0
    
    batch = torch.cat(tensors=audio_tensors, dim=0)
    
    # TODO 4: Create lengths tensor
    # Hint: Convert original_lengths list to tensor
    
    lengths_tensor = torch.tensor(original_lengths)
    
    # TODO 5: Move to device
    
    batch = batch.to(device)
    lengths_tensor = lengths_tensor.to(device=device)
    
    return batch, lengths_tensor


# Test the batch loader

# Get some audio files
data_dir = Path.cwd() / "data" / "raw"
audio_files = list(data_dir.rglob("*.flac"))[:5]  # Take 5 files

print(data_dir)

print(f"Loading {len(audio_files)} audio files...")

# Load batch
batch, lengths = load_audio_batch(
    audio_files=audio_files,
    target_length=48000,  # 3 seconds
    device=device
)

print("\n✅ BATCH CREATED!")
print(f"Batch shape: {batch.shape}")
print(f"Expected: (5, 1, 48000)")
print(f"\nOriginal lengths: {lengths}")
print(f"Device: {batch.device}")

# Verify
assert batch.shape == (5, 1, 48000), "Shape should be (batch, channels, samples)"
assert batch.device.type == device.type, f"Should be on {device}"
assert len(lengths) == 5, "Should have 5 lengths"