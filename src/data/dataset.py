# -------------------------------------------------------------------------------
#   add a function for padding and truncating 
#
#   what to do if speakers in metadata is greater then max_speaker
# -------------------------------------------------------------------------------

import json
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio


# constrain 
SAMPLE_RATE = 16_000
MAX_DURATION_SEC = 30
MAX_SPEAKERS = 3
LABEL_FRAME_SIZE_MS = 10


class EENDSSDataset(Dataset):
    def __init__(
            self, 
            manifest_path: str | Path, 
            sample_rate: int = SAMPLE_RATE,
            max_duration_sec: int = MAX_DURATION_SEC,
            max_speakers: int = MAX_SPEAKERS,
            label_frame_size_ms: int = LABEL_FRAME_SIZE_MS,
            limit: int | None = None
            ):
        
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.max_speakers = max_speakers
        self.label_frame_size_ms = label_frame_size_ms
        
        self.max_sample = self.sample_rate * self.max_duration_sec
        self.label_frame_sample = self.sample_rate *self.label_frame_size_ms // 1000
        self.max_label_frames = self.max_sample // self.label_frame_sample
        
        with open(manifest_path, "r") as f:
            self.data = json.load(f)
        
        if limit is not None:
            self.data = self.data[:limit]
        
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        meta = self.data[idx]
        
        # load mixture audio
        mixture, sr = torchaudio.load(meta["mixture_path"])
        mixture = mixture.squeeze()
        
        mask_length = mixture.size(0)
        
        if mixture.size(0) < self.max_sample:
            padding_count = self.max_sample - mixture.size(0)
            mixture = F.pad(mixture, (0, padding_count), "constant", 0)
        elif mixture.size(0) > self.max_sample:
            mixture = mixture[:self.max_sample]
        
        # load clean audio
        source_list = []
        for path in meta["source_paths"]:
            clean, _ = torchaudio.load(path)
            clean = clean.squeeze()
            
            if clean.size(0) < self.max_sample:
                padding_count = self.max_sample - clean.size(0)
                clean = F.pad(clean, (0, padding_count), "constant", 0)
            elif clean.size(0) > self.max_sample:
                clean = clean[:self.max_sample]
            
            source_list.append(clean)
        
        source = torch.stack(source_list)
        if source.size(0) < self.max_speakers:
            padding_count = self.max_speakers - source.size(0)
            source = F.pad(source, (0, 0, 0, padding_count), "constant", 0)
        
        # load diarization label 
        labels_np = np.load(meta["label_path"])
        labels = torch.from_numpy(labels_np).float()
        
        if labels.shape[0] < self.max_label_frames:
            padding_size = self.max_label_frames - labels.shape[0]
            labels = F.pad(labels, (0, 0, 0, padding_size), "constant", 0)
        elif labels.shape[0] > self.max_label_frames:
            labels = labels[:self.max_label_frames, :]
        
        if labels.shape[-1] < self.max_speakers:
            padding_size = self.max_speakers - labels.shape[-1]
            labels = F.pad(labels, (0, padding_size), "constant", 0)
        
        # load number of speaker
        num_speakers = meta["num_speakers"]
        
        # creating the mask
        mask = torch.arange(self.max_sample) < min(mask_length, self.max_sample)
        
        output = {
            "mixture"     : mixture,
            "source"      : source,
            "labels"      : labels,
            "num_speakers": num_speakers,
            "mask"        : mask
        }
        
        return output

