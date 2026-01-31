# voice_isolation
identify how many speakers are there, isolate every speaker even if they overlap while speaking


---

## **PROJECT ROADMAP**

### **Phase 1: Foundation (Week 1-2)**
1. **Audio Basics & Preprocessing**
   - Understanding audio signals (waveforms, sampling rates, channels)
   - Audio file I/O (loading .flac, .wav files)
   - Feature extraction: spectrograms, mel-spectrograms, STFT
   - Data augmentation for audio

2. **PyTorch Fundamentals**
   - Tensors and operations
   - Building simple neural networks
   - Training loops, loss functions, optimizers
   - GPU usage basics

### **Phase 2: Core Components (Week 3-4)**
3. **Speaker Diarization Component (EEND)**
   - Transformer encoders
   - Permutation Invariant Training (PIT)
   - Binary cross-entropy loss for multi-label classification

4. **Speech Separation Component (Conv-TasNet)**
   - 1D convolutions for time-domain processing
   - Temporal Convolutional Networks (TCN)
   - Encoder-Decoder architecture
   - SI-SDR loss function

5. **Speaker Counting Component (EDA)**
   - Attractor-based architecture
   - LSTM encoder-decoder
   - Existence probability prediction

### **Phase 3: Integration (Week 5-6)**
6. **Joint Model (EEND-SS)**
   - Shared feature learning
   - Multi-task loss function
   - Multiple 1×1 conv layers for flexible speakers
   - Fusion technique (combining diarization + separation)

### **Phase 4: Training & Evaluation (Week 7-8)**
7. **Dataset Preparation**
   - Creating synthetic mixtures (mixing clean speech)
   - Generating ground truth labels
   - Train/val/test splits

8. **Training Pipeline**
   - Data loaders
   - Training loop with multi-task loss
   - Checkpointing and logging

9. **Evaluation**
   - DER (Diarization Error Rate)
   - SI-SDR, SDR (separation metrics)
   - Speaker counting accuracy

---

## **THEORY EXPLANATION**

Let me explain the key concepts you'll need:

### **1. The Problem**
You have audio with 2-3 people talking (possibly overlapping). You need to:
- **Diarization**: Label who spoke when (timeline of Speaker A, B, C)
- **Separation**: Extract clean audio for each speaker
- **Counting**: Determine how many speakers (2 or 3?)

### **2. Why Joint Learning?**
The paper's key insight: these tasks help each other!
- If you know "who spoke when" → easier to separate overlapping speech
- If you can separate speakers → easier to identify their activities
- Training them together shares knowledge through a common feature extractor

### **3. Architecture Overview**

Think of it like this (simplified):

```
Input Audio (mixture of 2-3 speakers)
         ↓
   [SHARED ENCODER] ← learns features useful for BOTH tasks
         ↓
    ┌────┴────┐
    ↓         ↓
[DIARIZATION] [SEPARATION]
    ↓         ↓
 Timeline   Clean Audio
           (Speaker 1, 2, ...)
```

### **4. Key Components Explained**

**A. Conv-TasNet (Separation)**
- **Encoder**: Converts raw audio → learned features (like edge detectors in images)
- **Separator**: Learns to create "masks" that isolate each speaker
- **Decoder**: Converts masked features → clean audio

Think: Similar to image segmentation, but for audio in time domain!

**B. EEND (Diarization)**
- **Transformer Encoder**: Captures long-range dependencies (who's talking context)
- **Multi-label Classification**: For each time frame, predict [Speaker1: yes/no, Speaker2: yes/no, ...]
- **PIT Loss**: Handles the speaker permutation problem (Speaker A in prediction could be Speaker B in ground truth)

**C. EDA (Counting)**
- Creates "attractor vectors" for each potential speaker
- Predicts existence probability for each attractor
- Count = number of attractors with high probability

### **5. Multi-Task Learning**
Combined loss function:
```
Total Loss = λ₁(Separation Loss) + λ₂(Diarization Loss) + λ₃(Counting Loss)
```
The λ values control importance of each task.

### **6. The Fusion Trick**
After getting both outputs:
- Use diarization timeline to "mask" separated audio
- Zeros out audio when speaker isn't talking
- Reduces background noise in separated signals

---

## **SIMPLIFIED VERSION FOR YOUR PROTOTYPE**

Since this is a learning project, I'll guide you to build:

1. **Simplified Conv-TasNet** (fewer layers, smaller model)
2. **Simplified EEND** (fewer transformer layers, basic attention)
3. **Simple counting** (based on diarization outputs, not full EDA)
4. **Joint training** with multi-task loss

We'll skip some optimizations from the paper and focus on core concepts.
