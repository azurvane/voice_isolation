# Voice Isolation System - Project Workflow

## Project Overview
A multi-task deep learning system for speaker diarization, speech separation, and speaker counting using joint end-to-end neural networks (EEND-SS architecture).

---

## 1. Data Pipeline Workflow

```mermaid
graph TD
    A[LibriSpeech Dataset<br/>Clean Speech Recordings] --> B[Data Preprocessing]
    B --> C[Audio Loading & Resampling<br/>16kHz Sample Rate]
    C --> D[Feature Extraction]
    D --> E[Log-Mel Filterbank Features]
    D --> F[Raw Waveform]
    
    C --> G[Synthetic Mixture Generation]
    G --> H[Mix 2-3 Speakers]
    H --> I[Create Overlap<br/>0-100% Overlap Ratio]
    I --> J[Generate Ground Truth Labels]
    J --> K[Diarization Labels<br/>who spoke when]
    J --> L[Separation Labels<br/>clean source signals]
    J --> M[Speaker Count Labels<br/>number of speakers]
    
    K --> N[Training Dataset]
    L --> N
    M --> N
    E --> N
    F --> N
```

---

## 2. Model Architecture Workflow

```mermaid
graph TB
    subgraph Input
        A[Mixed Audio<br/>2-3 speakers]
    end
    
    subgraph "Shared Encoder"
        B[1D Conv Encoder<br/>Time-domain features]
        C[Layer Normalization]
        D[1×1 Convolution]
        E[Temporal Convolutional Networks<br/>TCN Blocks × 3]
        F[TCN Bottleneck Features<br/>Shared Representations]
    end
    
    subgraph "Separation Branch"
        G[Multiple 1×1 Conv Layers<br/>One per speaker count]
        H[Select Layer Based on<br/>Estimated Speaker Count]
        I[Generate Separation Masks<br/>per speaker]
        J[Element-wise Multiplication<br/>Mask × Encoder Output]
        K[1D Transposed Conv Decoder]
        L[Separated Audio Signals<br/>Speaker 1, 2, 3...]
    end
    
    subgraph "Diarization Branch"
        M[Concat: TCN Features +<br/>Log-Mel Features]
        N[Transformer Encoder<br/>4 layers, 4 attention heads]
        O[Diarization Embeddings]
        P[LSTM Encoder-Decoder<br/>Attractor Generation]
        Q[Speaker Activity Probabilities<br/>Binary per speaker per frame]
        R[Attractor Existence Probabilities<br/>Speaker Counting]
    end
    
    subgraph "Fusion & Output"
        S[Speaker Alignment<br/>Match diarization to separation]
        T[Multiply: Separated Audio ×<br/>Activity Probabilities]
        U[Refined Separated Signals]
        V[Diarization Timeline<br/>who spoke when]
        W[Speaker Count<br/>2 or 3 speakers]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    
    F --> G
    F --> M
    
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
    
    M --> N
    N --> O
    O --> P
    P --> Q
    P --> R
    
    L --> S
    Q --> S
    S --> T
    T --> U
    
    Q --> V
    R --> W
```

---

## 3. Training Workflow

```mermaid
graph TD
    A[Start Training] --> B[Load Batch of<br/>Mixed Audio + Labels]
    
    B --> C[Forward Pass]
    
    subgraph "Multi-Task Loss Calculation"
        C --> D[Separation Branch Output]
        C --> E[Diarization Branch Output]
        C --> F[Speaker Count Output]
        
        D --> G[SI-SDR Loss<br/>λ₁ = 1.0]
        E --> H[Binary Cross-Entropy Loss<br/>λ₂ = 0.2]
        F --> I[Existence Probability Loss<br/>λ₃ = 0.2]
        
        G --> J[Total Loss = λ₁×L_sep +<br/>λ₂×L_diar + λ₃×L_count]
        H --> J
        I --> J
    end
    
    J --> K[Backward Pass<br/>Compute Gradients]
    K --> L[Optimizer Step<br/>Update Weights]
    
    L --> M{Epoch Complete?}
    M -->|No| B
    M -->|Yes| N[Validation]
    
    N --> O[Calculate Metrics:<br/>DER, SI-SDRi, SCA]
    O --> P{Improvement?}
    
    P -->|Yes| Q[Save Best Model]
    P -->|No| R[Increment Counter]
    
    Q --> S{Max Epochs?}
    R --> T{Early Stop Criterion?}
    
    T -->|Yes| U[Stop Training]
    T -->|No| S
    S -->|No| B
    S -->|Yes| U
    
    U --> V[Training Complete]
```

---

## 4. Inference (2-Pass) Workflow

```mermaid
graph TD
    A[Test Audio<br/>Mixed Speech] --> B[Pass 1: Diarization]
    
    subgraph "Pass 1: Get Speaker Info"
        B --> C[Forward through<br/>Shared Encoder]
        C --> D[Forward through<br/>Diarization Branch]
        D --> E[Speaker Activity<br/>Probabilities]
        D --> F[Estimate Speaker Count Ĉ<br/>using Attractor Existence]
    end
    
    subgraph "Pass 2: Separation"
        F --> G[Select 1×1 Conv Layer<br/>corresponding to Ĉ]
        C --> G
        G --> H[Generate Ĉ Masks]
        H --> I[Apply Masks to<br/>Encoder Features]
        I --> J[Decode to Waveforms]
        J --> K[Ĉ Separated Signals]
    end
    
    subgraph "Optional Fusion"
        K --> L[Align Speakers<br/>Separation ↔ Diarization]
        E --> L
        L --> M[Multiply Signals ×<br/>Activity Probabilities]
        M --> N[Refined Separated Signals<br/>Background Noise Reduced]
    end
    
    E --> O[Output: Diarization<br/>Timeline Labels]
    N --> P[Output: Clean Audio<br/>per Speaker]
    F --> Q[Output: Speaker Count]
```

---

## 5. Evaluation Workflow

```mermaid
graph TD
    A[Test Dataset] --> B[Run Inference]
    
    subgraph "Separation Metrics"
        B --> C[Compare Separated Audio<br/>vs Ground Truth]
        C --> D[SI-SDRi<br/>Scale-Invariant SDR Improvement]
        C --> E[SDRi<br/>Source-to-Distortion Ratio]
        C --> F[STOI<br/>Short-Time Objective Intelligibility]
    end
    
    subgraph "Diarization Metrics"
        B --> G[Compare Timeline<br/>vs Ground Truth]
        G --> H[DER %<br/>Diarization Error Rate]
        H --> I[False Alarm +<br/>Missed Speech +<br/>Speaker Confusion]
    end
    
    subgraph "Speaker Counting Metrics"
        B --> J[Compare Count<br/>vs Ground Truth]
        J --> K[SCA %<br/>Speaker Counting Accuracy]
    end
    
    D --> L[Aggregate Results]
    E --> L
    F --> L
    I --> L
    K --> L
    
    L --> M[Generate Report]
    M --> N[Compare with Baselines:<br/>Conv-TasNet, EEND-EDA]
```

---

## 6. Docker Deployment Workflow

```mermaid
graph TD
    A[Development Environment] --> B[Write Code<br/>Python + PyTorch]
    
    B --> C[Create requirements.txt<br/>Pin Dependencies]
    C --> D[Write Dockerfile]
    
    subgraph "Docker Build"
        D --> E[FROM python:3.10]
        E --> F[Install System Dependencies<br/>libsndfile, ffmpeg]
        F --> G[Copy requirements.txt]
        G --> H[pip install dependencies]
        H --> I[Copy Project Code]
    end
    
    I --> J[Build Docker Image<br/>docker build -t voice-isolation:v1]
    
    subgraph "Docker Run"
        J --> K[Mount Volumes<br/>data/, src/, outputs/]
        K --> L[Run Container<br/>Isolated Environment]
    end
    
    L --> M[Execute Training/Inference<br/>Inside Container]
    
    M --> N{GPU Needed?}
    N -->|Yes| O[Rebuild with<br/>CUDA Support]
    N -->|No| P[CPU Processing]
    
    O --> Q[Production Ready<br/>Reproducible Environment]
    P --> Q
```

---

## 7. Project Directory Structure

```
voice-isolation-project/
│
├── data/
│   ├── raw/                    # LibriSpeech clean recordings
│   │   └── dev-clean/
│   │       └── [speaker-id]/
│   │           └── [chapter-id]/
│   │               └── *.flac
│   ├── processed/              # Preprocessed features
│   │   ├── features/           # Log-mel spectrograms
│   │   └── mixtures/           # Synthetic mixed audio
│   └── splits/                 # Train/val/test splits
│
├── src/
│   ├── preprocessing/
│   │   ├── audio_utils.py      # Load, resample, plot audio
│   │   ├── feature_extraction.py  # Mel-spectrograms, STFT
│   │   └── mixture_generator.py   # Create 2-3 speaker mixes
│   │
│   ├── models/
│   │   ├── conv_tasnet.py      # Separation encoder-decoder
│   │   ├── eend.py             # Diarization transformer
│   │   ├── eda.py              # Speaker counting module
│   │   └── eend_ss.py          # Joint model (main)
│   │
│   ├── training/
│   │   ├── train.py            # Training loop
│   │   ├── losses.py           # SI-SDR, BCE, PIT loss
│   │   └── optimizer.py        # Adam optimizer setup
│   │
│   └── evaluation/
│       ├── metrics.py          # DER, SI-SDRi, SCA
│       └── inference.py        # 2-pass inference
│
├── test_code/                  # Unit tests and experiments
├── outputs/                    # Saved models, logs, results
├── notebooks/                  # Jupyter notebooks for analysis
│
├── Dockerfile                  # Container specification
├── requirements.txt            # Python dependencies
├── .dockerignore              # Exclude from Docker build
└── README.md                  # Project documentation
```

---

## 8. Key Technical Components

### **Preprocessing**
- **Input:** LibriSpeech .flac files (clean speech)
- **Process:** Resample to 16kHz, extract mel-spectrograms
- **Output:** Mixed audio (2-3 speakers) + ground truth labels

### **Shared Encoder (Conv-TasNet Style)**
- 1D convolution: Raw waveform → learned features
- Temporal Convolutional Networks (TCNs): Capture temporal patterns
- Output: Bottleneck features (shared between tasks)

### **Separation Branch**
- Multiple 1×1 conv layers (one per speaker count)
- Dynamically select layer based on estimated count
- Generate masks → apply to encoder features → decode to audio

### **Diarization Branch**
- Transformer encoder: Capture long-range context
- LSTM encoder-decoder attractors: Speaker representations
- Output: Per-frame speaker activity (binary labels)

### **Speaker Counting**
- Attractor existence probabilities
- Threshold to determine number of active speakers

### **Multi-Task Training**
- Joint loss: Separation + Diarization + Counting
- Backpropagation updates all branches simultaneously
- Shared encoder learns representations useful for all tasks

### **Fusion Technique**
- Align diarization output with separated audio
- Multiply separated signals by activity probabilities
- Reduces background noise when speaker is silent

---

## 9. Simplified Prototype Approach

For this learning project, we simplify:

1. **Smaller model:** Fewer TCN layers, smaller transformer
2. **Limited speakers:** Only 2-3 speakers (not 5+)
3. **Synthetic data:** Mix clean recordings (not real overlapped audio)
4. **Basic counting:** Use diarization outputs (not full EDA module)
5. **CPU training:** Start without GPU optimization

**Goal:** Understand core concepts, not state-of-the-art performance

---

## 10. Timeline Estimate

| Phase | Tasks | Duration |
|-------|-------|----------|
| **Phase 1** | Audio preprocessing, feature extraction | Week 1-2 |
| **Phase 2** | Conv-TasNet separation model | Week 3-4 |
| **Phase 3** | EEND diarization model | Week 5-6 |
| **Phase 4** | Joint training (EEND-SS) | Week 7-8 |
| **Phase 5** | Evaluation, Docker setup, documentation | Week 9-10 |

**Total:** ~10 weeks at 5-6 hours/week

---

## References

- **Paper:** EEND-SS (Maiti et al., 2022) - arXiv:2203.17068
- **Dataset:** LibriSpeech - http://www.openslr.org/12
- **Framework:** PyTorch for deep learning
- **Deployment:** Docker for reproducibility

---

*This workflow provides a comprehensive overview of the voice isolation system, from data preparation through model deployment and evaluation.*