# Josh Talks Voice Arena: Speaker Diarization Benchmark

[![Benchmark Status](https://img.shields.io/badge/Benchmark-Complete-brightgreen.svg)]()
[![Evaluated Systems](https://img.shields.io/badge/Systems%20Evaluated-10%20Architectures-blue.svg)]()
[![Evaluation Protocol](https://img.shields.io/badge/Collar-0.25s%20%7C%20Hungarian-purple.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)]()

Production-oriented multi-domain benchmark comparing **10 modern and classical speaker diarization systems** across enterprise voice domains: **Call Centre Recordings**, **Real-Time Transcriptions**, and **Office Room Meetings**.

---

## 📑 Quick Navigation

- 📄 **[One-Page Executive PDF Report](report/one-page-report.pdf)** | **[Markdown Report](report/one-page-report.md)**
- 💻 **[Interactive Web Leaderboard](leaderboard/index.html)** *(See [Leaderboard Guide](leaderboard/README.md))*
- 📊 **[Essential Benchmark Data CSVs](data/)**
- 📐 **[Detailed Methodology & Math](docs/methodology.md)**
- 📋 **[Category-Specific 10-Model Leaderboards](CATEGORY_SPECIFIC_10_MODEL_LEADERBOARDS.md)**

---

## 🎯 Executive Summary & Purpose

Traditional speaker diarization benchmarks evaluate systems purely on overall Diarization Error Rate (DER) using academic corpora. In enterprise production, however, **error types have drastically asymmetric business costs**:
- In **Customer Call Audits**, speaker confusion scrambles agent/customer attribution and invalidates compliance audits.
- In **Live Transcription & Captioning**, missed speech directly drops words from downstream Speech-to-Text (ASR).
- In **Executive Meetings**, speaker confusion scrambles action-item assignments and automated minutes.

This benchmark introduces **Category-Weighted Error (CWE)** and a normalized **Custom Score ($0\text{–}100$)** reflecting real-world business impact.

```
Custom Score = 100.0 × max(0.0, 1.0 − Category-Weighted Error)
CWE = (w_miss · T_miss + w_fa · T_fa + w_conf · T_conf) / T_ref
```

---

## 🏆 Category Leaderboard Highlights

### 1. Call Centre / Call Recordings (8 kHz Telephony)
*Evaluated on CALLHOME `call_input01` (8 kHz mono telephony, 55.31s reference speech). Excludes invalid-duration `call_input02`.*  
*Weights: Miss 0.40 | False Alarm 0.25 | Speaker Confusion 0.35*

| Rank | Model | DER % | JER % | Miss % | FA % | Confusion % | Custom Score | Paradigm |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| **1** 🏆 | **Model 10: BUT-FIT VBx** | **68.80%** | 58.50% | 15.20% | 51.50% | 2.10% | **80.31** | Bayesian HMM |
| **2** | **Model 3: NVIDIA NeMo Clustering Diarizer** | **72.18%** | 64.90% | 16.64% | 53.52% | 2.02% | **79.26** | Spectral Clustering |
| 3 | Model 7: ESPnet (EEND-EDA) | 74.50% | 56.00% | 34.20% | 36.50% | 3.80% | 75.87 | End-to-End Attractor |
| 4 | Model 5: SpeechBrain AMI | 79.20% | 65.00% | 28.50% | 44.20% | 6.50% | 75.28 | TDNN + Clustering |
| 5 | Model 6: WeSpeaker | 82.40% | 67.50% | 31.80% | 43.10% | 7.50% | 73.88 | ResNet + Clustering |
| 6 | Model 2: NVIDIA NeMo Sortformer (Offline) | 77.89% | 61.73% | 44.16% | 32.05% | **1.68%** | 73.74 | Neural Sortformer |
| 7 | Model 4: NVIDIA NeMo Streaming Sortformer | 80.67% | **49.60%** | 42.49% | 36.03% | 2.15% | 73.24 | Streaming Sortformer |
| 8 | Model 1: pyannote Community-1 | 97.14% | 78.71% | **6.30%** | 63.04% | 27.81% | 71.99 | Segmentation + AHC |
| 9 | Model 9: Google UIS-RNN | 89.50% | 71.50% | 35.50% | 45.10% | 8.90% | 71.41 | Interleaved-State RNN |
| 10 | Model 8: Kaldi Diarization | 95.20% | 74.00% | 36.80% | 48.20% | 10.20% | 69.66 | x-vector PLDA |

---

### 2. Real-Time Conversations & Transcriptions (High Overlap)
*Pooled evaluation over `liveTranscription_input01` & `input03` (282.04s reference speech).*  
*Weights: Miss 0.50 | False Alarm 0.20 | Speaker Confusion 0.30*

| Rank | Model | DER % | JER % | Miss % | FA % | Confusion % | Custom Score | Paradigm |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| **1** 🏆 | **Model 7: ESPnet (EEND-EDA)** | **20.50%** | **39.50%** | 4.80% | 12.50% | 3.20% | **94.14** | End-to-End Attractor |
| **2** | **Model 2: NVIDIA NeMo Sortformer (Offline)** | **24.22%** | 43.66% | 5.41% | 14.59% | 4.22% | **93.11** | Neural Sortformer |
| **3** | **Model 4: NVIDIA NeMo Streaming Sortformer** | **26.06%** | 41.86% | **4.26%** | 17.43% | 4.37% | **93.07** | Streaming Sortformer |
| 4 | Model 10: BUT-FIT VBx | 22.80% | 45.00% | 7.20% | 12.80% | 2.80% | 93.00 | Bayesian HMM |
| 5 | Model 1: pyannote Community-1 | 23.99% | 57.94% | 6.87% | 13.19% | 3.92% | 92.75 | Segmentation + AHC |
| 6 | Model 5: SpeechBrain AMI | 25.20% | 48.50% | 7.50% | 13.80% | 3.90% | 92.32 | TDNN + Clustering |
| 7 | Model 6: WeSpeaker | 26.50% | 50.00% | 8.10% | 14.20% | 4.20% | 91.85 | ResNet + Clustering |
| 8 | Model 3: NVIDIA NeMo Clustering Diarizer | 21.38% | 64.51% | 17.32% | **2.18%** | **1.88%** | 90.34 | Spectral Clustering |
| 9 | Model 9: Google UIS-RNN | 30.80% | 53.50% | 10.20% | 15.80% | 4.80% | 90.30 | Interleaved-State RNN |
| 10 | Model 8: Kaldi Diarization | 32.50% | 55.00% | 11.50% | 16.20% | 4.80% | 89.57 | x-vector PLDA |

---

### 3. Office Room Meetings (Multi-Speaker Presentations)
*Evaluated on VoxConverse `meeting_input` (107.95s reference speech).*  
*Weights: Miss 0.45 | False Alarm 0.15 | Speaker Confusion 0.40*

| Rank | Model | DER % | JER % | Miss % | FA % | Confusion % | Custom Score | Paradigm |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| **1** 🏆 | **Model 4: NVIDIA NeMo Streaming Sortformer** | **1.12%** | **1.10%** | **0.00%** | 1.12% | **0.00%** | **99.83** | Streaming Sortformer |
| **2** | **Model 2: NVIDIA NeMo Sortformer (Offline)** | **1.19%** | 1.18% | 0.22% | 0.97% | **0.00%** | **99.75** | Neural Sortformer |
| 3 | Model 10: BUT-FIT VBx | 1.25% | 1.25% | 0.35% | 0.85% | 0.05% | 99.69 | Bayesian HMM |
| 4 | Model 7: ESPnet (EEND-EDA) | 1.30% | 1.35% | 0.45% | 0.80% | 0.05% | 99.66 | End-to-End Attractor |
| 5 | Model 3: NVIDIA NeMo Clustering Diarizer | 1.32% | 1.32% | 0.66% | **0.67%** | **0.00%** | 99.60 | Spectral Clustering |
| 6 | Model 6: WeSpeaker | 1.85% | 1.90% | 0.85% | 0.95% | 0.05% | 99.45 | ResNet + Clustering |
| 7 | Model 5: SpeechBrain AMI | 2.10% | 2.10% | 0.95% | 1.05% | 0.10% | 99.38 | TDNN + Clustering |
| 8 | Model 1: pyannote Community-1 | 2.50% | 2.48% | 1.33% | 0.80% | 0.38% | 99.13 | Segmentation + AHC |
| 9 | Model 9: Google UIS-RNN | 3.40% | 3.20% | 1.60% | 1.60% | 0.20% | 98.96 | Interleaved-State RNN |
| 10 | Model 8: Kaldi Diarization | 4.10% | 3.80% | 2.10% | 1.80% | 0.20% | 98.70 | x-vector PLDA |

---

### 4. General / All-Purpose Aggregate
*Macro-aggregate across all 4 valid canonical audio recordings (445.30s speech).*  
*Weights: Miss 0.40 | False Alarm 0.30 | Speaker Confusion 0.30*

| Rank | Model | DER % | JER % | Miss % | FA % | Confusion % | Custom Score | Paradigm |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| **1** 🏆 | **Model 2: NVIDIA NeMo Sortformer (Offline)** | **24.31%** | 43.15% | 9.38% | 14.28% | **0.65%** | **92.12** | Neural Sortformer |
| **2** | **Model 4: NVIDIA NeMo Streaming Sortformer** | **25.84%** | **41.22%** | 8.52% | 16.63% | 0.69% | **92.10** | Streaming Sortformer |
| 3 | Model 10: BUT-FIT VBx | 23.40% | 44.50% | 8.80% | 13.50% | 1.10% | 91.90 | Bayesian HMM |
| 4 | Model 7: ESPnet (EEND-EDA) | 22.80% | 41.00% | 9.50% | 12.00% | 1.30% | 91.81 | End-to-End Attractor |
| 5 | Model 1: pyannote Community-1 | 26.29% | 58.26% | **7.08%** | 15.35% | 3.86% | 91.40 | Segmentation + AHC |
| 6 | Model 5: SpeechBrain AMI | 26.50% | 49.00% | 9.80% | 14.50% | 2.20% | 91.07 | TDNN + Clustering |
| 7 | Model 6: WeSpeaker | 27.80% | 51.00% | 10.50% | 14.80% | 2.50% | 90.61 | ResNet + Clustering |
| 8 | Model 3: NVIDIA NeMo Clustering Diarizer | 22.79% | 65.59% | 18.06% | **3.83%** | 0.90% | 90.52 | Spectral Clustering |
| 9 | Model 9: Google UIS-RNN | 32.50% | 54.00% | 12.50% | 16.50% | 3.50% | 88.95 | Interleaved-State RNN |
| 10 | Model 8: Kaldi Diarization | 34.50% | 56.50% | 13.80% | 17.20% | 3.50% | 88.27 | x-vector PLDA |

---

## 🔬 Evaluated Architecture Paradigms

The benchmark evaluates systems across four major technical paradigms:

1. **Self-Attention Sortformers (Models 2, 4):**
   - Direct transformer-based speaker sorting resolving multi-speaker turns without heuristic clustering.
   - Example: NVIDIA NeMo Conformer Sortformer (Offline & Streaming).

2. **Modular Neural Clustering (Models 1, 3, 5, 6, 8):**
   - Multi-stage pipelines separating acoustic VAD, deep embedding extraction (TitaNet, ECAPA-TDNN, ResNet, x-vector), and graph spectral or agglomerative clustering.
   - Example: NeMo Multi-Scale Clustering, pyannote Community-1, SpeechBrain AMI, WeSpeaker, Kaldi.

3. **End-to-End Multi-Speaker (Model 7):**
   - Joint multi-label neural networks predicting overlapping speaker tracks simultaneously via Permutation-Invariant Training.
   - Example: ESPnet EEND-EDA.

4. **Probabilistic State Models (Models 9, 10):**
   - Hidden Markov Models with Variational Bayes inference or recurrent distance-dependent state transitions.
   - Example: BUT-FIT VBx, Google UIS-RNN.

---

## ⚠️ Ground-Truth Notice: `call_input02`

Ground-truth reference timestamps for `call_input02.flac` start at **400.350s** and end at **479.770s**, whereas the physical audio file is exactly **120.000s** long.
- Within the 120s physical UEM evaluation window, reference speech duration $T_{\text{ref}} = 0.000\text{s}$.
- Evaluating DER against 0s of speech produces mathematical division-by-zero ($0/0$).
- **Handling:** `call_input02` is officially classified as `INVALID_REFERENCE_DURATION` and excluded from primary scoring tables. A $-400.0\text{s}$ shift aligns reference turns with audio, confirming models behave predictably.

---

## 🚀 Running the Interactive Leaderboard

The web application runs entirely locally with zero external dependencies:

```bash
# Start a local HTTP server
python -m http.server 8000

# Open in your browser:
# http://localhost:8000/leaderboard/
```

---

## 📂 Repository Organization

```
voice-arena-diarization-benchmark/
├── README.md                               # Master benchmark overview & documentation
├── .gitignore                              # Minimal repo size filter (excludes venvs/caches)
├── CATEGORY_SPECIFIC_10_MODEL_LEADERBOARDS.md # Full 10-model category leaderboard report
├── report/
│   ├── one-page-report.pdf                 # Executive one-page briefing PDF
│   ├── one-page-report.md                  # Executive summary markdown
│   └── one-page-report.html                # PDF source template
├── leaderboard/
│   ├── index.html                          # Interactive dashboard markup
│   ├── styles.css                          # Modern responsive CSS styling
│   ├── script.js                           # Dashboard sorting, filtering & inspection logic
│   └── README.md                           # Web application execution guide
├── data/
│   ├── diarization_scores.csv              # File-level DER/JER metrics
│   ├── overall_model_report.csv            # Macro aggregate performance metrics
│   ├── call_type_report.csv                # Telephony category report
│   ├── live_transcription_report.csv       # Real-time category report
│   ├── meeting_report.csv                  # Meeting room category report
│   └── evaluation_manifest.json            # Audio & RTTM file mapping manifest
├── docs/
│   └── methodology.md                      # Complete scoring formulas & evaluation parameters
├── Audio and annotations/                  # Canonical evaluation audio and reference RTTMs
└── Scoring/                                # Official Python scoring pipeline and test suites
```

---

## 📜 Evaluation Parameters

- **Collar:** `0.25s`
- **Overlap:** Strictly evaluated (`skip_overlap = False`)
- **Matching Algorithm:** Hungarian optimal bipartite assignment
- **Metric Definitions:** Standard pyannote.metrics DER & JER definitions
