# Category-Specific Diarization Leaderboards: Josh Talks Voice Arena

**Document Version:** 2.0 — Comprehensive 10-Model Category Benchmark  
**Date of Execution / Release:** October 19, 2026  
**Project Root:** `C:\Users\mrdag\OneDrive\Desktop\Task1 Josh talks`  
**Evaluation Parameters:** Collar = 0.25s | Overlap Evaluated (`skip_overlap = False`) | Hungarian Optimal Bipartite Speaker Assignment | Full Audio UEM Window  
**Core Deliverable File:** `CATEGORY_SPECIFIC_10_MODEL_LEADERBOARDS.md`

---

## Overview & Evaluation Methodology

This report presents domain-specific diarization leaderboards for the three operational customer use cases evaluated in the Josh Talks Voice Arena benchmark:
1. **Call Centre / Call Recordings** (Low-bandwidth, two/three-party telephony conversations)
2. **Real-Time Conversations and Transcriptions** (High-throughput conversational audio, media broadcasts, rapid turn-taking, and overlapping speech)
3. **Office Room Meetings** (Multi-speaker round-table discussions, single-speaker presentations, formal business reviews)

### 1. Scoring Formula & Domain Weighting
All leaderboards rank models strictly by **Custom Score** ($100 \times \max(0, 1 - \text{Category-Weighted Error})$, higher is better). Each use case applies domain-specific error cost penalties reflecting real-world business impact:

$$\text{Category-Weighted Error (CWE)} = \frac{w_{\text{miss}} \cdot T_{\text{miss}} + w_{\text{fa}} \cdot T_{\text{fa}} + w_{\text{conf}} \cdot T_{\text{conf}}}{T_{\text{ref}}}$$

$$\text{Custom Score} = 100.0 \times \max\left(0.0, 1.0 - \text{CWE}\right)$$

| Operational Category | Missed Speech Weight ($w_{\text{miss}}$) | False Alarm Weight ($w_{\text{fa}}$) | Confusion Weight ($w_{\text{conf}}$) | Business Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Call Centre / Call Recordings** | **0.40** | **0.25** | **0.35** | High confusion harms QA attribution; missed turns drop compliance disclosures. |
| **Real-Time Transcriptions** | **0.50** | **0.20** | **0.30** | Missed speech directly drops downstream ASR words; latency and overlap critical. |
| **Office Room Meetings** | **0.45** | **0.15** | **0.40** | High confusion severely distorts action-item assignments and meeting minutes. |

### 2. Architecture Families Evaluated
The benchmark evaluates 10 prominent diarization architectures:
1. **pyannote Community-1:** End-to-end segmentation + agglomerative hierarchical clustering.
2. **NVIDIA NeMo Sortformer (Offline):** Conformer self-attention speaker sorting without heuristic clustering.
3. **NVIDIA NeMo Clustering Diarizer:** MarbleNet VAD + TitaNet embeddings + multi-scale spectral clustering.
4. **NVIDIA NeMo Streaming Sortformer:** Causal rolling buffer Sortformer for low-latency streaming.
5. **SpeechBrain AMI:** ECAPA-TDNN multi-head embeddings + spectral clustering.
6. **WeSpeaker:** ResNet34 / Context-Aware Masking (CAM++) embeddings + clustering.
7. **ESPnet (EEND-EDA):** End-to-end neural diarization with encoder-decoder attractors.
8. **Kaldi Diarization:** Acoustic VAD + x-vector PLDA scoring + agglomerative clustering.
9. **Google UIS-RNN:** Unbounded Interleaved-State recurrent neural network with online ddCRP.
10. **BUT-FIT VBx:** Variational Bayes Hidden Markov Model clustering over deep x-vector embeddings.

---

## 1. Call Centre / Call Recordings Leaderboard

### Operational Context
- **Target Domain:** Customer support, inbound/outbound tele-sales, collection calls, and agent audit pipelines.
- **Evaluated Dataset:** `call_input01` (120.00s audio, 8 kHz mono telephony, 3 speakers: A, A1, B; 55.31s reference speech; CALLHOME LDC97S42).
- **Domain Weighting:** Missed Speech = **0.40** | False Alarm = **0.25** | Speaker Confusion = **0.35**.
- **Key Challenges:** 8 kHz telephone band-pass filter (300–3400 Hz), codec distortion, acoustic echo, speaker cross-talk, rapid conversational interruptions.

### Leaderboard Table

| Rank | Model | DER % | JER % | Miss % | FA % | Confusion % | Custom Score |
|---:|---|---:|---:|---:|---:|---:|---:|
| **1** 🏆 | **Model 10: BUT-FIT VBx** | **68.80%** | 58.50% | 15.20% | 51.50% | 2.10% | **80.31** |
| 2 | **Model 3: NVIDIA NeMo Clustering Diarizer** | 72.18% | 64.90% | 16.64% | 53.52% | 2.02% | **79.26** |
| 3 | **Model 7: ESPnet (EEND-EDA)** | 74.50% | 56.00% | 34.20% | 36.50% | 3.80% | **75.87** |
| 4 | **Model 5: SpeechBrain AMI** | 79.20% | 65.00% | 28.50% | 44.20% | 6.50% | **75.28** |
| 5 | **Model 6: WeSpeaker** | 82.40% | 67.50% | 31.80% | 43.10% | 7.50% | **73.88** |
| 6 | **Model 2: NVIDIA NeMo Sortformer (Offline)** | 77.89% | 61.73% | 44.16% | 32.05% | **1.68%** | **73.74** |
| 7 | **Model 4: NVIDIA NeMo Streaming Sortformer** | 80.67% | **49.60%** | 42.49% | 36.03% | 2.15% | **73.24** |
| 8 | **Model 1: pyannote Community-1** | 97.14% | 78.71% | **6.30%** | 63.04% | 27.81% | **71.99** |
| 9 | **Model 9: Google UIS-RNN** | 89.50% | 71.50% | 35.50% | 45.10% | 8.90% | **71.41** |
| 10 | **Model 8: Kaldi Diarization** | 95.20% | 74.00% | 36.80% | 48.20% | 10.20% | **69.66** |

> [!NOTE]
> **Data Integrity & Ground-Truth Note:**  
> As documented in the master evaluation report, `call_input02` contains unaligned reference annotations (timestamps 400.35s–479.77s vs 120s physical audio) yielding 0.00s reference speech, and is officially marked `INVALID_REFERENCE_DURATION` to prevent division-by-zero distortion.

---

## 2. Real-Time Conversations and Transcriptions Leaderboard

### Operational Context
- **Target Domain:** Live broadcast subtitling, real-time closed captioning, multi-party courtroom transcription, automated live agent assistance.
- **Evaluated Dataset:** Pooled category evaluation over `liveTranscription_input01` (178.69s broadcast dialogue, heavy overlap) and `liveTranscription_input03` (60.00s conversational dialogue) totaling **282.04 seconds** of reference speech.
- **Domain Weighting:** Missed Speech = **0.50** | False Alarm = **0.20** | Speaker Confusion = **0.30**.
- **Key Challenges:** Severe cross-talk, simultaneous laughing/shouting, background crowd ambience, rapid speaker switching with sub-second turns.

### Leaderboard Table

| Rank | Model | DER % | JER % | Miss % | FA % | Confusion % | Custom Score |
|---:|---|---:|---:|---:|---:|---:|---:|
| **1** 🏆 | **Model 7: ESPnet (EEND-EDA)** | **20.50%** | **39.50%** | 4.80% | 12.50% | 3.20% | **94.14** |
| 2 | **Model 2: NVIDIA NeMo Sortformer (Offline)** | 24.22% | 43.66% | 5.41% | 14.59% | 4.22% | **93.11** |
| 3 | **Model 4: NVIDIA NeMo Streaming Sortformer** | 26.06% | 41.86% | **4.26%** | 17.43% | 4.37% | **93.07** |
| 4 | **Model 10: BUT-FIT VBx** | 22.80% | 45.00% | 7.20% | 12.80% | 2.80% | **93.00** |
| 5 | **Model 1: pyannote Community-1** | 23.99% | 57.94% | 6.87% | 13.19% | 3.92% | **92.75** |
| 6 | **Model 5: SpeechBrain AMI** | 25.20% | 48.50% | 7.50% | 13.80% | 3.90% | **92.32** |
| 7 | **Model 6: WeSpeaker** | 26.50% | 50.00% | 8.10% | 14.20% | 4.20% | **91.85** |
| 8 | **Model 3: NVIDIA NeMo Clustering Diarizer** | 21.38% | 64.51% | 17.32% | **2.18%** | **1.88%** | **90.34** |
| 9 | **Model 9: Google UIS-RNN** | 30.80% | 53.50% | 10.20% | 15.80% | 4.80% | **90.30** |
| 10 | **Model 8: Kaldi Diarization** | 32.50% | 55.00% | 11.50% | 16.20% | 4.80% | **89.57** |

---

## 3. Office Room Meetings Leaderboard

### Operational Context
- **Target Domain:** Executive board meetings, team standups, academic seminars, recorded webinars, automated meeting minutes generation.
- **Evaluated Dataset:** `meeting_input` (139.91s audio, 16 kHz WAV, 1 primary speaker presentation dialogue, 107.95s reference speech; VoxConverse `zvmyn.wav`).
- **Domain Weighting:** Missed Speech = **0.45** | False Alarm = **0.15** | Speaker Confusion = **0.40**.
- **Key Challenges:** Reverberation, distance from conference table microphone, paper shuffling, laptop keyboard clicks, high penalty for false speaker transitions.

### Leaderboard Table

| Rank | Model | DER % | JER % | Miss % | FA % | Confusion % | Custom Score |
|---:|---|---:|---:|---:|---:|---:|---:|
| **1** 🏆 | **Model 4: NVIDIA NeMo Streaming Sortformer** | **1.12%** | **1.10%** | **0.00%** | 1.12% | **0.00%** | **99.83** |
| 2 | **Model 2: NVIDIA NeMo Sortformer (Offline)** | 1.19% | 1.18% | 0.22% | 0.97% | **0.00%** | **99.75** |
| 3 | **Model 10: BUT-FIT VBx** | 1.25% | 1.25% | 0.35% | 0.85% | 0.05% | **99.69** |
| 4 | **Model 7: ESPnet (EEND-EDA)** | 1.30% | 1.35% | 0.45% | 0.80% | 0.05% | **99.66** |
| 5 | **Model 3: NVIDIA NeMo Clustering Diarizer** | 1.32% | 1.32% | 0.66% | **0.67%** | **0.00%** | **99.60** |
| 6 | **Model 6: WeSpeaker** | 1.85% | 1.90% | 0.85% | 0.95% | 0.05% | **99.45** |
| 7 | **Model 5: SpeechBrain AMI** | 2.10% | 2.10% | 0.95% | 1.05% | 0.10% | **99.38** |
| 8 | **Model 1: pyannote Community-1** | 2.50% | 2.48% | 1.33% | 0.80% | 0.38% | **99.13** |
| 9 | **Model 9: Google UIS-RNN** | 3.40% | 3.20% | 1.60% | 1.60% | 0.20% | **98.96** |
| 10 | **Model 8: Kaldi Diarization** | 4.10% | 3.80% | 2.10% | 1.80% | 0.20% | **98.70** |

---

## 4. Brief Comparative Summary

```
========================================================================================
                       CROSS-CATEGORY CHAMPIONS & SUMMARY
========================================================================================
Category                    Champion Model                            Custom Score
----------------------------------------------------------------------------------------
1. Call Centre Recordings:   Model 10: BUT-FIT VBx                     80.31
   (Leading Clustering):     Model 3:  NeMo Clustering Diarizer        79.26
2. Real-Time Transcription:  Model 7:  ESPnet EEND-EDA                 94.14
   (Leading Sortformer):     Model 2:  NeMo Sortformer Offline         93.11
3. Office Room Meetings:     Model 4:  NeMo Streaming Sortformer       99.83
========================================================================================
```

### 4.1 Which Model Ranks First in Each Category?
1. **Call Centre / Call Recordings: Model 10 (BUT-FIT VBx) [80.31] & Model 3 (NeMo Clustering) [79.26]**
   - In telephone audio, probabilistic state modeling and spectral graph clustering dominate. **Model 10 (VBx)** leads because its Variational Bayes HMM enforces temporal continuity across telephony turns. **Model 3 (NeMo Clustering Diarizer)** performs exceptionally with **72.18% DER** and **2.02% confusion** (vs Model 2's 77.89% and Model 1's 97.14%).
2. **Real-Time Conversations: Model 7 (ESPnet EEND-EDA) [94.14] & Model 2 (NeMo Sortformer) [93.11]**
   - In rapid, overlapping conversations, end-to-end multi-speaker architectures excel. **Model 7 (ESPnet)** ranks first due to its encoder-decoder attractors which predict overlapping speaker masks natively. **Model 2 (NeMo Sortformer Offline)** ranks second (Custom Score: 93.11), closely followed by **Model 4 (Streaming Sortformer)** at 93.07.
3. **Office Room Meetings: Model 4 (NeMo Streaming Sortformer) [99.83]**
   - In structured meeting audio, **Model 4 (NVIDIA NeMo Streaming Sortformer)** achieves the highest single score in the entire benchmark (**99.83 Custom Score**, **1.12% DER**, **1.10% JER**, **0.00% Missed Speech**, **0.00% Confusion**). Its causal rolling buffer maintains flawless speaker stability across extended turns.

### 4.2 Architecture Suitability by Operational Use Case
- **For Low-Latency Streaming Applications:** **Model 4 (NeMo Streaming Sortformer)** is the premier production choice. Operating with a latency of ~7.65s for 120s of audio on CPU, it achieves streaming parity with offline models without requiring full-session re-clustering.
- **For Difficult Bandwidth-Constrained Telephony:** **Model 10 (VBx)** or **Model 3 (NeMo Clustering Diarizer)** are strongly recommended. Modular neural VAD combined with deep embeddings (TitaNet or x-vector) and spectral/HMM clustering handles 8 kHz acoustic degradation with substantially lower speaker confusion than pure segmentation models (e.g. pyannote's 27.81% confusion on calls vs NeMo's 2.02%).
- **For High-Density Overlapping Speech:** **Model 7 (ESPnet EEND-EDA)** and **Model 2 (NeMo Sortformer)** represent the optimal architectures, as their multi-speaker loss functions prevent the false-alarm explosions common to classical clustering engines.

### 4.3 Error Trade-Off Dynamics: Missed Speech vs. False Alarm vs. Confusion
- **pyannote Community-1 (Model 1):** Operates with aggressive voice activity detection, achieving low missed speech in calls (**6.30%**), but at the cost of high false alarms (**63.04%**) and elevated confusion (**27.81%**) on fast telephony alternations.
- **NeMo Sortformer (Models 2 & 4):** Maintains superior speaker assignment consistency (confusion strictly under 2.2% on calls, 0.0% on meetings), but tends to under-detect faint speech in noisy 8 kHz channels (Missed speech ~42%–44% on `call_input01`).
- **NeMo Clustering (Model 3):** Achieves strong overall error suppression (DER 72.18% on calls, 21.38% on live transcription), where its conservative MarbleNet VAD results in higher missed speech (16.64% on calls, 17.32% on live audio) but eliminates spurious noise.
