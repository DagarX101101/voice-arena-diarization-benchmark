# Josh Talks Voice Arena: Speaker Diarization Benchmark
**Executive Briefing & Comparative Evaluation** | *Date: October 2026* | *Corpus: Josh Talks Voice Arena*

---

### 1. Objective & Evaluation Framework
The **Josh Talks Voice Arena Diarization Benchmark** evaluates 10 speaker diarization systems across enterprise speech workflows: **Call Centre Recordings** (8 kHz telephony), **Real-Time Transcriptions** (dense conversational overlap), and **Office Meetings** (presentation & multi-party dialogue). Evaluations use a strict **0.25s collar**, evaluate overlap (`skip_overlap = False`), and apply **Hungarian optimal bipartite matching**.

Standard DER weights Missed Speech, False Alarms, and Speaker Confusion identically. Because business costs diverge by use case, systems are ranked by **Custom Score** ($100 \times \max(0, 1 - \text{Category-Weighted Error})$):
- **Call Recordings ($w$: Miss 0.40 / FA 0.25 / Conf 0.35):** Heavy confusion penalty protects agent/customer attribution and compliance auditing.
- **Real-Time Transcription ($w$: Miss 0.50 / FA 0.20 / Conf 0.30):** High miss penalty ensures downstream ASR captures all spoken words.
- **Office Meetings ($w$: Miss 0.45 / FA 0.15 / Conf 0.40):** High confusion penalty preserves action-item ownership in meeting summaries.

---

### 2. Category Performance & Leaderboard Summary
The benchmark provides comprehensive multi-domain evaluation across all 10 systems:

| Operational Category | Category Champion | Custom Score | DER % | Leading Pipeline Architecture | Key Trade-off / Characteristic |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Call Centre / Telephony** | **Model 10: BUT-FIT VBx** | **80.31** | 68.80% | Variational Bayes HMM + x-vector | Bayesian smoothing suppresses telephony confusion; Model 3 achieves 79.26. |
| **Real-Time Transcription** | **Model 7: ESPnet EEND-EDA** | **94.14** | 20.50% | Encoder-Decoder Attractor Network | Multi-speaker attractors resolve overlapping turns; Model 2 achieves 93.11. |
| **Office Room Meetings** | **Model 4: NeMo Streaming** | **99.83** | 1.12% | Causal Rolling Context Sortformer | Near-flawless tracking (0.0% Miss, 0.0% Confusion, 1.12% DER) on CPU. |
| **General / All-Purpose** | **Model 2: NeMo Sortformer** | **92.12** | 24.31% | Conformer-based Self-Attention | Highest macro stability across clean dialogue and degraded audio channels. |

---

### 3. Key Findings Across Architecture Families
1. **Streaming Sortformer Achieves Offline Parity:** **Model 4 (NeMo Streaming Sortformer)** achieves virtually identical performance to offline models (**99.83** vs **99.75** in meetings; **93.07** vs **93.11** in live audio) with minimal latency (~7.65s CPU runtime), making it the premier choice for live production speech pipelines.
2. **Clustering Resists Telephony Degradation:** On 8 kHz band-pass audio (`call_input01`), **Model 10 (BUT-FIT VBx)** and **Model 3 (NeMo Clustering Diarizer)** significantly outperform standard segmentation, achieving **2.02%–2.10% confusion** versus pyannote's **27.81% confusion**.
3. **VAD Sensitivity vs. False Alarm Trade-off:** **Model 1 (pyannote Community-1)** exhibits aggressive VAD, capturing faint turns (lowest call miss: 6.30%), but suffers elevated false alarms (63.04% on calls, 13.19% on live audio).

---

### 4. Data Quality & Limitations: `call_input02`
Ground truth annotations for `call_input02.flac` span timestamps **400.35s–479.77s**, whereas the physical recording is **120.00s**. Within the valid audio window, reference speech is 0.00s ($0/0$ division). To protect statistical validity without altering original ground truth files, `call_input02` is officially classified as `INVALID_REFERENCE_DURATION` and excluded from primary scoring. A $-400.0\text{s}$ offset recovers valid alignment, confirming models behave predictably.

---

### 5. Architectural Paradigms Evaluated
The benchmark evaluates systems across four major technical paradigms:
- **Self-Attention Sortformers (Models 2, 4):** Direct transformer-based speaker sorting resolving multi-speaker turns without heuristic clustering.
- **Modular Neural Clustering (Models 1, 3, 5, 6, 8):** Multi-stage pipelines separating acoustic VAD, deep embedding extraction (TitaNet, ECAPA, ResNet, x-vector), and graph spectral or agglomerative clustering.
- **End-to-End Multi-Speaker (Model 7):** Joint multi-label neural networks predicting overlapping speaker tracks simultaneously.
- **Probabilistic State Models (Models 9, 10):** Hidden Markov Models with Variational Bayes or recurrent state transitions.

---

### 6. Production Recommendations
- **For Live ASR & Streaming Subtitles:** Deploy **NVIDIA NeMo Streaming Sortformer (Model 4)** for instant turn detection without session re-clustering.
- **For Customer Call Audits & Contact Centers:** Deploy **NVIDIA NeMo Clustering Diarizer (Model 3)** or **BUT-FIT VBx (Model 10)** for superior speaker attribution in band-limited telephony.
- **For High Overlap Broadcasts:** Deploy **NeMo Sortformer (Model 2)** or **ESPnet EEND-EDA (Model 7)** to resolve simultaneous speakers.
