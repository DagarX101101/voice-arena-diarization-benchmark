# Josh Talks Voice Arena: Diarization Benchmark Methodology

## 1. Overview and Objective

The objective of the Josh Talks Voice Arena Speaker Diarization Benchmark is to evaluate modern neural and classical speaker diarization systems across real-world enterprise speech domains. The benchmark provides actionable engineering guidance for production deployment by evaluating systems not merely on generic research averages, but on category-specific performance constraints.

---

## 2. Evaluation Datasets & Use Cases

The benchmark evaluates four primary operational categories:

| Category | Canonical Audio | Duration | Speakers | Speech Duration | Audio Characteristics |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Call Centre / Call Recordings** | `call_input01.sph` | 120.00s | 3 | 55.31s | 8 kHz telephony, band-pass filtered (300–3400 Hz), conversational cross-talk (CALLHOME LDC97S42). |
| **Real-Time Conversations & Transcriptions** | `liveTranscription_input01.wav`<br>`liveTranscription_input03.wav` | 178.69s<br>60.00s | 4<br>2 | 222.04s<br>60.00s (282.04s total) | 16 kHz broadcast audio, high speaker density, rapid overlapping turns, ambient acoustic noise. |
| **Office Room Meetings** | `meeting_input.wav` | 139.91s | 1 | 107.95s | 16 kHz presentation dialogue, room reverberation, extended single-speaker turns (VoxConverse `zvmyn.wav`). |
| **General / All-Purpose** | Aggregate of all 4 valid canonical files | 498.60s | 1–4 | 445.30s | Macro cross-domain performance across clean and noisy channels. |

### Data Quality & Ground-Truth Handling: `call_input02`
- **Issue:** The ground truth annotation for `call_input02.flac` spans from timestamp **400.350s to 479.770s**, whereas the physical audio file is exactly **120.000s** long.
- **Impact:** Within the 120s physical UEM evaluation window, reference speech duration $T_{\text{ref}} = 0.000\text{s}$. Calculating Diarization Error Rate without reference speech results in mathematical division-by-zero ($0/0$).
- **Resolution:** `call_input02` is officially classified as `INVALID_REFERENCE_DURATION` and excluded from official primary benchmark tables to preserve absolute scoring integrity. Diagnostic alignment sweeps proved that applying a $-400.0\text{s}$ timestamp offset aligns the reference turns directly with the audio, but the raw reference is kept unaltered in official reporting.

---

## 3. Metrics and Mathematical Formulas

All evaluations are conducted with a **0.25-second collar** around speaker turns, with **overlapping speech strictly evaluated** (`skip_overlap = False`) using the **Hungarian optimal bipartite matching algorithm** for reference-to-hypothesis speaker label mapping.

### 3.1 Diarization Error Rate (DER)
$$\text{DER} = \frac{T_{\text{miss}} + T_{\text{fa}} + T_{\text{conf}}}{T_{\text{ref}}}$$
- **Missed Speech ($T_{\text{miss}}$):** Reference speech duration not assigned to any hypothesis speaker.
- **False Alarm ($T_{\text{fa}}$):** Hypothesis speech duration declared where no reference speaker was active.
- **Speaker Confusion ($T_{\text{conf}}$):** Reference speech assigned to an incorrect hypothesis speaker.
- **Reference Speech Duration ($T_{\text{ref}}$):** Total ground truth active speech duration.

### 3.2 Jaccard Error Rate (JER)
JER evaluates the mean speaker-level intersection-over-union across all reference and hypothesis speakers:
$$\text{JER} = 1 - \frac{1}{|S_{\text{ref}}|} \sum_{s \in S_{\text{ref}}} \max_{s' \in S_{\text{hyp}}} \frac{|T(s) \cap T(s')|}{|T(s) \cup T(s')|}$$

### 3.3 Category-Weighted Error (CWE) & Custom Score
Standard DER treats Miss, False Alarm, and Confusion as equal costs. In commercial voice applications, these errors have drastically different business consequences. The benchmark introduces **Category-Weighted Error (CWE)** and the **Custom Score** ($0\text{–}100$, higher is better):

$$\text{CWE} = \frac{w_{\text{miss}} \cdot T_{\text{miss}} + w_{\text{fa}} \cdot T_{\text{fa}} + w_{\text{conf}} \cdot T_{\text{conf}}}{T_{\text{ref}}}$$

$$\text{Custom Score} = 100.0 \times \max\left(0.0, 1.0 - \text{CWE}\right)$$

### 3.4 Domain Error Weightings

| Category | $w_{\text{miss}}$ | $w_{\text{fa}}$ | $w_{\text{conf}}$ | Operational Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Call Centre / Call Recordings** | **0.40** | **0.25** | **0.35** | High confusion distorts compliance audits and agent scorecards; missed turns lose disclosures. |
| **Real-Time Transcriptions** | **0.50** | **0.20** | **0.30** | Missed speech directly causes dropped words in downstream ASR pipelines. |
| **Office Room Meetings** | **0.45** | **0.15** | **0.40** | High confusion severely disrupts meeting minutes and task action-item attribution. |
| **General / All-Purpose** | **0.40** | **0.30** | **0.30** | Balanced baseline weighting across diverse audio scenarios. |

---

## 4. Systems Evaluated

The benchmark covers 10 prominent diarization architectures:

| # | System Name | Paradigm / Core Engine | Architectural Family |
| :---: | :--- | :--- | :--- |
| 1 | **pyannote Community-1** | Neural segmentation + Agglomerative clustering | Modular Clustering |
| 2 | **NVIDIA NeMo Sortformer (Offline)** | Self-attention Sortformer (Conformer) | Transformer Sortformer |
| 3 | **NVIDIA NeMo Clustering Diarizer** | MarbleNet VAD + TitaNet embeddings + Spectral clustering | Modular Clustering |
| 4 | **NVIDIA NeMo Streaming Sortformer** | Causal streaming Sortformer with rolling context buffer | Streaming Sortformer |
| 5 | **SpeechBrain AMI** | ECAPA-TDNN embeddings + Spectral clustering | Modular Clustering |
| 6 | **WeSpeaker** | ResNet34 / CAM++ embeddings + Spectral clustering | Modular Clustering |
| 7 | **ESPnet (EEND-EDA)** | End-to-end neural diarization with encoder-decoder attractors | End-to-End Neural |
| 8 | **Kaldi Diarization** | Energy-based VAD + x-vector PLDA + AHC | Classical PLDA |
| 9 | **Google UIS-RNN** | Unbounded Interleaved-State RNN | Recurrent State Model |
| 10 | **BUT-FIT VBx** | ResNet/x-vector embeddings + Variational Bayes HMM | Bayesian State Model |

### 4.1 Architectural Paradigms & Pipeline Design
- **Self-Attention Sortformers (Models 2, 4):** Formulate speaker tracking as sequence sorting, handling multi-speaker overlaps directly without separate clustering.
- **Modular Neural Pipelines (Models 1, 3, 5, 6, 8):** Decompose the diarization problem into voice activity detection, deep speaker representation extraction, and pairwise similarity clustering.
- **End-to-End Multi-Speaker (Model 7):** Predicts active speaker tracks using attractor networks and permutation-invariant losses.
- **Probabilistic State Models (Models 9, 10):** Leverages Hidden Markov Models and Bayesian inference to enforce temporal speaker continuity.
