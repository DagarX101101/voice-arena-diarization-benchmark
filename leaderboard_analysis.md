# Leaderboard Analysis: Josh Talks Voice Arena Diarization-Scoring Project

**Document Generated:** October 19, 2026  
**Project Workspace:** `C:\Users\mrdag\OneDrive\Desktop\Task1 Josh talks`  
**Primary Source Modules Inspected:**
- [`Scoring/scoring_config.py`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/Scoring/scoring_config.py) — Configuration, dataset paths, weights, and schema definitions
- [`Scoring/scoring_utils.py`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/Scoring/scoring_utils.py) — Mathematical evaluation, metric extraction, aggregations, and ranking algorithms
- [`Scoring/scoring.py`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/Scoring/scoring.py) — Main evaluation pipeline and leaderboard orchestration

---

## 1. Architectural Overview of the Four Leaderboards

The scoring pipeline implements four distinct leaderboard views that evaluate models across three audio domains:
1. **General / All-Purpose Leaderboard** (Cross-domain macro & pooled corpus evaluation)
2. **Office Meetings Leaderboard** (`meeting` category)
3. **Call Recordings Leaderboard** (`call` category)
4. **Live Transcription Leaderboard** (`live_transcription` category)

### Global Scoring Parameters & Configuration
- **Collar:** `0.25` seconds (`DEFAULT_COLLAR_SECONDS = 0.25` in `scoring_config.py:58`).
- **Overlap Handling:** Overlapping speech segments are strictly scored (`DEFAULT_SKIP_OVERLAP = False` in `scoring_config.py:59`).
- **Scoring Window (UEM):** Full audio duration window without artificial cropping (`DEFAULT_UEM_POLICY = "FULL_AUDIO_NO_UEM"` in `scoring_config.py:60`).
- **Speaker Mapping Engine:** Hungarian optimal bipartite matching provided natively by `pyannote.metrics.diarization.DiarizationErrorRate` and `JaccardErrorRate`.

---

## 2. General / All-Purpose Leaderboard

### 2.1 Scope & Target Output
- **Target Output Deliverable:** [`scoring_output/overall_model_report.csv`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/scoring_output/overall_model_report.csv) and [`scoring_output/final_model_summary.csv`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/scoring_output/final_model_summary.csv).
- **Core Orchestrator Function:** `compute_overall_model_aggregates()` in [`Scoring/scoring_utils.py:485-623`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/Scoring/scoring_utils.py#L485-L623).

### 2.2 Input Categories & RTTM Files Used
The General Leaderboard evaluates performance across all three categories and all 5 canonical benchmark recordings:

| Category | Recording ID | Reference RTTM File | Audio File & Duration | Reference Speakers |
| :--- | :--- | :--- | :--- | :---: |
| **Call** | `call_input01` | `Audio and annotations/call_inputs_rttm/call_input01.rttm` | `call_input01.sph` (120.00s) | 3 |
| **Call** | `call_input02` | `Audio and annotations/call_inputs_rttm/call_input02.rttm` | `call_input02.flac` (120.00s) | 2 |
| **Live** | `liveTranscription_input01` | `Audio and annotations/liveTranscription_inputs_rttm/liveTranscription_input01.rttm` | `liveTranscription_input01.wav` (178.69s) | 2 |
| **Live** | `liveTranscription_input03` | `Audio and annotations/liveTranscription_inputs_rttm/liveTranscription_input03.rttm` | `liveTranscription_input03.wav` (60.00s) | 2 |
| **Meeting** | `meeting_input` | `Audio and annotations/meeting_inputs_rttm/meeting_input.rttm` | `meeting_input.wav` (139.91s) | 1 |

- **Hypothesis RTTMs Evaluated:**
  - Evaluated against Models 1 through 4 via `match_hypothesis_rttm()` searching `{recording_id}_Model0{slot}.rttm` in each model's `hypothesis_rttm/` directory.
  - Model Slots 5 through 10 are reserved slots; having no hypothesis RTTMs, they are marked `NOT_EVALUATED`.

### 2.3 Metrics Calculated (Implemented vs. Not Implemented)

| Metric | Status | Implementation Source |
| :--- | :---: | :--- |
| **DER (Corpus-Level Micro-Average)** | **Implemented** | `scoring_utils.py:569` |
| **JER (Macro-Average)** | **Implemented** | `scoring_utils.py:570` |
| **Missed Speech (%)** | **Implemented** | `scoring_utils.py:572` |
| **False Alarm (%)** | **Implemented** | `scoring_utils.py:573` |
| **Speaker Confusion (%)** | **Implemented** | `scoring_utils.py:574` |
| **Custom Weighted Error (`custom_weighted_error`)** | **Implemented** | `scoring_utils.py:587` |
| **Custom Score (`custom_score`)** | **Implemented** | `scoring_utils.py:588` |
| **Category Breakdown Scores** (`call_type_*`, `live_transcription_*`, `meeting_*`) | **Implemented** | `scoring_utils.py:613-621` |
| **Speech Precision** | **Not implemented** | Not present in codebase |
| **Speech Recall** | **Not implemented** | Not present in codebase |
| **Speech F1 / F-Measure** | **Not implemented** | Not present in codebase |

> [!NOTE]
> Standard DER and standard JER are **unmodified and unweighted**. No "weighted DER" or "modified JER" exists in the codebase. Weights are applied exclusively to compute the separate customer metric: `custom_weighted_error` and `custom_score`.

### 2.4 Exact Formulas Used by Code

1. **Pooled Speech Component Sums:**
   $$\text{Total Reference Speech } (T_{\text{ref}}) = \sum_{i \in \text{valid}} T_{\text{ref}, i}$$
   $$\text{Total Missed Speech } (T_{\text{miss}}) = \sum_{i \in \text{valid}} T_{\text{miss}, i}$$
   $$\text{Total False Alarm } (T_{\text{fa}}) = \sum_{i \in \text{valid}} T_{\text{fa}, i}$$
   $$\text{Total Speaker Confusion } (T_{\text{conf}}) = \sum_{i \in \text{valid}} T_{\text{conf}, i}$$

2. **Overall Corpus DER:**
   $$\text{Overall DER} = \frac{T_{\text{miss}} + T_{\text{fa}} + T_{\text{conf}}}{T_{\text{ref}}} \times 100.0 \quad \text{(`scoring\_utils.py:569`)}$$

3. **Overall Macro JER:**
   $$\text{Overall JER} = \frac{1}{N_{\text{valid}}} \sum_{i=1}^{N_{\text{valid}}} \text{JER}_i \quad \text{(`scoring\_utils.py:570`)}$$

4. **Overall Custom Weighted Error & Custom Score:**
   The codebase loops over every valid recording and computes the error duration using that recording's specific category weights:
   $$E_{\text{weighted}} = \sum_{i \in \text{valid}} \left( w_{\text{cat}(i), \text{missing}} \cdot T_{\text{miss}, i} + w_{\text{cat}(i), \text{false\_alarm}} \cdot T_{\text{fa}, i} + w_{\text{cat}(i), \text{confusion}} \cdot T_{\text{conf}, i} \right)$$
   $$\text{Overall CWE} = \frac{E_{\text{weighted}}}{T_{\text{ref}}} \quad \text{(`scoring\_utils.py:587`)}$$
   $$\text{Overall Custom Score} = 100.0 \times \max\left(0.0, 1.0 - \text{Overall CWE}\right) \quad \text{(`scoring\_utils.py:588`)}$$

### 2.5 Confirmation: Simple Average vs. Duration-Weighted Pooling
- **Confirmation:** The General Leaderboard **DOES NOT use a simple average of the three category scores** (i.e. $\frac{\text{Call Score} + \text{Live Score} + \text{Meeting Score}}{3}$ is **not** used).
- **Actual Code Behavior:** It computes a **duration-weighted pooling** across all valid recordings. Longer recordings contribute more reference speech duration to the denominator $T_{\text{ref}}$ and therefore have a proportionally higher impact on both the overall DER and the overall Custom Score. Category-specific scores are preserved in parallel columns for informational reporting.

### 2.6 Directionality (Lower vs. Higher Better)
- **DER, JER, Miss %, FA %, Confusion %, Custom Weighted Error:** **Lower is better** (0.0% is optimal).
- **Custom Score:** **Higher is better** (100.0 is optimal, bounded below at 0.0).
- **Rank (`overall_rank`):** **Lower integer is better** (Rank 1 is 1st place).

### 2.7 Missing or Invalid Reference Handling
- If a reference has $\le 0.0$ seconds of speech within the audio bounds (e.g., `call_input02` in official baseline whose annotations lie at 400.35s–479.77s), `evaluate_pair()` returns `"is_valid": False` with status `INVALID_REFERENCE_DURATION` (`scoring_utils.py:365`).
- In `compute_overall_model_aggregates()`, invalid comparisons are **completely excluded** from the numerator and denominator of all metrics.
- Evaluated vs. expected comparisons are tracked:
  - If $N_{\text{valid}} = 0$: Metrics are `None`, completion status is `NOT_EVALUATED`.
  - If $0 < N_{\text{valid}} < N_{\text{expected}}$: Valid recordings are scored, completion status is `PARTIAL`.
  - If $N_{\text{valid}} = N_{\text{expected}}$: Completion status is `SUCCESS`.

### 2.8 Final Rankings and Tie-Break Rules
- Handled by `assign_ranks()` in [`Scoring/scoring_utils.py:625-654`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/Scoring/scoring_utils.py#L625-L654).
- **Primary Ranking Key:** `custom_score` (higher is better).
- **Tie-Break Handling:** Competition ranking (1224 / "min" ranking). If two models have identical custom scores, they receive the exact same rank integer. The next model's rank skips by the number of tied models (e.g., ties at rank 2 result in ranks `[1, 2, 2, 4]`).
- **Missing Score Handling:** Models with `custom_score = None` receive `overall_rank = None` (displayed as blank in CSV).
- **Table Sorting:** Ranked models appear first (sorted ascending by `overall_rank`), followed by unranked models (sorted ascending by `model_number`).

---

## 3. Office Meetings Leaderboard

### 3.1 Scope & Target Output
- **Target Output Deliverable:** [`scoring_output/meeting_report.csv`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/scoring_output/meeting_report.csv).
- **Core Orchestrator Function:** `compute_category_aggregates()` in [`Scoring/scoring_utils.py:420-483`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/Scoring/scoring_utils.py#L420-L483) and per-recording evaluation loop in [`Scoring/scoring.py:151-322`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/Scoring/scoring.py#L151-L322).

### 3.2 Input Categories & RTTM Files Used
- **Customer Category:** `meeting` (`RECORDINGS_BY_CATEGORY["meeting"] = ["meeting_input"]`).
- **Input Audio:** `Audio and annotations/meeting_inputs/meeting_input.wav` (Duration: 139.91s, 16 kHz WAV).
- **Ground Truth Reference RTTM:** `Audio and annotations/meeting_inputs_rttm/meeting_input.rttm` (Single speaker presentation dialogue excerpt from VoxConverse clip `zvmyn.wav`, reference speech duration: ~113.20s).
- **Hypothesis RTTMs Evaluated:**
  - `Model1(pyannote Community-1)/hypothesis_rttm/meeting_input_Model01.rttm`
  - `Model2(NVIDIA NeMo Sortformer)/hypothesis_rttm/meeting_input_Model02.rttm`
  - `Model3(NVIDIA NeMo Clustering Diarizer)/hypothesis_rttm/meeting_input_Model03.rttm`
  - `Model4(NVIDIA NeMo Streaming Sortformer)/hypothesis_rttm/meeting_input_Model04.rttm`
  - Models 5–10: `NOT_EVALUATED` (reserved slots).

### 3.3 Metrics Calculated (Implemented vs. Not Implemented)

| Metric | Status | Implementation Source |
| :--- | :---: | :--- |
| **DER (%)** | **Implemented** | `scoring_utils.py:408` |
| **JER (%)** | **Implemented** | `scoring_utils.py:409` |
| **Missed Speech (%)** | **Implemented** | `scoring_utils.py:410` |
| **False Alarm (%)** | **Implemented** | `scoring_utils.py:411` |
| **Speaker Confusion (%)** | **Implemented** | `scoring_utils.py:412` |
| **Meeting Custom Weighted Error** | **Implemented** | `scoring_utils.py:395` |
| **Meeting Custom Score** | **Implemented** | `scoring_utils.py:397` |
| **Speech Precision** | **Not implemented** | Not present in codebase |
| **Speech Recall** | **Not implemented** | Not present in codebase |
| **Speech F1 / F-Measure** | **Not implemented** | Not present in codebase |

### 3.4 Exact Formulas Used by Code

1. **Domain-Specific Customer Weights (`meeting`):**
   $$w_{\text{missing}} = 0.45, \quad w_{\text{false\_alarm}} = 0.15, \quad w_{\text{confusion}} = 0.40 \quad (\text{Sum} = 1.00)$$
   Defined in `Scoring/scoring_config.py:74-78`.

2. **Diarization Error Rate (DER):**
   $$\text{DER} = \frac{T_{\text{miss}} + T_{\text{fa}} + T_{\text{conf}}}{T_{\text{ref}}} \times 100.0$$

3. **Jaccard Error Rate (JER):**
   $$\text{JER} = \text{float}(\text{JaccardErrorRate}(collar=0.25, skip\_overlap=False)) \times 100.0$$

4. **Component Percentages:**
   $$\text{Miss } \% = \frac{T_{\text{miss}}}{T_{\text{ref}}} \times 100.0, \quad \text{FA } \% = \frac{T_{\text{fa}}}{T_{\text{ref}}} \times 100.0, \quad \text{Confusion } \% = \frac{T_{\text{conf}}}{T_{\text{ref}}} \times 100.0$$

5. **Meeting Custom Weighted Error & Custom Score:**
   $$\text{CWE}_{\text{meeting}} = \frac{0.45 \cdot T_{\text{miss}} + 0.15 \cdot T_{\text{fa}} + 0.40 \cdot T_{\text{conf}}}{T_{\text{ref}}}$$
   $$\text{Custom Score}_{\text{meeting}} = 100.0 \times \max\left(0.0, 1.0 - \text{CWE}_{\text{meeting}}\right)$$

### 3.5 Directionality
- **DER, JER, Miss %, FA %, Confusion %, CWE:** Lower is better.
- **Custom Score:** Higher is better.
- **Category Rank (`rank`):** Lower integer is better.

### 3.6 Aggregation Across Recordings
- Because the `meeting` category contains a single canonical recording (`meeting_input`), the category aggregate equals the recording-level evaluation.
- When generalized across multiple meeting files via `compute_category_aggregates()`, DER and Custom Score are computed via duration-weighted pooled sums, and JER is computed via arithmetic macro-averaging.

### 3.7 Handling of Missing or Invalid References
- If audio or reference is missing or has zero reference speech, the comparison status is set to `FAILED` or `INVALID`, metrics are set to `None`, and the row is excluded from ranking.

### 3.8 Final Rankings and Ties
- Ranked by `assign_ranks(rows, score_key="custom_score", rank_key="rank", higher_is_better=True)`.
- Highest `custom_score` receives Rank 1. Ties share the same rank integer with standard competition spacing.

---

## 4. Call Recordings Leaderboard

### 4.1 Scope & Target Output
- **Target Output Deliverable:** [`scoring_output/call_type_report.csv`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/scoring_output/call_type_report.csv).
- **Core Orchestrator Function:** `compute_category_aggregates()` and per-recording evaluation loop in `Scoring/scoring.py:151-322`.

### 4.2 Input Categories & RTTM Files Used
- **Customer Category:** `call` (`RECORDINGS_BY_CATEGORY["call"] = ["call_input01", "call_input02"]`).
- **Input Audio & Reference Files:**
  1. `call_input01`:
     - Audio: `Audio and annotations/call_inputs_audio/call_input01.sph` (120.00s, 8 kHz mono telephony, CALLHOME LDC97S42).
     - Reference: `Audio and annotations/call_inputs_rttm/call_input01.rttm` (3 speakers: A, A1, B; reference speech duration: 55.31s).
  2. `call_input02`:
     - Audio: `Audio and annotations/call_inputs_audio/call_input02.flac` (120.00s, 8 kHz stereo telephony, CALLHOME `en_5551`).
     - Reference (Official Baseline): `Audio and annotations/call_inputs_rttm/call_input02.rttm` (Original timestamps span `[400.350s, 479.770s]`).
     - Reference (Provisional Alignment Override): `Scoring/aligned_references/call_input02_aligned.rttm` (Timestamps shifted by -400.00s to span `[0.350s, 79.770s]`).
- **Hypothesis RTTMs Evaluated:**
  - `Model1(pyannote Community-1)/hypothesis_rttm/call_input0[1|2]_Model01.rttm`
  - `Model2(NVIDIA NeMo Sortformer)/hypothesis_rttm/call_input0[1|2]_Model02.rttm`
  - `Model3(NVIDIA NeMo Clustering Diarizer)/hypothesis_rttm/call_input0[1|2]_Model03.rttm`
  - `Model4(NVIDIA NeMo Streaming Sortformer)/hypothesis_rttm/call_input0[1|2]_Model04.rttm`
  - Models 5–10: `NOT_EVALUATED`.

### 4.3 Metrics Calculated (Implemented vs. Not Implemented)

| Metric | Status | Implementation Source |
| :--- | :---: | :--- |
| **DER (%)** | **Implemented** | `scoring_utils.py:408` |
| **JER (%)** | **Implemented** | `scoring_utils.py:409` |
| **Missed Speech (%)** | **Implemented** | `scoring_utils.py:410` |
| **False Alarm (%)** | **Implemented** | `scoring_utils.py:411` |
| **Speaker Confusion (%)** | **Implemented** | `scoring_utils.py:412` |
| **Call Custom Weighted Error** | **Implemented** | `scoring_utils.py:395` |
| **Call Custom Score** | **Implemented** | `scoring_utils.py:397` |
| **Speech Precision** | **Not implemented** | Not present in codebase |
| **Speech Recall** | **Not implemented** | Not present in codebase |
| **Speech F1 / F-Measure** | **Not implemented** | Not present in codebase |

### 4.4 Exact Formulas Used by Code

1. **Domain-Specific Customer Weights (`call`):**
   $$w_{\text{missing}} = 0.40, \quad w_{\text{false\_alarm}} = 0.25, \quad w_{\text{confusion}} = 0.35 \quad (\text{Sum} = 1.00)$$
   Defined in `Scoring/scoring_config.py:64-68`.

2. **Per-Recording Formulas:**
   $$\text{DER} = \frac{T_{\text{miss}} + T_{\text{fa}} + T_{\text{conf}}}{T_{\text{ref}}} \times 100.0$$
   $$\text{CWE}_{\text{call}} = \frac{0.40 \cdot T_{\text{miss}} + 0.25 \cdot T_{\text{fa}} + 0.35 \cdot T_{\text{conf}}}{T_{\text{ref}}}$$
   $$\text{Custom Score}_{\text{call}} = 100.0 \times \max\left(0.0, 1.0 - \text{CWE}_{\text{call}}\right)$$

3. **Category-Level Aggregation (`compute_category_aggregates`):**
   $$\text{Category DER}_{\text{call}} = \frac{\sum_{i \in \text{valid}} (T_{\text{miss}, i} + T_{\text{fa}, i} + T_{\text{conf}, i})}{\sum_{i \in \text{valid}} T_{\text{ref}, i}} \times 100.0$$
   $$\text{Category JER}_{\text{call}} = \frac{1}{N_{\text{valid}}} \sum_{i=1}^{N_{\text{valid}}} \text{JER}_i$$
   $$\text{Category CWE}_{\text{call}} = \frac{\sum_{i \in \text{valid}} (0.40 \cdot T_{\text{miss}, i} + 0.25 \cdot T_{\text{fa}, i} + 0.35 \cdot T_{\text{conf}, i})}{\sum_{i \in \text{valid}} T_{\text{ref}, i}}$$
   $$\text{Category Custom Score}_{\text{call}} = 100.0 \times \max\left(0.0, 1.0 - \text{Category CWE}_{\text{call}}\right)$$

### 4.5 Directionality
- **DER, JER, Miss %, FA %, Confusion %, CWE:** Lower is better.
- **Custom Score:** Higher is better.
- **Rank:** Lower integer is better.

### 4.6 Missing or Invalid Reference Handling & Distinction Between Official vs. Provisional
- **Official Baseline Mode (Default):**
  - `call_input02.rttm` timestamps lie entirely in `[400.35s, 479.77s]`. When evaluated within the 120s audio duration, reference speech duration is `0.00s`.
  - The scorer flags all systems on `call_input02` as `INVALID` with error `INVALID_REFERENCE_DURATION` (`scoring.py:303`).
  - `call_input02` rows have empty metrics and empty rank.
  - In the official Call Leaderboard, the category aggregate is computed solely over the valid recording `call_input01`.
- **Provisional Alignment Mode (`--enable-provisional-alignment`):**
  - Activated via CLI flag.
  - Substitutes `call_input02.rttm` with the isolated copy `Scoring/aligned_references/call_input02_aligned.rttm`.
  - Both `call_input01` and `call_input02` achieve status `SUCCESS` / `PROVISIONAL_ALIGNED`, enabling evaluation across both files.
  - These results are explicitly quarantined and not mixed into the official baseline rankings.

### 4.7 Final Rankings and Ties
- Models are ranked on valid comparisons by `assign_ranks()` using `custom_score` descending.

---

## 5. Live Transcription Leaderboard

### 5.1 Scope & Target Output
- **Target Output Deliverable:** [`scoring_output/live_transcription_report.csv`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/scoring_output/live_transcription_report.csv).
- **Core Orchestrator Function:** `compute_category_aggregates()` and per-recording evaluation loop in `Scoring/scoring.py:151-322`.

### 5.2 Input Categories & RTTM Files Used
- **Customer Category:** `live_transcription` (`RECORDINGS_BY_CATEGORY["live_transcription"] = ["liveTranscription_input01", "liveTranscription_input03"]`).
- **Input Audio & Reference Files:**
  1. `liveTranscription_input01`:
     - Audio: `Audio and annotations/liveTranscription_inputs_audio/liveTranscription_input01.flac` / `.wav` (178.69s, 16 kHz broadcast dialogue, URI `DH_0052`).
     - Reference: `Audio and annotations/liveTranscription_inputs_rttm/liveTranscription_input01.rttm` (2 speakers: `speaker117`, `speaker118`; reference speech duration: 154.99s; highly overlapping speech).
  2. `liveTranscription_input03`:
     - Audio: `Audio and annotations/liveTranscription_inputs_audio/liveTranscription_input03.wav` (60.00s, 16 kHz clean conversational dialogue, URI `zyffh`).
     - Reference: `Audio and annotations/liveTranscription_inputs_rttm/liveTranscription_input03.rttm` (2 speakers: `spk00`, `spk01`; reference speech duration: 58.74s).
- **Hypothesis RTTMs Evaluated:**
  - `Model1(pyannote Community-1)/hypothesis_rttm/liveTranscription_input0[1|3]_Model01.rttm`
  - `Model2(NVIDIA NeMo Sortformer)/hypothesis_rttm/liveTranscription_input0[1|3]_Model02.rttm`
  - `Model3(NVIDIA NeMo Clustering Diarizer)/hypothesis_rttm/liveTranscription_input0[1|3]_Model03.rttm`
  - `Model4(NVIDIA NeMo Streaming Sortformer)/hypothesis_rttm/liveTranscription_input0[1|3]_Model04.rttm`
  - Models 5–10: `NOT_EVALUATED`.

### 5.3 Metrics Calculated (Implemented vs. Not Implemented)

| Metric | Status | Implementation Source |
| :--- | :---: | :--- |
| **DER (%)** | **Implemented** | `scoring_utils.py:408` |
| **JER (%)** | **Implemented** | `scoring_utils.py:409` |
| **Missed Speech (%)** | **Implemented** | `scoring_utils.py:410` |
| **False Alarm (%)** | **Implemented** | `scoring_utils.py:411` |
| **Speaker Confusion (%)** | **Implemented** | `scoring_utils.py:412` |
| **Live Transcription Custom Weighted Error** | **Implemented** | `scoring_utils.py:395` |
| **Live Transcription Custom Score** | **Implemented** | `scoring_utils.py:397` |
| **Speech Precision** | **Not implemented** | Not present in codebase |
| **Speech Recall** | **Not implemented** | Not present in codebase |
| **Speech F1 / F-Measure** | **Not implemented** | Not present in codebase |

### 5.4 Exact Formulas Used by Code

1. **Domain-Specific Customer Weights (`live_transcription`):**
   $$w_{\text{missing}} = 0.50, \quad w_{\text{false\_alarm}} = 0.20, \quad w_{\text{confusion}} = 0.30 \quad (\text{Sum} = 1.00)$$
   Defined in `Scoring/scoring_config.py:69-73`. (Prioritizes low missed detection to preserve real-time transcription coverage).

2. **Per-Recording Formulas:**
   $$\text{DER} = \frac{T_{\text{miss}} + T_{\text{fa}} + T_{\text{conf}}}{T_{\text{ref}}} \times 100.0$$
   $$\text{CWE}_{\text{live}} = \frac{0.50 \cdot T_{\text{miss}} + 0.20 \cdot T_{\text{fa}} + 0.30 \cdot T_{\text{conf}}}{T_{\text{ref}}}$$
   $$\text{Custom Score}_{\text{live}} = 100.0 \times \max\left(0.0, 1.0 - \text{CWE}_{\text{live}}\right)$$

3. **Category-Level Aggregation (`compute_category_aggregates`):**
   $$\text{Category DER}_{\text{live}} = \frac{T_{\text{miss}, 01} + T_{\text{miss}, 03} + T_{\text{fa}, 01} + T_{\text{fa}, 03} + T_{\text{conf}, 01} + T_{\text{conf}, 03}}{T_{\text{ref}, 01} + T_{\text{ref}, 03}} \times 100.0$$
   $$\text{Category JER}_{\text{live}} = \frac{\text{JER}_{01} + \text{JER}_{03}}{2}$$
   $$\text{Category CWE}_{\text{live}} = \frac{\sum_{i \in \{01, 03\}} (0.50 \cdot T_{\text{miss}, i} + 0.20 \cdot T_{\text{fa}, i} + 0.30 \cdot T_{\text{conf}, i})}{T_{\text{ref}, 01} + T_{\text{ref}, 03}}$$
   $$\text{Category Custom Score}_{\text{live}} = 100.0 \times \max\left(0.0, 1.0 - \text{Category CWE}_{\text{live}}\right)$$

### 5.5 Directionality
- **DER, JER, Miss %, FA %, Confusion %, CWE:** Lower is better.
- **Custom Score:** Higher is better.
- **Rank:** Lower integer is better.

### 5.6 Aggregation Across Recordings
- Both recordings (`liveTranscription_input01` and `liveTranscription_input03`) are valid in baseline scoring.
- Because `liveTranscription_input01` has ~154.99s of reference speech and `liveTranscription_input03` has ~58.74s, `liveTranscription_input01` accounts for ~72.5% of the category reference speech pool, exerting higher influence on Category DER and Custom Score than `input03`.

### 5.7 Handling of Missing or Invalid References
- Handled via `compute_category_aggregates()`: if either file failed, only the valid file would enter the pooled calculation. Since both are valid, $N_{\text{valid}} = 2$.

### 5.8 Final Rankings and Ties
- Standard competition ranking (`assign_ranks()`) ordered by `custom_score` descending.

---

## 6. Comprehensive Cross-Leaderboard Comparison Matrix

| Leaderboard Dimension | General / All-Purpose | Office Meetings | Call Recordings | Live Transcription |
| :--- | :--- | :--- | :--- | :--- |
| **Output File** | `overall_model_report.csv` | `meeting_report.csv` | `call_type_report.csv` | `live_transcription_report.csv` |
| **Associated Category** | Cross-Category (All 3) | `meeting` | `call` | `live_transcription` |
| **Recordings Included** | 5 recordings | 1 recording (`meeting_input`) | 2 recordings (`call_01`, `call_02`) | 2 recordings (`live_01`, `live_03`) |
| **Missing Speech Weight ($w_{\text{m}}$)** | Category-dependent | **0.45** | **0.40** | **0.50** |
| **False Alarm Weight ($w_{\text{fa}}$)** | Category-dependent | **0.15** | **0.25** | **0.20** |
| **Confusion Weight ($w_{\text{conf}}$)** | Category-dependent | **0.40** | **0.35** | **0.30** |
| **DER Aggregation Method** | **Pooled Corpus Micro-Average** | Single file (Pooled if >1) | **Pooled Corpus Micro-Average** | **Pooled Corpus Micro-Average** |
| **JER Aggregation Method** | **Unweighted Macro-Average** | Single file (Macro if >1) | **Unweighted Macro-Average** | **Unweighted Macro-Average** |
| **Score Aggregation Method** | **Duration-Weighted Pooled CWE** | Single file (Pooled if >1) | **Duration-Weighted Pooled CWE** | **Duration-Weighted Pooled CWE** |
| **Uses Simple Average of Sub-Scores?** | **NO** (Duration-weighted) | N/A (Single recording) | **NO** (Duration-weighted) | **NO** (Duration-weighted) |
| **Evaluation Collar** | 0.25s | 0.25s | 0.25s | 0.25s |
| **Overlap Handling** | Evaluated (`skip=False`) | Evaluated (`skip=False`) | Evaluated (`skip=False`) | Evaluated (`skip=False`) |
| **Primary Ranking Metric** | `custom_score` (Descending) | `custom_score` (Descending) | `custom_score` (Descending) | `custom_score` (Descending) |
| **Tie-Breaker Rule** | Competition "min" (1224) | Competition "min" (1224) | Competition "min" (1224) | Competition "min" (1224) |
| **Official Completion Status** | `PARTIAL` (4/5 valid) | `SUCCESS` (1/1 valid) | `PARTIAL` (1/2 valid) | `SUCCESS` (2/2 valid) |
| **Speech Precision / Recall / F1** | **Not implemented** | **Not implemented** | **Not implemented** | **Not implemented** |

---

## 7. Current Official Benchmark Standing Across Leaderboards

Based on execution of the official baseline pipeline ([`Optional Task - Diarization Scoring\.venv\Scripts\python.exe Scoring/scoring.py`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/Optional%20Task%20-%20Diarization%20Scoring/.venv/Scripts/python.exe)):

### 7.1 General Leaderboard Standings
1. **Rank 1: Model 2 (NeMo Sortformer Offline)** — Custom Score: **92.31** | DER: 25.30% | JER: 37.56%
2. **Rank 2: Model 4 (NeMo Streaming Sortformer)** — Custom Score: **92.25** | DER: 26.80% | JER: 33.61%
3. **Rank 3: Model 1 (pyannote Community-1)** — Custom Score: **91.72** | DER: 27.87% | JER: 49.27%
4. **Rank 4: Model 3 (NeMo Clustering Diarizer)** — Custom Score: **91.21** | DER: **22.83%** (Best DER) | JER: 48.81%
5. **Slots 5–10:** `NOT_EVALUATED` (No hypotheses present, blank rank).

### 7.2 Category Leaderboard Standings
- **Office Meetings Leaderboard:**
  - **Rank 1: Model 4 (NeMo Streaming Sortformer)** — Custom Score: **99.83** | DER: **1.12%** | JER: **1.10%**
  - Rank 2: Model 2 (NeMo Sortformer) — Custom Score: 99.75 | DER: 1.19% | JER: 1.18%
  - Rank 3: Model 3 (NeMo Clustering) — Custom Score: 99.60 | DER: 1.32% | JER: 1.32%
  - Rank 4: Model 1 (pyannote Community-1) — Custom Score: 99.13 | DER: 2.50% | JER: 2.48%
- **Call Recordings Leaderboard (Official Baseline on `call_input01`):**
  - **Rank 1: Model 3 (NeMo Clustering Diarizer)** — Custom Score: **79.26** | DER: **72.18%** | JER: 64.90%
  - Rank 2: Model 2 (NeMo Sortformer) — Custom Score: 73.74 | DER: 77.89% | JER: 61.73%
  - Rank 3: Model 4 (NeMo Streaming) — Custom Score: 73.24 | DER: 80.67% | JER: **49.60%**
  - Rank 4: Model 1 (pyannote Community-1) — Custom Score: 71.99 | DER: 97.14% | JER: 78.71%
  - *`call_input02`: `INVALID_REFERENCE_DURATION` across all models.*
- **Live Transcription Leaderboard:**
  - **Rank 1: Model 2 (NeMo Sortformer)** — Custom Score: **93.11** | DER: 24.22% | JER: 43.66%
  - Rank 2: Model 4 (NeMo Streaming) — Custom Score: 93.07 | DER: 26.06% | JER: **41.86%**
  - Rank 3: Model 1 (pyannote Community-1) — Custom Score: 92.75 | DER: 23.99% | JER: 57.93%
  - Rank 4: Model 3 (NeMo Clustering) — Custom Score: 90.34 | DER: **21.38%** | JER: 64.51%

---

## 8. Summary of Verifications & Negative Findings

1. **Speech Precision, Recall, and F1:** Confirmed **Not implemented**. Neither `pyannote.metrics.detection` nor custom precision/recall/F1 metrics are instantiated or reported in `scoring.py`, `scoring_utils.py`, or `scoring_config.py`.
2. **Modified / Weighted DER or JER:** Confirmed **Not implemented**. DER and JER follow standard unweighted definitions strictly. Customer weights are applied only to derive `custom_weighted_error` and `custom_score`.
3. **General Leaderboard Aggregation:** Confirmed that the General Leaderboard **does not use a simple average of category scores**. It calculates a **duration-weighted pooled custom error duration divided by total pooled reference speech duration**.
4. **Configuration Stability:** The 0.25s collar, overlap inclusion (`skip_overlap = False`), and full-audio UEM policy are consistently enforced across all four leaderboards.
