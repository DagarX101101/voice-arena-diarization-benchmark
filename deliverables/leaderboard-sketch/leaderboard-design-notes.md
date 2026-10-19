# Voice Arena Leaderboard: Product Design Notes & Architecture Sketch

**Deliverable Specification:** Josh Talks AI Assessment — Speaker Diarization Benchmark Product Mockup  
**Product Mockup Deliverable:** [`deliverables/leaderboard-sketch/leaderboard-mockup.html`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/deliverables/leaderboard-sketch/leaderboard-mockup.html)  
**Screenshot Deliverable:** [`deliverables/leaderboard-sketch/leaderboard-mockup.png`](file:///c:/Users/mrdag/OneDrive/Desktop/Task1%20Josh%20talks/deliverables/leaderboard-sketch/leaderboard-mockup.png)  
**Date:** October 19, 2026  

---

## 1. Product Vision & Executive Overview

The **Voice Arena Diarization Leaderboard** is designed as a modern SaaS AI evaluation platform for engineering leaders, speech scientists, and product managers selecting speaker diarization models for enterprise deployment. Rather than presenting static research spreadsheets, Voice Arena provides an interactive product interface that communicates:
1. **Domain-Specific Fitness:** How models perform in distinct operational environments (telephony, live overlapping broadcast, and multi-speaker office meetings).
2. **Business-Aligned Scoring:** Moving beyond raw DER to a normalized **Custom Score ($0\text{–}100$)** that reflects the asymmetric financial and operational costs of missed speech, false alarms, and speaker confusion.
3. **Transparent Architectural Inspection:** Direct visibility into the underlying pipelines (Voice Activity Detectors, Embedding representations, and Clustering/Sorting mechanisms).

---

## 2. Layout Structure & Visual Hierarchy

The interface follows a top-down visual hierarchy designed for rapid decision-making:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  HEADER: Product Branding • Executive PDF • Methodology • Theme Toggle       │
├──────────────────────────────────────────────────────────────────────────────┤
│  HERO BANNER: Benchmark Objective & Core Protocol (0.25s Collar, Hungarian)   │
├──────────────────────────────────────────────────────────────────────────────┤
│  SUMMARY KPI CARDS: Active Category • Category Champion • Top Score • Scale  │
├──────────────────────────────────────────────────────────────────────────────┤
│  VIEW SELECTOR: Segmented Tabs (General, Call Centre, Real-Time, Meetings)   │
│                 + Dynamic Category Error Cost Weights (w_m, w_fa, w_conf)     │
├──────────────────────────────────────────────────────────────────────────────┤
│  CONTROLS BAR: Live Search • Architecture Filter • Sort Options • Reset      │
├──────────────────────────────────────────────────────────────────────────────┤
│  MAIN LEADERBOARD TABLE:                                                     │
│  Rank | Model & Architecture | Custom Score + Bar | DER | JER | Miss | FA... │
│  [Row 1 Highlighted with Gold Accent & Champion Trophy]                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  SCORING FORMULA PANEL: Math Definition • Domain Weights Impact Table        │
├──────────────────────────────────────────────────────────────────────────────┤
│  SLIDE-OUT INSPECTION DRAWER: (Triggered on Row / Inspect Click)             │
│  - Relative Error Stacked Bar (Missed % vs FA % vs Conf %)                   │
│  - Technical Pipeline (VAD, Embeddings, Clustering)                          │
│  - Operational Strengths & Production Trade-offs                             │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. What Sits Next to Each Score?

In the product sketch, the **Custom Score** column is designed as a composite visual metric rather than a standalone number:

1. **Numerical Score Value:** Rendered in high-legibility bold monospace font (`JetBrains Mono`, 16px/1rem). Top-ranked models display in warm amber gold (`#d97706`), while subsequent models display in primary indigo (`#2563eb`).
2. **Compact Score Meter Bar:** Directly below the number sits a 6px horizontal progress bar filled proportionally from 0 to 100%. This provides instant visual cognition of model capability without requiring the user to read fine decimal numbers.
3. **Performance Tier Badge:** Adjacent to the score sits a categorical tier pill:
   - `Elite` ($95.00+$ Custom Score): Near-flawless production stability (e.g. NeMo Streaming Sortformer in meetings).
   - `High` ($90.00\text{–}94.99$ Custom Score): Excellent enterprise candidate with minimal attribution errors.
   - `Good` ($75.00\text{–}89.99$ Custom Score): Strong performance in challenging environments (e.g. 8 kHz telephony).
   - `Standard` ($<75.00$ Custom Score): Viable baseline with elevated false alarms or missed speech.
4. **Interactive Information Tooltip:** Hovering over the column header or score provides an instant popover with the mathematical formula:  
   `Custom Score = 100.0 × max(0.0, 1.0 − CWE)`

---

## 4. Filters, Controls, and Navigation

### 4.1 Category Navigation (Domain Switching)
The benchmark features four segmented category views:
- **General / All-Purpose:** Macro evaluation across 445.3 seconds of reference speech ($w_{\text{m}}=0.40, w_{\text{fa}}=0.30, w_{\text{conf}}=0.30$).
- **Call Centre Telephony:** Band-limited 8 kHz mono audio from CALLHOME ($w_{\text{m}}=0.40, w_{\text{fa}}=0.25, w_{\text{conf}}=0.35$).
- **Real-Time Transcriptions:** Rapid multi-speaker broadcast dialogue with heavy cross-talk ($w_{\text{m}}=0.50, w_{\text{fa}}=0.20, w_{\text{conf}}=0.30$).
- **Office Room Meetings:** Formal presentation audio with reverberation ($w_{\text{m}}=0.45, w_{\text{fa}}=0.15, w_{\text{conf}}=0.40$).

Switching categories instantly re-computes the KPI cards, displays the exact domain error weights in the top pill, and re-ranks all 10 models.

### 4.2 Multi-Dimensional Filtering & Sorting
- **Live Search:** Instant fuzzy filtering by model name (e.g. `NeMo`, `Sortformer`, `pyannote`, `VBx`) or architecture keyword.
- **Architecture Paradigm Filter:** Filters models by foundational design family:
  - *Self-Attention / Sortformers* (Models 2, 4)
  - *Modular VAD + Clustering* (Models 1, 3, 5, 6, 8)
  - *End-to-End Attractors* (Model 7)
  - *Probabilistic / Bayesian HMM* (Models 9, 10)
- **Multi-Metric Sorting:** Allows one-click sorting by Custom Score (High to Low), DER % (Low to High), JER % (Low to High), Missed % (Low to High), False Alarm % (Low to High), or Confusion % (Low to High).

---

## 5. Model Inspection Drawer (Deep Architectural Drill-Down)

Clicking on any row or the **"Inspect"** button opens a focused slide-out panel that details:

1. **Relative Error Distribution Bar:** A stacked, proportional bar chart showing the relative percentage breakdown of the model's total error budget divided into:
   - 🟧 **Missed Speech** (Orange)
   - 🟥 **False Alarms** (Red)
   - 🟪 **Speaker Confusion** (Purple)
2. **Three-Tier Architectural Pipeline:**
   - **VAD Module:** Frame-level voice activity detector (e.g., MarbleNet 8x4, Neural bimodal segmentation, or causal joint frames).
   - **Speaker Embedding:** Representation dimension and architecture (e.g., TitaNet-Large, ECAPA-TDNN 192-dim, ResNet34 CAM++).
   - **Clustering / Sorting Engine:** Multi-scale spectral graph clustering, Variational Bayes HMM, or cross-attention Sortformer sorting.
3. **Operational Assessment:** Concrete engineering notes on production strengths and known channel trade-offs.

---

## 6. Design System & Accessibility Decisions

- **Color Palette:** Professional slate/neutral base (`#0f172a`, `#f8fafc`) with accent indigo (`#2563eb`) for interactive states and warm amber (`#d97706`) for champions. Fully WCAG AAA contrast-compliant.
- **Typography:** Dual-typeface pairing using **Plus Jakarta Sans** for UI elements and **JetBrains Mono** for numerical scores, DER percentages, and mathematical formulas.
- **Responsive Layout:** Adapts seamlessly from 4K ultrawide monitors down to mobile viewports (1024px, 768px, and 480px breakpoints).
- **Zero External Dependencies:** Built with vanilla HTML, modern CSS variables, and clean JavaScript. Runs locally, completely offline, and without npm/bundler toolchains.
