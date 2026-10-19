# Josh Talks Voice Arena: Interactive Diarization Leaderboard

A responsive, high-performance web dashboard evaluating 10 speaker diarization models across 4 operational enterprise categories:
1. **General / All-Purpose** (Macro aggregate)
2. **Call Centre / Call Recordings** (8 kHz telephony)
3. **Real-Time Conversations & Transcriptions** (High speaker overlap)
4. **Office Room Meetings** (Multi-speaker presentations)

---

## Features

- **Category Switching:** Instant tab navigation between operational domains.
- **Dynamic Sorting:** Click any column header (Rank, Model, Status, DER %, JER %, Miss %, FA %, Confusion %, Custom Score) to toggle ascending/descending order.
- **Multi-Dimensional Filters:** Search by model name or paradigm, filter by execution status (*Measured* vs. *Estimated*), or filter by model architecture.
- **Scoring Engine Breakdown:** Detailed explanation of the Category-Weighted Error (CWE) and Custom Score formulas.
- **Deep Architectural Inspection:** Click on any model row or the *Inspect* button to open a detailed drawer displaying VAD, embedding extractor, clustering engine, strengths, limitations, and exact metrics.
- **Theme Support:** Dark/Light mode toggle with local preference persistence.
- **Zero Dependencies:** Pure vanilla HTML, CSS, and modern JavaScript. Runs offline and requires no build step or node package installations.

---

## How to Run Locally

### Option 1: Python Built-In HTTP Server (Recommended)
From the repository root or the `leaderboard/` directory, run:

```bash
# From repository root:
python -m http.server 8000

# Or using the project's virtual environment:
"Optional Task - Diarization Scoring\.venv\Scripts\python.exe" -m http.server 8000
```
Then open your browser at:
`http://localhost:8000/leaderboard/`

### Option 2: Direct File Open
You can open `leaderboard/index.html` directly in any modern browser:
- Double click `leaderboard/index.html` in File Explorer, or
- Run in PowerShell:
  ```powershell
  Start-Process "leaderboard/index.html"
  ```

### Option 3: Node `npx serve`
```bash
npx serve leaderboard
```

---

## Data Provenance
- **Models 1–4:** Experimentally measured end-to-end using local Python scoring pipelines with Hungarian optimal assignment and 0.25s collar.
- **Models 5–10:** Architecture-informed illustrative estimates calibrated on CALLHOME, AMI, and VoxConverse benchmarks. Clearly marked with the `Estimated` badge.
