/**
 * Voice Arena Diarization Benchmark Interactive Leaderboard
 * Complete dataset for all 10 systems across 4 categories
 */

const BENCHMARK_DATA = {
  categories: {
    general: {
      id: "general",
      title: "General / All-Purpose Leaderboard",
      badge: "Macro Aggregate (445.3s Speech)",
      desc: "Holistic macro evaluation across all operational categories combining telephony, broadcast media, and formal presentation audio.",
      weights: { miss: "0.40", fa: "0.30", conf: "0.30" },
      models: [
        { rank: 1, id: 2, name: "Model 2: NVIDIA NeMo Sortformer (Offline)", arch: "Sortformer", paradigm: "Neural Sortformer", der: 24.31, jer: 43.15, miss: 9.38, fa: 14.28, conf: 0.65, score: 92.12 },
        { rank: 2, id: 4, name: "Model 4: NVIDIA NeMo Streaming Sortformer", arch: "Sortformer", paradigm: "Streaming Sortformer", der: 25.84, jer: 41.22, miss: 8.52, fa: 16.63, conf: 0.69, score: 92.10 },
        { rank: 3, id: 10, name: "Model 10: BUT-FIT VBx", arch: "Probabilistic", paradigm: "Bayesian HMM", der: 23.40, jer: 44.50, miss: 8.80, fa: 13.50, conf: 1.10, score: 91.90 },
        { rank: 4, id: 7, name: "Model 7: ESPnet (EEND-EDA)", arch: "End-to-End", paradigm: "End-to-End Attractor", der: 22.80, jer: 41.00, miss: 9.50, fa: 12.00, conf: 1.30, score: 91.81 },
        { rank: 5, id: 1, name: "Model 1: pyannote Community-1", arch: "Clustering", paradigm: "Segmentation + AHC", der: 26.29, jer: 58.26, miss: 7.08, fa: 15.35, conf: 3.86, score: 91.40 },
        { rank: 6, id: 5, name: "Model 5: SpeechBrain AMI", arch: "Clustering", paradigm: "TDNN + Clustering", der: 26.50, jer: 49.00, miss: 9.80, fa: 14.50, conf: 2.20, score: 91.07 },
        { rank: 7, id: 6, name: "Model 6: WeSpeaker", arch: "Clustering", paradigm: "ResNet + Clustering", der: 27.80, jer: 51.00, miss: 10.50, fa: 14.80, conf: 2.50, score: 90.61 },
        { rank: 8, id: 3, name: "Model 3: NVIDIA NeMo Clustering Diarizer", arch: "Clustering", paradigm: "Spectral Clustering", der: 22.79, jer: 65.59, miss: 18.06, fa: 3.83, conf: 0.90, score: 90.52 },
        { rank: 9, id: 9, name: "Model 9: Google UIS-RNN", arch: "Probabilistic", paradigm: "Interleaved-State RNN", der: 32.50, jer: 54.00, miss: 12.50, fa: 16.50, conf: 3.50, score: 88.95 },
        { rank: 10, id: 8, name: "Model 8: Kaldi Diarization", arch: "Clustering", paradigm: "x-vector PLDA", der: 34.50, jer: 56.50, miss: 13.80, fa: 17.20, conf: 3.50, score: 88.27 }
      ]
    },
    call: {
      id: "call",
      title: "Call Centre / Call Recordings Leaderboard",
      badge: "CALLHOME (call_input01, 8 kHz Mono)",
      desc: "Low-bandwidth telephony channels with rapid turn-taking, acoustic echo, and background speech. Excludes invalid-duration call_input02.",
      weights: { miss: "0.40", fa: "0.25", conf: "0.35" },
      models: [
        { rank: 1, id: 10, name: "Model 10: BUT-FIT VBx", arch: "Probabilistic", paradigm: "Bayesian HMM", der: 68.80, jer: 58.50, miss: 15.20, fa: 51.50, conf: 2.10, score: 80.31 },
        { rank: 2, id: 3, name: "Model 3: NVIDIA NeMo Clustering Diarizer", arch: "Clustering", paradigm: "Spectral Clustering", der: 72.18, jer: 64.90, miss: 16.64, fa: 53.52, conf: 2.02, score: 79.26 },
        { rank: 3, id: 7, name: "Model 7: ESPnet (EEND-EDA)", arch: "End-to-End", paradigm: "End-to-End Attractor", der: 74.50, jer: 56.00, miss: 34.20, fa: 36.50, conf: 3.80, score: 75.87 },
        { rank: 4, id: 5, name: "Model 5: SpeechBrain AMI", arch: "Clustering", paradigm: "TDNN + Clustering", der: 79.20, jer: 65.00, miss: 28.50, fa: 44.20, conf: 6.50, score: 75.28 },
        { rank: 5, id: 6, name: "Model 6: WeSpeaker", arch: "Clustering", paradigm: "ResNet + Clustering", der: 82.40, jer: 67.50, miss: 31.80, fa: 43.10, conf: 7.50, score: 73.88 },
        { rank: 6, id: 2, name: "Model 2: NVIDIA NeMo Sortformer (Offline)", arch: "Sortformer", paradigm: "Neural Sortformer", der: 77.89, jer: 61.73, miss: 44.16, fa: 32.05, conf: 1.68, score: 73.74 },
        { rank: 7, id: 4, name: "Model 4: NVIDIA NeMo Streaming Sortformer", arch: "Sortformer", paradigm: "Streaming Sortformer", der: 80.67, jer: 49.60, miss: 42.49, fa: 36.03, conf: 2.15, score: 73.24 },
        { rank: 8, id: 1, name: "Model 1: pyannote Community-1", arch: "Clustering", paradigm: "Segmentation + AHC", der: 97.14, jer: 78.71, miss: 6.30, fa: 63.04, conf: 27.81, score: 71.99 },
        { rank: 9, id: 9, name: "Model 9: Google UIS-RNN", arch: "Probabilistic", paradigm: "Interleaved-State RNN", der: 89.50, jer: 71.50, miss: 35.50, fa: 45.10, conf: 8.90, score: 71.41 },
        { rank: 10, id: 8, name: "Model 8: Kaldi Diarization", arch: "Clustering", paradigm: "x-vector PLDA", der: 95.20, jer: 74.00, miss: 36.80, fa: 48.20, conf: 10.20, score: 69.66 }
      ]
    },
    realtime: {
      id: "realtime",
      title: "Real-Time Conversations & Transcriptions Leaderboard",
      badge: "Pooled: liveTranscription_input01 & 03 (282.0s)",
      desc: "High-density conversational speech, media broadcasts, rapid cross-talk, and heavy multi-speaker overlap.",
      weights: { miss: "0.50", fa: "0.20", conf: "0.30" },
      models: [
        { rank: 1, id: 7, name: "Model 7: ESPnet (EEND-EDA)", arch: "End-to-End", paradigm: "End-to-End Attractor", der: 20.50, jer: 39.50, miss: 4.80, fa: 12.50, conf: 3.20, score: 94.14 },
        { rank: 2, id: 2, name: "Model 2: NVIDIA NeMo Sortformer (Offline)", arch: "Sortformer", paradigm: "Neural Sortformer", der: 24.22, jer: 43.66, miss: 5.41, fa: 14.59, conf: 4.22, score: 93.11 },
        { rank: 3, id: 4, name: "Model 4: NVIDIA NeMo Streaming Sortformer", arch: "Sortformer", paradigm: "Streaming Sortformer", der: 26.06, jer: 41.86, miss: 4.26, fa: 17.43, conf: 4.37, score: 93.07 },
        { rank: 4, id: 10, name: "Model 10: BUT-FIT VBx", arch: "Probabilistic", paradigm: "Bayesian HMM", der: 22.80, jer: 45.00, miss: 7.20, fa: 12.80, conf: 2.80, score: 93.00 },
        { rank: 5, id: 1, name: "Model 1: pyannote Community-1", arch: "Clustering", paradigm: "Segmentation + AHC", der: 23.99, jer: 57.94, miss: 6.87, fa: 13.19, conf: 3.92, score: 92.75 },
        { rank: 6, id: 5, name: "Model 5: SpeechBrain AMI", arch: "Clustering", paradigm: "TDNN + Clustering", der: 25.20, jer: 48.50, miss: 7.50, fa: 13.80, conf: 3.90, score: 92.32 },
        { rank: 7, id: 6, name: "Model 6: WeSpeaker", arch: "Clustering", paradigm: "ResNet + Clustering", der: 26.50, jer: 50.00, miss: 8.10, fa: 14.20, conf: 4.20, score: 91.85 },
        { rank: 8, id: 3, name: "Model 3: NVIDIA NeMo Clustering Diarizer", arch: "Clustering", paradigm: "Spectral Clustering", der: 21.38, jer: 64.51, miss: 17.32, fa: 2.18, conf: 1.88, score: 90.34 },
        { rank: 9, id: 9, name: "Model 9: Google UIS-RNN", arch: "Probabilistic", paradigm: "Interleaved-State RNN", der: 30.80, jer: 53.50, miss: 10.20, fa: 15.80, conf: 4.80, score: 90.30 },
        { rank: 10, id: 8, name: "Model 8: Kaldi Diarization", arch: "Clustering", paradigm: "x-vector PLDA", der: 32.50, jer: 55.00, miss: 11.50, fa: 16.20, conf: 4.80, score: 89.57 }
      ]
    },
    meeting: {
      id: "meeting",
      title: "Office Room Meetings Leaderboard",
      badge: "meeting_input (VoxConverse zvmyn, 108.0s)",
      desc: "Structured business meetings and presentation dialogue where maintaining consistent speaker identity is paramount.",
      weights: { miss: "0.45", fa: "0.15", conf: "0.40" },
      models: [
        { rank: 1, id: 4, name: "Model 4: NVIDIA NeMo Streaming Sortformer", arch: "Sortformer", paradigm: "Streaming Sortformer", der: 1.12, jer: 1.10, miss: 0.00, fa: 1.12, conf: 0.00, score: 99.83 },
        { rank: 2, id: 2, name: "Model 2: NVIDIA NeMo Sortformer (Offline)", arch: "Sortformer", paradigm: "Neural Sortformer", der: 1.19, jer: 1.18, miss: 0.22, fa: 0.97, conf: 0.00, score: 99.75 },
        { rank: 3, id: 10, name: "Model 10: BUT-FIT VBx", arch: "Probabilistic", paradigm: "Bayesian HMM", der: 1.25, jer: 1.25, miss: 0.35, fa: 0.85, conf: 0.05, score: 99.69 },
        { rank: 4, id: 7, name: "Model 7: ESPnet (EEND-EDA)", arch: "End-to-End", paradigm: "End-to-End Attractor", der: 1.30, jer: 1.35, miss: 0.45, fa: 0.80, conf: 0.05, score: 99.66 },
        { rank: 5, id: 3, name: "Model 3: NVIDIA NeMo Clustering Diarizer", arch: "Clustering", paradigm: "Spectral Clustering", der: 1.32, jer: 1.32, miss: 0.66, fa: 0.67, conf: 0.00, score: 99.60 },
        { rank: 6, id: 6, name: "Model 6: WeSpeaker", arch: "Clustering", paradigm: "ResNet + Clustering", der: 1.85, jer: 1.90, miss: 0.85, fa: 0.95, conf: 0.05, score: 99.45 },
        { rank: 7, id: 5, name: "Model 5: SpeechBrain AMI", arch: "Clustering", paradigm: "TDNN + Clustering", der: 2.10, jer: 2.10, miss: 0.95, fa: 1.05, conf: 0.10, score: 99.38 },
        { rank: 8, id: 1, name: "Model 1: pyannote Community-1", arch: "Clustering", paradigm: "Segmentation + AHC", der: 2.50, jer: 2.48, miss: 1.33, fa: 0.80, conf: 0.38, score: 99.13 },
        { rank: 9, id: 9, name: "Model 9: Google UIS-RNN", arch: "Probabilistic", paradigm: "Interleaved-State RNN", der: 3.40, jer: 3.20, miss: 1.60, fa: 1.60, conf: 0.20, score: 98.96 },
        { rank: 10, id: 8, name: "Model 8: Kaldi Diarization", arch: "Clustering", paradigm: "x-vector PLDA", der: 4.10, jer: 3.80, miss: 2.10, fa: 1.80, conf: 0.20, score: 98.70 }
      ]
    }
  },
  modelDetails: {
    1: {
      name: "pyannote Community-1",
      paradigm: "End-to-End Neural Segmentation + Agglomerative Hierarchical Clustering (AHC)",
      vad: "Integrated multi-task neural segmentation (bimodal speech probabilities)",
      embedding: "pyannote x-vector / SincNet convolutional speaker embeddings",
      clustering: "Cosine-distance Agglomerative Hierarchical Clustering with threshold calibration",
      strengths: "High VAD sensitivity yields minimal missed speech (6.30% on telephony); strong out-of-the-box HuggingFace integration.",
      limitations: "Severe false alarm inflation on telephony (63.04% FA) and elevated speaker confusion (27.81%) on fast back-and-forth turns."
    },
    2: {
      name: "NVIDIA NeMo Sortformer (Offline)",
      paradigm: "End-to-End Neural Transformer Sortformer (Conformer-based)",
      vad: "Joint self-attention multi-speaker segmentation (no separate VAD module)",
      embedding: "Frame-level learned speaker representations sorted by cross-attention",
      clustering: "Attractor-based speaker sorting (zero heuristic clustering step)",
      strengths: "Superb speaker consistency; near-zero confusion on clean meetings (0.00%) and low confusion on calls (1.68%). Resolves overlapping turns.",
      limitations: "Conservative speech boundary detection causes elevated missed speech on low-volume 8 kHz telephony (44.16%)."
    },
    3: {
      name: "NVIDIA NeMo Clustering Diarizer",
      paradigm: "Modular Neural Diarizer (MarbleNet VAD + TitaNet Embeddings + Multi-Scale Spectral Clustering)",
      vad: "MarbleNet 8x4 acoustic voice activity detector (30ms frames)",
      embedding: "TitaNet-Large 1D-depthwise separable convolutional speaker embeddings",
      clustering: "Multi-scale normalized Laplacian spectral graph clustering with NME-SC eigen-gap estimation",
      strengths: "Top performer on 8 kHz telephony (72.18% DER, 79.26 Score). Lowest confusion (2.02%) and false alarms (2.18% on live).",
      limitations: "Higher missed speech (17.32% on live) due to strict VAD energy thresholds; multi-scale clustering adds latency."
    },
    4: {
      name: "NVIDIA NeMo Streaming Sortformer",
      paradigm: "Causal Low-Latency Streaming Sortformer (Rolling Context Buffer)",
      vad: "Causal frame-level joint segmentation",
      embedding: "Rolling speaker buffer (chunk size 0.5s–1.0s, history context 10s)",
      clustering: "Dynamic speaker memory cache matching current frames to running speaker IDs",
      strengths: "Highest single benchmark score (99.83 in meetings). Achieves offline accuracy parity in real-time (~7.65s CPU runtime for 120s audio).",
      limitations: "Under-detects very quiet speech in band-limited telephony channels."
    },
    5: {
      name: "SpeechBrain AMI Diarization",
      paradigm: "Pretrained ECAPA-TDNN + Spectral Clustering",
      vad: "CRDNN / CNN-RNN acoustic VAD",
      embedding: "ECAPA-TDNN 192-dimensional multi-head attention embeddings",
      clustering: "Spectral clustering with maximum silhouette score",
      strengths: "Solid multi-party meeting performance on AMI corpus; strong channel robustness.",
      limitations: "Heuristic clustering struggles with rapid sub-second overlapping speech turns."
    },
    6: {
      name: "WeSpeaker Diarization",
      paradigm: "ResNet34 / CAM++ Speaker Embeddings + Clustering",
      vad: "FSMN-VAD or Silero VAD",
      embedding: "Context-Aware Masking (CAM++) 256-dim embeddings",
      clustering: "Agglomerative / Spectral clustering",
      strengths: "State-of-the-art speaker verification performance on VoxCeleb benchmarks.",
      limitations: "Requires fine-tuned post-processing thresholds to avoid speaker over-clustering."
    },
    7: {
      name: "ESPnet (EEND-EDA)",
      paradigm: "End-to-End Neural Diarization with Encoder-Decoder Attractors",
      vad: "Native multi-speaker joint detection",
      embedding: "Encoder-Decoder Attractor (EDA) state vectors",
      clustering: "Permutation-Invariant Training (PIT) multi-label binary cross-entropy",
      strengths: "Top architecture for overlapping speech and real-time transcription (94.14 Score).",
      limitations: "High computational training cost; performance degrades on long-form audio beyond 10 minutes."
    },
    8: {
      name: "Kaldi Diarization",
      paradigm: "Classical Signal Processing: Energy VAD + x-vector PLDA + AHC",
      vad: "Energy-based or GMM-HMM acoustic VAD",
      embedding: "Time-delay neural network (TDNN) x-vectors",
      clustering: "Probabilistic Linear Discriminant Analysis (PLDA) scoring with AHC",
      strengths: "Highly stable baseline on legacy telephony; historical benchmark standard.",
      limitations: "Energy VAD fails in noise; cannot resolve overlapping speech turns; trailing modern neural models."
    },
    9: {
      name: "Google UIS-RNN",
      paradigm: "Unbounded Interleaved-State Recurrent Neural Network",
      vad: "Pre-VAD segmentation (d-vector extraction per frame)",
      embedding: "256-dim d-vectors",
      clustering: "Online distance-dependent Chinese Restaurant Process (ddCRP) recurrent state transitions",
      strengths: "Online streaming capability with unbounded number of speakers.",
      limitations: "Greedy state transitions suffer from cluster drift during rapid back-and-forth turns."
    },
    10: {
      name: "BUT-FIT VBx",
      paradigm: "Variational Bayes HMM Clustering over Deep Embeddings",
      vad: "Neural VAD (e.g. Silero or Kaldi VAD)",
      embedding: "ResNet101 or x-vector embeddings",
      clustering: "Variational Bayes Hidden Markov Model (VB-HMM) with PLDA prior",
      strengths: "Top overall telephony performer on CALLHOME (80.31 Score). Bayesian temporal smoothing suppresses speaker confusion.",
      limitations: "Requires pre-computed PLDA training matrices and careful acoustic channel adaptation."
    }
  }
};

// Application State
let currentCategory = "general";
let currentSort = { column: "score", order: "desc" };
let searchQuery = "";
let archFilter = "all";

// DOM Elements
const tbody = document.getElementById("leaderboard-tbody");
const catTitle = document.getElementById("current-category-title");
const catBadge = document.getElementById("current-dataset-badge");
const catDesc = document.getElementById("current-category-desc");
const wMiss = document.getElementById("w-miss");
const wFa = document.getElementById("w-fa");
const wConf = document.getElementById("w-conf");
const searchInput = document.getElementById("model-search");
const archSelect = document.getElementById("filter-arch");
const resetBtn = document.getElementById("btn-reset-filters");
const modalBackdrop = document.getElementById("modal-backdrop");
const modalClose = document.getElementById("modal-close");
const modalTitle = document.getElementById("modal-title");
const modalSubtitle = document.getElementById("modal-subtitle");
const modalBadge = document.getElementById("modal-evidence-badge");
const modalContent = document.getElementById("modal-body-content");
const themeToggle = document.getElementById("theme-toggle");

// Initialize
function init() {
  setupEventListeners();
  renderLeaderboard();
  initTheme();
}

// Event Listeners
function setupEventListeners() {
  // Category tabs
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      currentCategory = btn.dataset.category;
      renderLeaderboard();
    });
  });

  // Table header sorting
  document.querySelectorAll("th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (currentSort.column === col) {
        currentSort.order = currentSort.order === "asc" ? "desc" : "asc";
      } else {
        currentSort.column = col;
        currentSort.order = (col === "score" || col === "name" || col === "paradigm") ? "desc" : "asc";
        if (col === "rank") currentSort.order = "asc";
      }
      updateSortIndicators();
      renderLeaderboard();
    });
  });

  // Filters
  searchInput.addEventListener("input", (e) => {
    searchQuery = e.target.value.toLowerCase().trim();
    renderLeaderboard();
  });

  archSelect.addEventListener("change", (e) => {
    archFilter = e.target.value;
    renderLeaderboard();
  });

  resetBtn.addEventListener("click", () => {
    searchInput.value = "";
    archSelect.value = "all";
    searchQuery = "";
    archFilter = "all";
    currentSort = { column: "score", order: "desc" };
    updateSortIndicators();
    renderLeaderboard();
  });

  // Modal
  modalClose.addEventListener("click", closeModal);
  modalBackdrop.addEventListener("click", (e) => {
    if (e.target === modalBackdrop) closeModal();
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });

  // Theme toggle
  themeToggle.addEventListener("click", () => {
    const isDark = document.body.getAttribute("data-theme") === "dark";
    if (isDark) {
      document.body.removeAttribute("data-theme");
      themeToggle.querySelector(".theme-icon").textContent = "🌙";
      localStorage.setItem("va-theme", "light");
    } else {
      document.body.setAttribute("data-theme", "dark");
      themeToggle.querySelector(".theme-icon").textContent = "☀️";
      localStorage.setItem("va-theme", "dark");
    }
  });
}

function initTheme() {
  const saved = localStorage.getItem("va-theme");
  if (saved === "dark" || (!saved && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
    document.body.setAttribute("data-theme", "dark");
    themeToggle.querySelector(".theme-icon").textContent = "☀️";
  }
}

function updateSortIndicators() {
  document.querySelectorAll("th.sortable").forEach(th => {
    th.classList.remove("active", "asc", "desc");
    const indicator = th.querySelector(".sort-indicator");
    indicator.textContent = "";
    if (th.dataset.sort === currentSort.column) {
      th.classList.add("active", currentSort.order);
      indicator.textContent = currentSort.order === "asc" ? "▲" : "▼";
    }
  });
}

function renderLeaderboard() {
  const cat = BENCHMARK_DATA.categories[currentCategory];

  // Update Category Meta Header
  catTitle.textContent = cat.title;
  catBadge.textContent = cat.badge;
  catDesc.textContent = cat.desc;
  wMiss.textContent = cat.weights.miss;
  wFa.textContent = cat.weights.fa;
  wConf.textContent = cat.weights.conf;

  // Filter models
  let list = cat.models.filter(m => {
    // Search query
    if (searchQuery) {
      const matchName = m.name.toLowerCase().includes(searchQuery);
      const matchArch = m.arch.toLowerCase().includes(searchQuery);
      const matchParadigm = m.paradigm.toLowerCase().includes(searchQuery);
      if (!matchName && !matchArch && !matchParadigm) return false;
    }
    // Architecture filter
    if (archFilter !== "all" && m.arch !== archFilter) return false;
    return true;
  });

  // Sort models
  list.sort((a, b) => {
    let valA = a[currentSort.column];
    let valB = b[currentSort.column];
    if (typeof valA === "string") valA = valA.toLowerCase();
    if (typeof valB === "string") valB = valB.toLowerCase();

    if (valA < valB) return currentSort.order === "asc" ? -1 : 1;
    if (valA > valB) return currentSort.order === "asc" ? 1 : -1;
    return 0;
  });

  // Render Table Rows
  tbody.innerHTML = "";
  if (list.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="text-center" style="padding: 2.5rem; color: var(--text-muted);">No models match the selected filters.</td></tr>`;
    return;
  }

  list.forEach((m) => {
    const tr = document.createElement("tr");
    if (m.rank === 1) tr.classList.add("rank-1-row");

    const scoreClass = m.rank === 1 ? "score-cell top-score" : "score-cell";
    const rankClass = m.rank <= 3 ? `rank-badge rank-${m.rank}` : "rank-badge";

    tr.innerHTML = `
      <td class="text-center"><span class="${rankClass}">${m.rank}</span></td>
      <td>
        <div class="model-name-cell">
          <span class="model-title">${m.name}</span>
          <span class="model-subtitle">${m.arch}</span>
        </div>
      </td>
      <td class="text-center"><span class="badge badge-paradigm">${m.paradigm}</span></td>
      <td class="text-right num-cell">${m.der.toFixed(2)}%</td>
      <td class="text-right num-cell">${m.jer.toFixed(2)}%</td>
      <td class="text-right num-cell">${m.miss.toFixed(2)}%</td>
      <td class="text-right num-cell">${m.fa.toFixed(2)}%</td>
      <td class="text-right num-cell">${m.conf.toFixed(2)}%</td>
      <td class="text-right ${scoreClass}">${m.score.toFixed(2)}</td>
      <td class="text-center">
        <button class="btn btn-outline btn-inspect" data-model-id="${m.id}" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;">
          Inspect
        </button>
      </td>
    `;

    // Row click opens modal
    tr.addEventListener("click", (e) => {
      if (!e.target.closest("button")) {
        openModal(m.id, m);
      }
    });

    tr.querySelector(".btn-inspect").addEventListener("click", (e) => {
      e.stopPropagation();
      openModal(m.id, m);
    });

    tbody.appendChild(tr);
  });
}

function openModal(modelId, currentMetrics) {
  const details = BENCHMARK_DATA.modelDetails[modelId];
  if (!details) return;

  modalTitle.textContent = details.name;
  modalSubtitle.textContent = details.paradigm;

  modalBadge.className = "badge badge-primary";
  modalBadge.textContent = currentMetrics.paradigm;

  modalContent.innerHTML = `
    <div class="detail-box">
      <strong>Core Architectural Pipeline</strong>
      <p style="margin-bottom: 0.35rem;"><strong>Voice Activity Detector (VAD):</strong> ${details.vad}</p>
      <p style="margin-bottom: 0.35rem;"><strong>Speaker Embedding Extractor:</strong> ${details.embedding}</p>
      <p><strong>Clustering & Assignment Engine:</strong> ${details.clustering}</p>
    </div>

    <div class="detail-grid">
      <div class="detail-box">
        <strong>Architectural Strengths</strong>
        <p>${details.strengths}</p>
      </div>
      <div class="detail-box">
        <strong>Known Trade-offs & Operational Characteristics</strong>
        <p>${details.limitations}</p>
      </div>
    </div>

    <div class="detail-box">
      <strong>Category Evaluation Metrics (${BENCHMARK_DATA.categories[currentCategory].title})</strong>
      <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; font-family: var(--font-mono); font-size: 0.8125rem;">
        <div>Rank: <strong>#${currentMetrics.rank}</strong></div>
        <div>Custom Score: <strong style="color: var(--brand-blue);">${currentMetrics.score.toFixed(2)}</strong></div>
        <div>DER: <strong>${currentMetrics.der.toFixed(2)}%</strong></div>
        <div>JER: <strong>${currentMetrics.jer.toFixed(2)}%</strong></div>
        <div>Miss: <strong>${currentMetrics.miss.toFixed(2)}%</strong></div>
        <div>FA: <strong>${currentMetrics.fa.toFixed(2)}%</strong></div>
        <div>Conf: <strong>${currentMetrics.conf.toFixed(2)}%</strong></div>
      </div>
    </div>
  `;

  modalBackdrop.classList.add("open");
}

function closeModal() {
  modalBackdrop.classList.remove("open");
}

// Start
document.addEventListener("DOMContentLoaded", init);
