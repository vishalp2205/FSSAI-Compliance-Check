/**
 * FSSAI Label Auditor — Main Application (Frontend + Multimodal Integration)
 * 
 * Orchestrates:
 *  - File upload & drag-drop
 *  - Visual bounding-box overlays for YOLO symbols and EasyOCR text
 *  - Backend API calls (FastAPI: YOLO11n + EasyOCR + ChromaDB RAG + 10-rule Expert Engine)
 *  - Structured extraction schema rendering (§5.3 Listing 1)
 *  - Historical audit trail browsing (FR8)
 *  - Client-side fallback to Tesseract.js with 10-rule parity
 */

import Tesseract from 'tesseract.js';
import { extractFields, runComplianceAudit } from './compliance-engine.js';

// ---- Config ----
const API_BASE = import.meta.env.VITE_API_BASE ?? (import.meta.env.PROD ? '' : 'http://localhost:8000');

// ---- DOM References ----
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewArea = document.getElementById('preview-area');
const previewImg = document.getElementById('preview-img');
const bboxCanvas = document.getElementById('bbox-canvas');
const clearBtn = document.getElementById('clear-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsPanel = document.getElementById('results-panel');
const processingOverlay = document.getElementById('processing-overlay');
const processingTitle = document.getElementById('processing-title');
const processingStep = document.getElementById('processing-step');
const processingFill = document.getElementById('processing-fill');
const toggleYolo = document.getElementById('toggle-yolo');
const toggleOcr = document.getElementById('toggle-ocr');

// Audit Modal DOM
const navHistoryBtn = document.getElementById('nav-history-btn');
const auditModal = document.getElementById('audit-modal');
const closeModalBtn = document.getElementById('close-modal-btn');
const modalBody = document.getElementById('modal-body');

const loadSampleBtn = document.getElementById('load-sample-btn');

let currentFile = null;
let backendAvailable = false;
let yoloModelLoaded = false;
let lastAnalysisData = null;
let backendPoller = null;

// ============================================
// BACKEND HEALTH & MODEL STATUS CHECK
// ============================================

async function checkBackend() {
  const badge = document.querySelector('.nav-badge');
  try {
    const res = await fetch(`${API_BASE}/api/status`, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    backendAvailable = data.ready;
    yoloModelLoaded = data.yolo_model_loaded;

    if (backendAvailable) {
      if (yoloModelLoaded) {
        badge.textContent = 'YOLO11n + RAG ACTIVE';
        badge.title = 'YOLO11n model loaded from backend/models/yolo_fssai.pt';
        badge.style.background = 'linear-gradient(135deg, #10b981, #06b6d4)';
      } else {
        badge.textContent = 'BACKEND (OPENCV)';
        badge.title = 'Backend online with OpenCV symbol detection';
        badge.style.background = 'linear-gradient(135deg, #3b82f6, #06b6d4)';
      }
    }
  } catch {
    backendAvailable = false;
    yoloModelLoaded = false;
    badge.textContent = 'CONNECTING TO BACKEND...';
    badge.title = 'Connecting to Python FastAPI server on localhost:8000';
    badge.style.background = 'linear-gradient(135deg, #f59e0b, #ef4444)';
  }

  return backendAvailable;
}

// Initial check + automatic retry polling
checkBackend();
backendPoller = setInterval(async () => {
  const isUp = await checkBackend();
  if (isUp) {
    clearInterval(backendPoller);
    // Continue periodic check every 15 seconds
    setInterval(checkBackend, 15000);
  }
}, 3000);

// ============================================
// LOAD SAMPLE TEST LABEL
// ============================================

if (loadSampleBtn) {
  loadSampleBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    try {
      const resp = await fetch('/test_label.jpg');
      if (!resp.ok) throw new Error('Could not fetch test_label.jpg');
      const blob = await resp.blob();
      const file = new File([blob], 'test_label.jpg', { type: 'image/jpeg' });
      handleFile(file);
    } catch (err) {
      console.error('Failed to load sample test label:', err);
      alert('Could not load sample test_label.jpg. Make sure test_label.jpg exists in the public directory.');
    }
  });
}

// ============================================
// FILE UPLOAD & DRAG/DROP
// ============================================

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0 && files[0].type.startsWith('image/')) {
    handleFile(files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleFile(e.target.files[0]);
  }
});

clearBtn.addEventListener('click', () => {
  currentFile = null;
  lastAnalysisData = null;
  previewImg.src = '';
  clearCanvas();
  previewArea.classList.add('hidden');
  dropZone.style.display = '';
  fileInput.value = '';
  resetResults();
});

function handleFile(file) {
  currentFile = file;
  lastAnalysisData = null;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.onload = () => {
      clearCanvas();
    };
    previewImg.src = e.target.result;
    previewArea.classList.remove('hidden');
    dropZone.style.display = 'none';
    resetResults();
  };
  reader.readAsDataURL(file);
}

function resetResults() {
  resultsPanel.innerHTML = `
    <div class="results-placeholder">
      <svg viewBox="0 0 64 64" fill="none" width="64" height="64" opacity="0.3">
        <rect x="8" y="8" width="48" height="48" rx="8" stroke="currentColor" stroke-width="1.5"/>
        <line x1="16" y1="22" x2="48" y2="22" stroke="currentColor" stroke-width="1.5"/>
        <line x1="16" y1="30" x2="40" y2="30" stroke="currentColor" stroke-width="1.5"/>
        <line x1="16" y1="38" x2="44" y2="38" stroke="currentColor" stroke-width="1.5"/>
        <line x1="16" y1="46" x2="36" y2="46" stroke="currentColor" stroke-width="1.5"/>
      </svg>
      <p>Audit results will appear here</p>
      <span>Upload and run audit to view compliance verdicts</span>
    </div>
  `;
}

// ============================================
// BOUNDING BOX CANVAS DRAWING
// ============================================

function clearCanvas() {
  if (!bboxCanvas) return;
  const ctx = bboxCanvas.getContext('2d');
  ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
}

function drawBoundingBoxes() {
  if (!bboxCanvas || !previewImg || !lastAnalysisData) return;
  if (!previewImg.naturalWidth || !previewImg.clientWidth) return;

  // Align canvas exactly over the rendered image element inside the container
  bboxCanvas.style.left = `${previewImg.offsetLeft}px`;
  bboxCanvas.style.top = `${previewImg.offsetTop}px`;
  bboxCanvas.style.width = `${previewImg.clientWidth}px`;
  bboxCanvas.style.height = `${previewImg.clientHeight}px`;

  // Canvas internal bitmap uses natural image resolution for 1:1 crisp precision
  bboxCanvas.width = previewImg.naturalWidth;
  bboxCanvas.height = previewImg.naturalHeight;

  const ctx = bboxCanvas.getContext('2d');
  ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);

  // 1. Draw OCR text boxes if toggled
  if (toggleOcr && toggleOcr.checked && lastAnalysisData.ocr_output) {
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)';
    ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';

    for (const item of lastAnalysisData.ocr_output) {
      if (item.bbox && item.bbox.length >= 4) {
        ctx.beginPath();
        ctx.moveTo(item.bbox[0][0], item.bbox[0][1]);
        for (let i = 1; i < item.bbox.length; i++) {
          ctx.lineTo(item.bbox[i][0], item.bbox[i][1]);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }
  }

  // 2. Draw YOLO Symbols if toggled
  if (toggleYolo && toggleYolo.checked && lastAnalysisData.symbol_detection) {
    const sd = lastAnalysisData.symbol_detection;

    const symbols = [
      { key: 'veg_mark', name: 'Veg Mark', color: '#10b981', bg: 'rgba(16, 185, 129, 0.95)' },
      { key: 'non_veg_mark', name: 'Non-Veg Mark', color: '#b45309', bg: 'rgba(180, 83, 9, 0.95)' },
      { key: 'fssai_logo', name: 'FSSAI Logo', color: '#06b6d4', bg: 'rgba(6, 182, 212, 0.95)' },
    ];

    for (const sym of symbols) {
      const det = sd[sym.key];
      if (det && det.detected && det.bbox) {
        const [x1, y1, x2, y2] = det.bbox;
        const bw = x2 - x1;
        const bh = y2 - y1;

        ctx.lineWidth = 4;
        ctx.strokeStyle = sym.color;
        ctx.strokeRect(x1, y1, bw, bh);

        // Semi-transparent fill
        ctx.fillStyle = `${sym.color}25`;
        ctx.fillRect(x1, y1, bw, bh);

        // Label Tag
        const label = `${sym.name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 22px Inter, sans-serif';
        const textWidth = ctx.measureText(label).width;
        const tagHeight = 32;
        const tagY = Math.max(0, y1 - tagHeight);

        ctx.fillStyle = sym.bg;
        ctx.fillRect(x1, tagY, textWidth + 18, tagHeight);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x1 + 9, tagY + 23);
      }
    }
  }
}

if (toggleYolo) toggleYolo.addEventListener('change', drawBoundingBoxes);
if (toggleOcr) toggleOcr.addEventListener('change', drawBoundingBoxes);
window.addEventListener('resize', drawBoundingBoxes);

// ============================================
// AUDIT RUNNER
// ============================================

analyzeBtn.addEventListener('click', async () => {
  if (!currentFile) return;

  analyzeBtn.disabled = true;
  showProcessing();

  await checkBackend();

  try {
    if (backendAvailable) {
      await analyzeWithBackend();
    } else {
      await analyzeWithFrontend();
    }
  } catch (err) {
    console.error('Audit failed:', err);
    resultsPanel.innerHTML = `
      <div class="results-placeholder">
        <svg viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="1.5" width="48" height="48">
          <circle cx="12" cy="12" r="10"/>
          <line x1="15" y1="9" x2="9" y2="15"/>
          <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        <p style="color:#ef4444">Analysis Failed</p>
        <span>${escapeHtml(err.message || 'An unknown error occurred.')}</span>
      </div>
    `;
  } finally {
    hideProcessing();
    analyzeBtn.disabled = false;
  }
});

// ============================================
// BACKEND PIPELINE (YOLO11n + OCR + RAG)
// ============================================

async function analyzeWithBackend() {
  updateProcessing('Connecting Pipeline...', 'Sending label to Python backend (YOLO11n + EasyOCR + RAG)', 15);

  const formData = new FormData();
  formData.append('file', currentFile);

  updateProcessing('Running Inference...', 'CLAHE Preprocessing → OCR → YOLO11n → Deterministic Rules → ChromaDB RAG', 45);

  const response = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Backend error: ${response.status}`);
  }

  updateProcessing('Finalizing Report...', 'Parsing structured extraction record and clause citations', 85);
  const data = await response.json();

  if (!data.success) {
    throw new Error('Analysis pipeline failed on server');
  }

  lastAnalysisData = data;
  updateProcessing('Rendering Results...', 'Building compliance report & bounding box overlays', 98);
  await sleep(150);

  renderBackendResults(data);
  drawBoundingBoxes();
}

// ============================================
// FRONTEND FALLBACK (Tesseract.js + 10-Rule Parity)
// ============================================

async function analyzeWithFrontend() {
  updateProcessing('Preprocessing Image...', 'Contrast enhancement for browser OCR', 15);
  const preprocessedImageData = await preprocessImageClient(currentFile);

  updateProcessing('Extracting Text...', 'Tesseract.js OCR in browser', 30);
  const result = await Tesseract.recognize(
    preprocessedImageData,
    'eng',
    {
      logger: (info) => {
        if (info.status === 'recognizing text') {
          const pct = Math.round(30 + info.progress * 45);
          updateProcessing('Recognizing Text...', `${Math.round(info.progress * 100)}%`, pct);
        }
      }
    }
  );

  const rawText = result.data.text;
  const confidence = result.data.confidence;

  updateProcessing('Executing Rules...', 'Evaluating 10 Statutory Rules (R01–R10)', 85);
  await sleep(250);

  const fields = extractFields(rawText);
  const auditResult = runComplianceAudit(fields);
  auditResult.ocr_confidence = confidence;

  updateProcessing('Rendering...', 'Generating report', 98);
  await sleep(150);

  renderFrontendResults(auditResult, rawText);
}

// ============================================
// RENDER BACKEND RESULTS (with Structured Record)
// ============================================

function renderBackendResults(data) {
  const audit = data.audit;
  const sr = data.structured_record || {};
  const ef = data.extracted_fields || {};
  const scoreClass = audit.overall_status === 'high' ? 'score-high' : audit.overall_status === 'mid' ? 'score-mid' : 'score-low';

  // 1. Structured Record (§5.3 Listing 1)
  const licVal = sr.license_no?.value || 'Not Detected';
  const licState = sr.license_no?.state_name || 'N/A';
  const licType = sr.license_no?.type || 'N/A';
  const dietClass = sr.dietary_mark?.class ? (sr.dietary_mark.class === 'veg' ? 'Vegetarian' : 'Non-Vegetarian') : 'Not Detected';
  const dietColorOk = sr.dietary_mark?.colour_ok ? '✅ Verified' : '⚠️ Unverified';
  const logoFound = sr.fssai_logo?.found ? `✅ Detected (${(sr.fssai_logo.conf * 100).toFixed(0)}%)` : '❌ Not Found';
  const dateVal = sr.date_marking?.value || 'Not Detected';
  const datePhrase = sr.date_marking?.phrase || 'Standard Date';
  const netQty = sr.net_quantity || 'Not Detected';
  const batchNo = sr.batch_number || 'Not Detected';

  // Allergens badges
  const allergensList = ef.allergens?.found || [];
  const allergenBadges = allergensList.length > 0
    ? allergensList.map(a => `<span class="tag-badge">⚠️ ${escapeHtml(a)}</span>`).join('')
    : '<span style="color: var(--text-muted); font-size: 0.78rem;">No mandatory allergens detected</span>';

  // Nutrition checklist
  const nutri = sr.nutrition || {};
  const nutriItems = [
    { key: 'energy', label: 'Energy' },
    { key: 'protein', label: 'Protein' },
    { key: 'carbohydrate', label: 'Carbs' },
    { key: 'sugar', label: 'Sugar' },
    { key: 'fat', label: 'Fat' },
    { key: 'sodium', label: 'Sodium' },
  ].map(item => {
    const checked = nutri[item.key];
    return `<div class="nutri-item ${checked ? 'checked' : ''}">
      <span>${checked ? '✅' : '⚪'}</span>
      <span>${item.label}</span>
    </div>`;
  }).join('');

  const structuredRecordHTML = `
    <div class="structured-record-card">
      <div class="structured-record-header">
        <h4>📋 Structured Information Record</h4>
        <span class="record-badge">Canonical JSON Schema</span>
      </div>
      <div class="record-grid">
        <div class="record-cell">
          <div class="record-cell-label">License (R01–R03)</div>
          <div class="record-cell-val">${escapeHtml(licVal)}</div>
          <div class="record-cell-sub">${escapeHtml(licType)} · State: ${escapeHtml(licState)}</div>
        </div>
        <div class="record-cell">
          <div class="record-cell-label">Dietary Symbol (R05–R07)</div>
          <div class="record-cell-val">${escapeHtml(dietClass)}</div>
          <div class="record-cell-sub">Colour: ${dietColorOk} (YOLO11n)</div>
        </div>
        <div class="record-cell">
          <div class="record-cell-label">FSSAI Logo (R04)</div>
          <div class="record-cell-val">${logoFound}</div>
          <div class="record-cell-sub">Perception: ${data.symbol_detection?.method || 'vision'}</div>
        </div>
        <div class="record-cell">
          <div class="record-cell-label">Date Marking (R09)</div>
          <div class="record-cell-val">${escapeHtml(dateVal)}</div>
          <div class="record-cell-sub">${escapeHtml(datePhrase)}</div>
        </div>
        <div class="record-cell">
          <div class="record-cell-label">Net Quantity & Batch (Reg. 9 & 3h)</div>
          <div class="record-cell-val">Qty: ${escapeHtml(netQty)}</div>
          <div class="record-cell-sub">Batch: ${escapeHtml(batchNo)}</div>
        </div>
        <div class="record-cell">
          <div class="record-cell-label">Declared Allergens (R08)</div>
          <div class="tag-list">${allergenBadges}</div>
        </div>
        <div class="record-cell" style="grid-column: 1 / -1;">
          <div class="record-cell-label">Nutritional Information Panel (R10)</div>
          <div class="nutri-checklist">${nutriItems}</div>
        </div>
      </div>
    </div>
  `;

  // Document discrepancy check (e.g. random paper or non-food upload)
  const hasNutriValues = Object.values(sr.nutrition || {}).filter(Boolean).length > 0;
  const isNonFoodOrIncomplete = (!sr.license_no?.value && !sr.dietary_mark?.class && !sr.fssai_logo?.found && !hasNutriValues);
  const nonFoodBannerHTML = isNonFoodOrIncomplete
    ? `
      <div class="non-food-warning-banner">
        <span class="non-food-badge-icon">⚠️</span>
        <div class="non-food-badge-text">
          <strong>Document Discrepancy Notice:</strong>
          <span>No statutory food labeling elements (FSSAI License, Veg/Non-Veg Mark, or Nutrition Panel) were detected. This image appears to be a non-food document (such as a question paper or generic text) or incomplete packaging. System fault-tolerance has prevented false positive compliance declarations.</span>
        </div>
      </div>
    `
    : '';

  // 1b. Indian Nutrition Rating (INR) & HFSS Card
  let inrHTML = '';
  if (data.inr_analysis) {
    const inr = data.inr_analysis;
    if (inr.available && inr.star_rating !== null && inr.star_rating !== undefined) {
      const fullStars = Math.floor(inr.star_rating);
      const hasHalf = inr.star_rating % 1 !== 0;
      const starIcons = '★'.repeat(fullStars) + (hasHalf ? '½' : '') + '☆'.repeat(Math.max(0, 5 - fullStars - (hasHalf ? 1 : 0)));

      const tl = inr.traffic_lights || {};
      const tlPills = [
        { key: 'sugar', label: 'Sugars', item: tl.sugar },
        { key: 'saturated_fat', label: 'Sat. Fat', item: tl.saturated_fat },
        { key: 'sodium', label: 'Sodium', item: tl.sodium },
      ].map(p => {
        const cls = p.item?.tier === 'red' ? 'tier-red' : p.item?.tier === 'amber' ? 'tier-amber' : 'tier-green';
        return `<div class="tl-pill ${cls}">
          <span class="tl-dot"></span>
          <span class="tl-name">${p.label}:</span>
          <span class="tl-val">${escapeHtml(p.item?.value || 'N/A')}</span>
          <span class="tl-tier">(${escapeHtml(p.item?.label || '')})</span>
        </div>`;
      }).join('');

      const alertBanner = inr.greenwashing_alert
        ? `<div class="inr-alert-banner">⚠️ <strong>Deceptive Marketing Risk:</strong> ${escapeHtml(inr.greenwashing_alert)}</div>`
        : '';

      inrHTML = `
        <div class="inr-card">
          <div class="inr-header">
            <div>
              <h4>⭐ Indian Nutrition Rating (INR) & Public Health Profiler</h4>
              <span class="inr-sub">FSSAI Front-of-Pack Nutrition Rating (FOPNL) · ML Regressor (R² = 0.924)</span>
            </div>
            <div class="inr-rating-box">
              <div class="inr-stars">${starIcons}</div>
              <div class="inr-score">${inr.star_rating} / 5.0 Stars</div>
            </div>
          </div>
          <div class="inr-traffic-lights">
            ${tlPills}
          </div>
          ${alertBanner}
        </div>
      `;
    } else {
      inrHTML = `
        <div class="inr-card inr-unavailable">
          <div class="inr-header">
            <div>
              <h4 style="color: var(--text-secondary);">⭐ Indian Nutrition Rating (INR) & Public Health Profiler</h4>
              <span class="inr-sub">FSSAI Front-of-Pack Nutrition Rating (FOPNL) · Fault-Tolerant Engine</span>
            </div>
            <div class="inr-na-badge">Not Applicable · No Nutrition Table</div>
          </div>
          <div class="inr-unavailable-body">
            <span class="inr-na-icon">📋</span>
            <div class="inr-na-text">
              <strong>No Statutory Nutritional Panel Detected on Document</strong>
              <p>Official FSSAI Indian Nutrition Rating (0.5 to 5.0 Stars) and High in Fat, Sugar, and Salt (HFSS) thresholds are computed exclusively from verified quantitative declarations (Energy, Sugars, Saturated Fat, Sodium per 100g) under FSSAI Regulation 8. Because this document lacks a verified nutritional panel, ratings are withheld to prevent false compliance evaluations.</p>
            </div>
          </div>
        </div>
      `;
    }
  }

  // 1c. Claims Verification & Greenwashing Detection Card
  let claimsHTML = '';
  if (data.claims_analysis && data.claims_analysis.results && data.claims_analysis.results.length > 0) {
    const claims = data.claims_analysis.results;
    const claimRows = claims.map(c => {
      const isVerified = c.status === 'verified';
      const badgeCls = isVerified ? 'badge-verified' : 'badge-misleading';
      return `
        <div class="claim-item">
          <div class="claim-top">
            <span class="claim-title">${escapeHtml(c.title)}</span>
            <span class="claim-badge ${badgeCls}">${isVerified ? '✅ Legitimate Claim' : '🚨 Deceptive / Misleading (Greenwashing)'}</span>
          </div>
          <div class="claim-criteria">📐 <strong>FSSAI Rule:</strong> ${escapeHtml(c.statutory_criteria)}</div>
          <div class="claim-models">
            <span class="model-tag tag-ens">ML Ensemble: <strong>${c.ensemble_pred} (${(c.ensemble_confidence * 100).toFixed(0)}%)</strong></span>
            <span class="model-tag tag-svm">SVM Classifier: <strong>${c.svm_pred}</strong></span>
            <span class="model-tag tag-dt">Decision Tree: <strong>${c.dt_pred}</strong></span>
          </div>
          <div class="claim-citation">📖 <em>${escapeHtml(c.citation)}</em></div>
        </div>
      `;
    }).join('');

    claimsHTML = `
      <div class="claims-card">
        <div class="claims-header">
          <div>
            <h4>🏷️ FSSAI Nutritional Claims & Greenwashing Verification</h4>
            <span class="claims-sub">Automated Claim Validation & Greenwashing Detection · Verified against FSSAI Advertising & Claims Reg. 2018/2022</span>
          </div>
        </div>
        <div class="claims-list">
          ${claimRows}
        </div>
      </div>
    `;
  }

  // 2. Checks R01–R10
  let checksHTML = '';
  for (const check of audit.checks) {
    const statusIcon = getStatusIcon(check.status);
    const statusClass = check.status === 'pass' ? 'status-pass' : check.status === 'fail' ? 'status-fail' : 'status-warn';
    const verdictClass = check.status === 'pass' ? 'verdict-pass' : check.status === 'fail' ? 'verdict-fail' : 'verdict-warn';

    const ragSection = check.section ? `<div class="check-rag-section">📚 ${escapeHtml(check.section)}</div>` : '';
    const ragExplanation = check.explanation ? `<div class="check-explanation">${escapeHtml(check.explanation)}</div>` : '';

    let ragChunksHTML = '';
    if (check.rag_chunks && check.rag_chunks.length > 0) {
      const chunkBadges = check.rag_chunks
        .map((c, i) => `<span class="rag-chunk-badge" title="Distance: ${c.distance?.toFixed(3) || 'N/A'}">Clause ${i + 1}: ${escapeHtml(c.subsection || c.section || 'FSSAI 2020')}</span>`)
        .join('');
      ragChunksHTML = `<div class="rag-chunks-row">🔍 RAG Retrieved: ${chunkBadges}</div>`;
    }

    checksHTML += `
      <div class="check-item">
        <div class="check-status ${statusClass}">${statusIcon}</div>
        <div class="check-body">
          <div class="check-title">
            <span class="rule-id-chip">${check.rule_id || check.id.toUpperCase()}</span>
            ${escapeHtml(check.title)}
            <span class="check-verdict ${verdictClass}">${check.verdict}</span>
          </div>
          <div class="check-detail">${escapeHtml(check.detail)}</div>
          ${check.extracted ? `<div class="check-extracted">Extracted: ${escapeHtml(check.extracted)}</div>` : ''}
          ${ragExplanation}
          ${check.regulation ? `<div class="check-regulation">📖 ${escapeHtml(check.regulation.substring(0, 300))}${check.regulation.length > 300 ? '...' : ''}</div>` : ''}
          ${check.fix ? `<div class="check-regulation" style="border-left-color: var(--accent-amber);">🔧 Recommended Fix: ${escapeHtml(check.fix)}</div>` : ''}
          ${ragChunksHTML}
        </div>
      </div>
    `;
  }

  // 3. Pipeline Timing
  let pipelineHTML = '';
  if (data.pipeline && data.pipeline.length > 0) {
    const steps = data.pipeline.map(s => `
      <div class="pipeline-step-row">
        <span class="pipeline-step-name">${escapeHtml(s.step)}</span>
        <span class="pipeline-step-time">${s.duration_ms}ms</span>
        <span class="pipeline-step-detail">${escapeHtml(s.detail)}</span>
      </div>
    `).join('');

    pipelineHTML = `
      <div class="result-pipeline-section">
        <button class="ocr-toggle-btn" id="pipeline-toggle">
          <span>⚡ Pipeline Stages & Latency (${data.total_time_ms}ms total)</span>
          <span class="arrow">▼</span>
        </button>
        <div class="ocr-raw-text" id="pipeline-detail">
          ${steps}
        </div>
      </div>
    `;
  }

  // Symbol summary
  const sd = data.symbol_detection || {};
  const vegStatus = sd.veg_mark?.detected ? `✅ Detected (${(sd.veg_mark.confidence * 100).toFixed(0)}%)` : '❌ Not Found';
  const nonVegStatus = sd.non_veg_mark?.detected ? `✅ Detected (${(sd.non_veg_mark.confidence * 100).toFixed(0)}%)` : '❌ Not Found';
  const logoStatus = sd.fssai_logo?.detected ? `✅ Detected (${(sd.fssai_logo.confidence * 100).toFixed(0)}%)` : '❌ Not Found';

  resultsPanel.innerHTML = `
    <div class="results-content">
      <div class="result-summary">
        <div class="summary-score ${scoreClass}">${audit.score}%</div>
        <div class="summary-info">
          <h3>${audit.overall_label}</h3>
          <p>${audit.pass_count} Passed · ${audit.fail_count} Failed · ${audit.review_count || audit.warn_count} Human Review — OCR Conf: ${Math.round(audit.ocr_confidence)}%</p>
          <p style="font-size: 0.75rem; color: var(--accent-cyan); margin-top: 0.25rem;">
            🟢 Perception: EasyOCR + ${sd.method?.toUpperCase() || 'YOLO11n'} · Reasoning: 10 Rules + ChromaDB RAG · ${data.total_time_ms}ms
          </p>
        </div>
      </div>

      <div class="action-export-bar">
        <button id="download-report-btn" class="btn btn-sm btn-ghost">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Export Full Report (JSON)
        </button>
        <span style="font-size: 0.76rem; color: var(--text-muted);">Timestamped & Logged to audit_log.jsonl (FR8)</span>
      </div>

      ${nonFoodBannerHTML}
      ${structuredRecordHTML}
      ${inrHTML}
      ${claimsHTML}

      <div class="result-checks">${checksHTML}</div>

      <div class="result-ocr-section">
        <button class="ocr-toggle-btn" id="symbol-toggle">
          <span>🔍 Symbol Detection Details (${sd.method || 'yolo11n'})</span>
          <span class="arrow">▼</span>
        </button>
        <div class="ocr-raw-text" id="symbol-detail">Veg Mark: ${vegStatus}\nNon-Veg Mark: ${nonVegStatus}\nFSSAI Logo: ${logoStatus}</div>
      </div>

      ${pipelineHTML}

      <div class="result-ocr-section">
        <button class="ocr-toggle-btn" id="ocr-toggle">
          <span>📝 View Raw OCR Text Output</span>
          <span class="arrow">▼</span>
        </button>
        <div class="ocr-raw-text" id="ocr-raw-text">${escapeHtml(data.raw_text || 'No text extracted')}</div>
      </div>
    </div>
  `;

  // Attach toggle and export listeners
  setupToggle('ocr-toggle', 'ocr-raw-text');
  setupToggle('symbol-toggle', 'symbol-detail');
  setupToggle('pipeline-toggle', 'pipeline-detail');

  const downloadReportBtn = document.getElementById('download-report-btn');
  if (downloadReportBtn) {
    downloadReportBtn.addEventListener('click', () => {
      exportCurrentAuditAsJson(data);
    });
  }

  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================
// RENDER FRONTEND RESULTS (Fallback mode)
// ============================================

function renderFrontendResults(audit, rawText) {
  const scoreClass = audit.overall_status === 'high' ? 'score-high' : audit.overall_status === 'mid' ? 'score-mid' : 'score-low';

  let checksHTML = '';
  for (const check of audit.checks) {
    const statusIcon = getStatusIcon(check.status);
    const statusClass = check.status === 'pass' ? 'status-pass' : check.status === 'fail' ? 'status-fail' : 'status-warn';
    const verdictClass = check.status === 'pass' ? 'verdict-pass' : check.status === 'fail' ? 'verdict-fail' : 'verdict-warn';

    checksHTML += `
      <div class="check-item">
        <div class="check-status ${statusClass}">${statusIcon}</div>
        <div class="check-body">
          <div class="check-title">
            <span class="rule-id-chip">${check.rule_id}</span>
            ${check.title}
            <span class="check-verdict ${verdictClass}">${check.verdict}</span>
          </div>
          <div class="check-detail">${check.detail}</div>
          ${check.extracted ? `<div class="check-extracted">Extracted: ${escapeHtml(check.extracted)}</div>` : ''}
          ${check.regulation ? `<div class="check-regulation">📖 ${check.regulation}</div>` : ''}
          ${check.fix ? `<div class="check-regulation" style="border-left-color: var(--accent-amber);">🔧 Recommended Fix: ${check.fix}</div>` : ''}
        </div>
      </div>
    `;
  }

  resultsPanel.innerHTML = `
    <div class="results-content">
      <div class="result-summary">
        <div class="summary-score ${scoreClass}">${audit.score}%</div>
        <div class="summary-info">
          <h3>${audit.overall_label}</h3>
          <p>${audit.pass_count} Passed · ${audit.fail_count} Failed · ${audit.warn_count} Human Review (R01–R10)</p>
          <p style="font-size: 0.75rem; color: var(--accent-amber); margin-top: 0.25rem;">
            🟡 Client Mode: Tesseract.js (Start backend for YOLO11n + ChromaDB RAG)
          </p>
        </div>
      </div>
      <div class="result-checks">${checksHTML}</div>
      <div class="result-ocr-section">
        <button class="ocr-toggle-btn" id="ocr-toggle">
          <span>📝 View Raw OCR Text</span>
          <span class="arrow">▼</span>
        </button>
        <div class="ocr-raw-text" id="ocr-raw-text">${escapeHtml(rawText || 'No text')}</div>
      </div>
    </div>
  `;

  setupToggle('ocr-toggle', 'ocr-raw-text');
  resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================
// AUDIT LOGS MODAL (FR8)
// ============================================

if (navHistoryBtn) {
  navHistoryBtn.addEventListener('click', async () => {
    auditModal.classList.remove('hidden');
    modalBody.innerHTML = '<div class="loading-state">Fetching persistent audit trail...</div>';

    try {
      const res = await fetch(`${API_BASE}/api/audit-logs?limit=30`);
      if (!res.ok) throw new Error('Could not fetch logs');
      const data = await res.json();

      if (!data.logs || data.logs.length === 0) {
        modalBody.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 2rem;">No historical audits recorded yet. Run your first audit!</p>';
        return;
      }

      const logsHTML = data.logs.map(log => {
        const timeFormatted = new Date(log.timestamp).toLocaleString();
        const scoreColor = log.score >= 80 ? 'var(--accent-emerald)' : log.score >= 50 ? 'var(--accent-amber)' : 'var(--accent-rose)';

        const checksSummary = (log.checks || []).map(c => `
          <span class="${c.verdict === 'Compliant' ? 'log-badge-pass' : 'log-badge-fail'}">
            ${c.rule_id}: ${c.verdict === 'Compliant' ? 'PASS' : 'FLAG'}
          </span>
        `).join(' · ');

        return `
          <div class="audit-log-item">
            <div class="audit-log-meta">
              <span class="audit-log-time">📅 ${timeFormatted}</span>
              <span class="audit-log-score" style="color: ${scoreColor}">${log.score}% (${log.overall_label})</span>
            </div>
            <div class="audit-log-checks">${checksSummary}</div>
          </div>
        `;
      }).join('');

      modalBody.innerHTML = logsHTML;
    } catch (err) {
      modalBody.innerHTML = `<p style="color: var(--accent-rose); padding: 1rem;">Could not load logs: Ensure FastAPI server is running on localhost:8000.</p>`;
    }
  });
}

if (closeModalBtn) {
  closeModalBtn.addEventListener('click', () => {
    auditModal.classList.add('hidden');
  });
}



if (auditModal) {
  auditModal.addEventListener('click', (e) => {
    if (e.target === auditModal) {
      auditModal.classList.add('hidden');
    }
  });
}

// ============================================
// EXPORT REPORT FUNCTION
// ============================================

function exportCurrentAuditAsJson(data) {
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `fssai_audit_report_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ============================================
// IMAGE PREPROCESSING (Client Fallback)
// ============================================

function preprocessImageClient(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const maxDim = 2000;
      let w = img.width;
      let h = img.height;
      if (w > maxDim || h > maxDim) {
        const scale = maxDim / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(img, 0, 0, w, h);

      const imageData = ctx.getImageData(0, 0, w, h);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        const contrast = 1.4;
        const factor = (259 * (contrast * 128 + 255)) / (255 * (259 - contrast * 128));
        const enhanced = Math.min(255, Math.max(0, factor * (gray - 128) + 128));
        data[i] = enhanced;
        data[i + 1] = enhanced;
        data[i + 2] = enhanced;
      }
      ctx.putImageData(imageData, 0, 0);
      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

// ============================================
// PROCESSING OVERLAY UTILITIES
// ============================================

function showProcessing() {
  processingOverlay.classList.remove('hidden');
}

function hideProcessing() {
  processingOverlay.classList.add('hidden');
}

function updateProcessing(title, step, progress) {
  processingTitle.textContent = title;
  processingStep.textContent = step;
  processingFill.style.width = `${progress}%`;
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function setupToggle(btnId, contentId) {
  const btn = document.getElementById(btnId);
  const content = document.getElementById(contentId);
  if (btn && content) {
    btn.onclick = () => {
      btn.classList.toggle('open');
      content.classList.toggle('show');
    };
  }
}

function getStatusIcon(status) {
  if (status === 'pass') {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
  } else if (status === 'fail') {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
  } else {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}
