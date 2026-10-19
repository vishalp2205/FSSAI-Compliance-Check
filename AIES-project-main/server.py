"""
FSSAI Label Auditor — FastAPI Backend Server

Endpoints:
  POST /api/analyze   — Full compliance audit (image upload)
  GET  /api/health    — Health check
  GET  /api/status    — System status (models loaded, etc.)

Pipeline:
  Image → Preprocessing (OpenCV) → OCR (EasyOCR) → Symbol Detection (OpenCV/YOLO)
  → Field Extraction (Regex) → Compliance Engine (Rules) → RAG Explanation (ChromaDB)
  → Audit Report
"""

import io
import os
import time
import traceback
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# ── Local modules ──
from backend.image_preprocessing import preprocess_for_ocr, preprocess_for_detection
from backend.symbol_detection import detect_symbols
from backend.compliance_engine import extract_fields, run_compliance_audit, build_structured_record
from backend.rag_pipeline import initialize_vector_store
from backend.claims_verifier import verify_claims, compute_inr_rating, get_ablation_benchmark

# ═══════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="FSSAI Label Auditor API",
    description="Explainable Multimodal RAG System for FSSAI Food Label Compliance Auditing",
    version="1.0.0",
)

# CORS — allow the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──
_ocr_reader = None
_system_ready = False


def _get_ocr_reader():
    """Lazy-load EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("[OCR] Loading EasyOCR model (English)...")
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("[OCR] EasyOCR loaded.")
    return _ocr_reader


# ═══════════════════════════════════════════════════════════════
# STARTUP EVENT
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    """Initialize models and vector store on server start."""
    global _system_ready
    print("\n" + "=" * 60)
    print("  FSSAI Label Auditor — Starting Up")
    print("=" * 60)

    try:
        # 1. Initialize RAG vector store (embed FSSAI regulations)
        print("\n[1/2] Initializing RAG Vector Store...")
        initialize_vector_store()

        # 2. Symbol detection is loaded on import
        print("\n[2/2] Symbol Detection ready.")

        # Note: OCR engine is lazy-loaded on first /api/analyze request
        # to avoid startup timeout during model download
        print("\n[INFO] OCR Engine will load on first analysis request.")

        _system_ready = True
        print("\n" + "=" * 60)
        print("  ✅ All systems ready!")
        print("  📡 API available at http://localhost:8000")
        print("  📄 Docs at http://localhost:8000/docs")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Startup error: {e}")
        traceback.print_exc()
        _system_ready = True  # Allow partial operation


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "ready": _system_ready}


@app.get("/api/status")
async def status():
    """System status — which models are loaded."""
    yolo_model_path = os.path.join(os.path.dirname(__file__), "backend", "models", "yolo_fssai.pt")
    yolo_available = os.path.exists(yolo_model_path)
    return {
        "ready": _system_ready,
        "ocr_engine": "EasyOCR" if _ocr_reader else "Ready (Lazy loaded)",
        "rag_pipeline": "ChromaDB + all-MiniLM-L6-v2",
        "symbol_detection": "YOLO11n (yolo_fssai.pt)" if yolo_available else "OpenCV Color Fallback",
        "yolo_model_loaded": yolo_available,
    }


@app.get("/api/audit-logs")
async def get_audit_logs(limit: int = 50):
    """Retrieve historical audit logs (newest first) per FR8."""
    from backend.compliance_engine import AUDIT_LOG_DIR
    import json
    log_file = os.path.join(AUDIT_LOG_DIR, "audit_log.jsonl")
    if not os.path.exists(log_file):
        return {"logs": [], "total": 0}

    logs = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))
        logs.reverse()
        return {"logs": logs[:limit], "total": len(logs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read audit logs: {str(e)}")


@app.get("/api/audit-logs/export")
async def export_audit_logs():
    """Export all audit log entries as JSON download per FR8."""
    from backend.compliance_engine import AUDIT_LOG_DIR
    import json
    log_file = os.path.join(AUDIT_LOG_DIR, "audit_log.jsonl")
    if not os.path.exists(log_file):
        return JSONResponse(content=[])
    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))
    return JSONResponse(content=logs)


@app.get("/api/ablation-study")
async def ablation_study():
    """Return benchmark ablation study comparing Base Paper (SVM, DT) vs. Our System."""
    data = get_ablation_benchmark()
    return JSONResponse(content=jsonable_encoder(data))


@app.post("/api/analyze")
async def analyze_label(file: UploadFile = File(...)):
    """
    Full FSSAI compliance audit pipeline.

    Accepts an image file and returns a comprehensive audit report with:
    - Per-element compliance verdicts
    - RAG-retrieved regulatory citations
    - Symbol detection results
    - Extracted structured fields
    - Actionable fix recommendations
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPG, PNG, WebP)")

    start_time = time.time()
    pipeline_steps = []

    try:
        # ── Step 1: Read image ──
        step_start = time.time()
        contents = await file.read()
        img_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image file")

        pipeline_steps.append({
            "step": "Image Loading",
            "duration_ms": round((time.time() - step_start) * 1000),
            "detail": f"Image loaded: {image.shape[1]}x{image.shape[0]} px",
        })

        # ── Step 2: Preprocess for OCR ──
        step_start = time.time()
        preprocessed_ocr = preprocess_for_ocr(image)
        pipeline_steps.append({
            "step": "Image Preprocessing (OCR)",
            "duration_ms": round((time.time() - step_start) * 1000),
            "detail": "CLAHE, deskew, denoise applied",
        })

        # ── Step 3: Preprocess for detection ──
        step_start = time.time()
        preprocessed_det = preprocess_for_detection(image)
        pipeline_steps.append({
            "step": "Image Preprocessing (Detection)",
            "duration_ms": round((time.time() - step_start) * 1000),
            "detail": "Color-enhanced for symbol detection",
        })

        # ── Step 4: OCR Text Extraction ──
        step_start = time.time()
        reader = _get_ocr_reader()

        ocr_results = reader.readtext(preprocessed_ocr)

        # Combine all text with confidence
        raw_text_parts = []
        total_confidence = 0
        for (bbox, text, conf) in ocr_results:
            raw_text_parts.append(text)
            total_confidence += conf

        raw_text = " ".join(raw_text_parts)
        avg_confidence = (total_confidence / len(ocr_results) * 100) if ocr_results else 0

        # Also build a structured OCR output
        ocr_output = [
            {"text": text, "confidence": round(conf, 3), "bbox": [list(map(int, p)) for p in bbox]}
            for (bbox, text, conf) in ocr_results
        ]

        pipeline_steps.append({
            "step": "OCR Text Extraction (EasyOCR)",
            "duration_ms": round((time.time() - step_start) * 1000),
            "detail": f"Extracted {len(ocr_results)} text regions, avg confidence: {avg_confidence:.1f}%",
        })

        # ── Step 5: Symbol Detection ──
        step_start = time.time()
        symbol_results = detect_symbols(preprocessed_det, ocr_results=ocr_results)
        pipeline_steps.append({
            "step": f"Symbol Detection ({symbol_results.get('method', 'unknown')})",
            "duration_ms": round((time.time() - step_start) * 1000),
            "detail": _format_symbol_summary(symbol_results),
        })

        # ── Step 6: Structured Field Extraction ──
        step_start = time.time()
        fields = extract_fields(raw_text)
        pipeline_steps.append({
            "step": "Structured Field Extraction (Regex + NER)",
            "duration_ms": round((time.time() - step_start) * 1000),
            "detail": "License, dates, allergens, nutrition, dietary keywords extracted",
        })

        # ── Step 7: Compliance Audit + RAG ──
        step_start = time.time()
        audit_result = run_compliance_audit(
            extracted_fields=fields,
            symbol_results=symbol_results,
            ocr_confidence=avg_confidence,
        )
        pipeline_steps.append({
            "step": "Compliance Engine + RAG Explanation (R01–R10)",
            "duration_ms": round((time.time() - step_start) * 1000),
            "detail": f"{audit_result['total']} rules evaluated, {audit_result['pass_count']} passed, {audit_result['fail_count']} failed, {audit_result['review_count']} review",
        })

        # ── Step 8: Build Structured Record (§5.3) ──
        structured_record = build_structured_record(fields, symbol_results, avg_confidence)

        # ── Step 9: Claims Verification & Greenwashing Engine (Base Paper Extension) ──
        step_start = time.time()
        claims_result = verify_claims(fields, raw_text, symbol_results)
        pipeline_steps.append({
            "step": "FSSAI Claim Verification (SVM / DT / Proposed Ensemble)",
            "duration_ms": round((time.time() - step_start) * 1000),
            "detail": f"{claims_result['claims_evaluated']} claims evaluated (Base paper SVM & Decision Tree reproduced)",
        })

        # ── Step 10: Indian Nutrition Rating (INR) & HFSS Public Health Analysis ──
        step_start = time.time()
        inr_result = compute_inr_rating(fields)
        pipeline_steps.append({
            "step": "Indian Nutrition Rating (INR 0.5–5.0 Stars) & HFSS",
            "duration_ms": round((time.time() - step_start) * 1000),
            "detail": f"INR Rating: {inr_result['star_rating']} Stars | HFSS Risk: {inr_result['is_hfss']}",
        })

        # ── Final response ──
        total_time = round((time.time() - start_time) * 1000)

        return JSONResponse(content=jsonable_encoder({
            "success": True,
            "audit": audit_result,
            "structured_record": structured_record,
            "claims_analysis": claims_result,
            "inr_analysis": inr_result,
            "raw_text": raw_text,
            "ocr_output": ocr_output,
            "symbol_detection": symbol_results,
            "extracted_fields": {
                "fssai_license": fields["fssai_license"],
                "date_marking": fields["date_marking"],
                "allergens": fields["allergens"],
                "nutritional_info": fields["nutritional_info"],
                "dietary_symbol_text": fields["dietary_symbol_text"],
                "ingredients_text": fields.get("ingredients_text", {}),
                "net_quantity": fields.get("net_quantity", {}),
                "batch_info": fields.get("batch_info", {}),
            },
            "pipeline": pipeline_steps,
            "total_time_ms": total_time,
        }))

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


def _format_symbol_summary(results: dict) -> str:
    """Format symbol detection results for logging."""
    parts = []
    for key in ["veg_mark", "non_veg_mark", "fssai_logo"]:
        det = results.get(key, {})
        if det.get("detected"):
            parts.append(f"{key}: ✅ ({det.get('confidence', 0):.1%})")
        else:
            parts.append(f"{key}: ❌")
    return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
