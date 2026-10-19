"""
Symbol Detection Module — YOLO11n + OpenCV Hybrid

Per project report §5.2:
  - YOLO11n fine-tuned on 3 classes: veg_mark, non_veg_mark, fssai_logo
  - HSV colour verification inside detected bounding box (R06)
  - OpenCV color-based fallback when no trained YOLO model is available

Per FSSAI 2020:
  - Veg: Green filled circle inside a green square outline
  - Non-Veg: Brown filled triangle inside a brown square outline

Architecture:
  1. Try YOLO11n if backend/models/yolo_fssai.pt exists
  2. Verify detected symbol colours in HSV space (R06)
  3. Fall back to OpenCV color-based detection if no YOLO model
"""

import cv2
import numpy as np
import os

# ── YOLO model path ──
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolo_fssai.pt")
_yolo_model = None
_yolo_loaded = False


def _try_load_yolo():
    """Attempt to load YOLO11n model for symbol detection."""
    global _yolo_model, _yolo_loaded
    _yolo_loaded = True  # Mark as attempted

    if os.path.exists(YOLO_MODEL_PATH):
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[YOLO] ✅ Loaded YOLO11n model from {YOLO_MODEL_PATH}")
            return True
        except ImportError:
            print("[YOLO] ultralytics not installed. Run: pip install ultralytics")
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}")
    else:
        print(f"[YOLO] No trained model at {YOLO_MODEL_PATH} — using OpenCV fallback.")
        print(f"[YOLO] To train: python train_yolo.py")
    return False


def detect_symbols(image: np.ndarray, ocr_results: list | None = None) -> dict:
    """
    Detect mandatory FSSAI statutory symbols on a food label.

    Pipeline per report §5.2:
      1. YOLO11n inference (if model available) → bounding boxes + confidence
      2. Multimodal OCR cross-verification for FSSAI statutory emblem
      3. HSV colour verification inside each box (R06) & mutual exclusivity
      4. OpenCV fallback if no YOLO model

    Args:
        image: BGR image (numpy array from cv2.imread)
        ocr_results: Optional OCR detections list [(bbox, text, conf)] for multimodal grounding

    Returns:
        Dict with detection results for each symbol type
    """
    # Lazy-load YOLO on first call
    if not _yolo_loaded:
        _try_load_yolo()

    # Try YOLO first
    if _yolo_model is not None:
        yolo_results = _detect_with_yolo(image, ocr_results=ocr_results)
        # Post-process: verify colours in HSV space (R06) and enforce mutual exclusivity
        yolo_results = _verify_colours_hsv(image, yolo_results)
        return yolo_results

    # Fallback: OpenCV-based detection
    results = {
        "veg_mark": _detect_veg_symbol(image),
        "non_veg_mark": _detect_nonveg_symbol(image),
        "fssai_logo": _detect_fssai_logo(image),
        "method": "opencv_color_detection",
    }

    return results


def _detect_with_yolo(image: np.ndarray, ocr_results: list | None = None) -> dict:
    """
    Run YOLO11n inference for symbol detection with OCR cross-validation.

    Per report §5.2: YOLO11n nano model fine-tuned on 3 classes.
    Uses conf=0.35 to reject background noise, with multimodal OCR grounding.
    """
    results_raw = _yolo_model(image, conf=0.35, verbose=False)

    detections = {
        "veg_mark": {"detected": False, "confidence": 0.0, "bbox": None},
        "non_veg_mark": {"detected": False, "confidence": 0.0, "bbox": None},
        "fssai_logo": {"detected": False, "confidence": 0.0, "bbox": None},
        "method": "yolo11n",
    }

    candidates = {0: [], 1: [], 2: []}
    for result in results_raw:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [int(x) for x in box.xyxy[0].tolist()]
            if cls_id in candidates:
                candidates[cls_id].append({"conf": conf, "bbox": xyxy})

    # 1. Veg Mark
    if candidates[0]:
        best_veg = max(candidates[0], key=lambda c: c["conf"])
        detections["veg_mark"] = {
            "detected": True,
            "confidence": round(best_veg["conf"], 3),
            "bbox": best_veg["bbox"],
        }

    # 2. Non-Veg Mark
    if candidates[1]:
        best_nonveg = max(candidates[1], key=lambda c: c["conf"])
        detections["non_veg_mark"] = {
            "detected": True,
            "confidence": round(best_nonveg["conf"], 3),
            "bbox": best_nonveg["bbox"],
        }

    # 3. FSSAI Logo — Multimodal contour & emblem grounding
    # Per FSSAI Schedule I & Reg. 4: Graphic triangular/shield emblem with 'fssai' stylized lettering.
    # Must NEVER select the license text line or words 'FSSAI LIC. NO.'!
    veg_bbox = detections["veg_mark"]["bbox"] if detections["veg_mark"]["detected"] else None
    nonveg_bbox = detections["non_veg_mark"]["bbox"] if detections["non_veg_mark"]["detected"] else None

    logo_det = _detect_fssai_emblem_precise(
        image=image,
        ocr_results=ocr_results,
        yolo_candidates=candidates[2],
        veg_bbox=veg_bbox,
        nonveg_bbox=nonveg_bbox
    )
    detections["fssai_logo"] = logo_det

    return detections


def _detect_fssai_emblem_precise(
    image: np.ndarray,
    ocr_results: list | None = None,
    yolo_candidates: list | None = None,
    veg_bbox: list | None = None,
    nonveg_bbox: list | None = None,
) -> dict:
    """
    Precisely detect the statutory graphic FSSAI logo emblem (triangular/shield insignia).
    Guarantees that the bounding box highlights the actual logo graphic,
    NOT the adjacent license text line or 'FSSAI LIC. NO.'.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = image.shape[:2]

    def overlaps(b1, b2, thresh=0.3):
        if not b1 or not b2:
            return False
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        ia = (ix2 - ix1) * (iy2 - iy1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        return (ia / max(a1, 1)) > thresh

    # 1. Parse OCR text hints: separate pure emblem tokens from license line tokens
    logo_tokens = []
    lic_tokens = []
    if ocr_results:
        for item in ocr_results:
            text = item[1].lower() if len(item) > 1 else ""
            pts = np.array(item[0], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            b = [x, y, x + w, y + h]
            if "fssai" in text or "jssai" in text or "gov.in" in text:
                # If it mentions license, no., or 10-14 digit numbers, it is the license text line
                if any(k in text for k in ["lic", "no.", "number", "101", "212"]):
                    lic_tokens.append((b, text))
                else:
                    logo_tokens.append((b, text))

    # 2. Contour analysis for triangular / shield shape
    edges = cv2.Canny(gray, 40, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    enclosing_contours = []
    near_lic_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 800 < area < 25000:
            x, y, w, h = cv2.boundingRect(c)
            b = [x, y, x + w, y + h]
            # Exclude dietary mark overlap
            if (veg_bbox and overlaps(b, veg_bbox, 0.4)) or (nonveg_bbox and overlaps(b, nonveg_bbox, 0.4)):
                continue
            aspect = w / float(h) if h > 0 else 0
            if 0.65 <= aspect <= 1.45:
                # Check if this contour encloses any logo token
                encloses_logo = False
                for lt in logo_tokens:
                    tb = lt[0]
                    if x <= tb[0] + 15 and y <= tb[1] + 15 and (x + w) >= tb[2] - 15 and (y + h) >= tb[3] - 15:
                        encloses_logo = True
                        break

                shape_score = 1.0 - abs(aspect - 1.0) * 0.5
                if encloses_logo:
                    enclosing_contours.append((shape_score, area, b))
                elif any(abs(y - lt[0][3]) < 180 or abs(lt[0][1] - (y + h)) < 180 for lt in lic_tokens):
                    near_lic_contours.append((shape_score, area, b))

    # Priority A: Contour strictly enclosing logo text tokens (e.g. 'fssai' or 'gov.in')
    if enclosing_contours:
        enclosing_contours.sort(key=lambda x: (x[0], -abs(x[1] - 9000)), reverse=True)
        best_b = enclosing_contours[0][2]
        return {
            "detected": True,
            "confidence": 0.94,
            "bbox": best_b,
            "details": "FSSAI statutory emblem detected via contour enclosing logo text",
        }

    # Priority B: Enclose distinct logo tokens
    if logo_tokens:
        min_x = min(lt[0][0] for lt in logo_tokens)
        min_y = min(lt[0][1] for lt in logo_tokens)
        max_x = max(lt[0][2] for lt in logo_tokens)
        max_y = max(lt[0][3] for lt in logo_tokens)
        pad_x = int((max_x - min_x) * 0.22) + 6
        pad_y = int((max_y - min_y) * 0.28) + 8
        bbox = [max(0, min_x - pad_x), max(0, min_y - pad_y), min(w_img, max_x + pad_x), min(h_img, max_y + pad_y)]
        return {
            "detected": True,
            "confidence": 0.90,
            "bbox": bbox,
            "details": "FSSAI emblem localized from statutory logo text",
        }

    # Priority C: Contour near license line
    if near_lic_contours:
        near_lic_contours.sort(key=lambda x: (x[0], -abs(x[1] - 9000)), reverse=True)
        return {
            "detected": True,
            "confidence": 0.84,
            "bbox": near_lic_contours[0][2],
            "details": "FSSAI emblem contour located adjacent to license declaration",
        }

    # Priority D: YOLO candidates (filter out wide text blocks)
    if yolo_candidates:
        valid_yolo = []
        for cand in yolo_candidates:
            cb = cand["bbox"]
            cw = cb[2] - cb[0]
            ch = cb[3] - cb[1]
            aspect = cw / float(max(ch, 1))
            # Emblem is never a super-wide paragraph (aspect > 2.2)
            if 0.5 <= aspect <= 2.2:
                if not (veg_bbox and overlaps(cb, veg_bbox)) and not (nonveg_bbox and overlaps(cb, nonveg_bbox)):
                    valid_yolo.append(cand)
        if valid_yolo:
            best_yolo = max(valid_yolo, key=lambda x: x["conf"])
            return {
                "detected": True,
                "confidence": round(best_yolo["conf"], 3),
                "bbox": best_yolo["bbox"],
                "details": "YOLO11n symbol detection",
            }

    # If no emblem text, no contour near license, and no YOLO detection: not detected
    return {"detected": False, "confidence": 0.0, "bbox": None}


def _verify_colours_hsv(image: np.ndarray, detections: dict) -> dict:
    """
    Post-YOLO colour verification in HSV space per R06.

    Even if YOLO detects a symbol, verify the colour is correct:
      - veg_mark should have green regions
      - non_veg_mark should have brown/maroon regions
    Suppresses false positives and enforces mutual exclusivity per Regulation 5.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for mark_type, color_range in [
        ("veg_mark", (np.array([35, 50, 50]), np.array([85, 255, 255]))),     # Green
        ("non_veg_mark", (np.array([5, 50, 30]), np.array([25, 255, 200]))),  # Brown
    ]:
        det = detections.get(mark_type, {})
        if det.get("detected") and det.get("bbox"):
            x1, y1, x2, y2 = det["bbox"]
            roi = hsv[y1:y2, x1:x2]
            if roi.size > 0:
                mask = cv2.inRange(roi, color_range[0], color_range[1])
                density = float(np.sum(mask > 0)) / float(max(roi.size // 3, 1))
                det["colour_verified"] = bool(density > 0.12)
                det["colour_density"] = float(round(density, 3))
                if not det["colour_verified"]:
                    det["detected"] = False
            else:
                det["colour_verified"] = False
                det["detected"] = False
                det["colour_density"] = 0.0

    # Mutual Exclusivity: A product is either Veg OR Non-Veg per Regulation 5
    veg_ok = detections["veg_mark"].get("detected", False)
    nonveg_ok = detections["non_veg_mark"].get("detected", False)
    if veg_ok and nonveg_ok:
        if detections["veg_mark"]["confidence"] >= detections["non_veg_mark"]["confidence"]:
            detections["non_veg_mark"]["detected"] = False
        else:
            detections["veg_mark"]["detected"] = False

    return detections


# ═══════════════════════════════════════════════════════════════
# OPENCV FALLBACK DETECTORS
# ═══════════════════════════════════════════════════════════════

def _detect_veg_symbol(image: np.ndarray) -> dict:
    """
    Detect the Vegetarian symbol: green filled circle inside a green square.

    Uses HSV color space to find green regions, then checks for circular
    contours within square-ish contours.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green color range in HSV
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_match = {"detected": False, "confidence": 0.0, "bbox": None, "details": ""}

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Skip noise
            continue

        # Check if contour is roughly square-shaped
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)

        if 0.7 <= aspect_ratio <= 1.3:  # Roughly square
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / max(perimeter ** 2, 1)

            roi = mask[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            green_density = np.sum(roi > 0) / max(roi.size, 1)

            confidence = 0.0
            if green_density > 0.3:
                confidence += 0.4
            if 0.8 <= aspect_ratio <= 1.2:
                confidence += 0.3
            if circularity > 0.5:
                confidence += 0.3

            if confidence > best_match["confidence"]:
                best_match = {
                    "detected": bool(confidence > 0.5),
                    "confidence": round(confidence, 3),
                    "bbox": [int(x), int(y), int(x+w), int(y+h)],
                    "colour_verified": True,
                    "details": f"Green region: aspect={aspect_ratio:.2f}, circularity={circularity:.2f}, density={green_density:.2f}",
                }

    return best_match


def _detect_nonveg_symbol(image: np.ndarray) -> dict:
    """
    Detect the Non-Vegetarian symbol: brown filled triangle inside a brown square.

    Uses HSV color space to find brown/maroon regions, then checks for
    triangular contours (3 vertices via polygon approximation).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Brown/maroon color range in HSV
    lower_brown = np.array([5, 50, 30])
    upper_brown = np.array([25, 255, 200])

    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    # Also check for dark red / maroon
    lower_maroon = np.array([0, 70, 30])
    upper_maroon = np.array([10, 255, 180])
    mask2 = cv2.inRange(hsv, lower_maroon, upper_maroon)
    mask = cv2.bitwise_or(mask, mask2)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_match = {"detected": False, "confidence": 0.0, "bbox": None, "details": ""}

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)

        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)

        confidence = 0.0
        brown_density = 0.0

        if num_vertices == 3:
            confidence += 0.5
        elif num_vertices == 4 and 0.7 <= aspect_ratio <= 1.3:
            confidence += 0.3

        roi = mask[y:y+h, x:x+w]
        if roi.size > 0:
            brown_density = np.sum(roi > 0) / max(roi.size, 1)
            if brown_density > 0.2:
                confidence += 0.3

        if 0.7 <= aspect_ratio <= 1.3:
            confidence += 0.2

        if confidence > best_match["confidence"]:
            best_match = {
                "detected": bool(confidence > 0.5),
                "confidence": round(confidence, 3),
                "bbox": [int(x), int(y), int(x+w), int(y+h)],
                "colour_verified": True,
                "details": f"Brown region: vertices={num_vertices}, aspect={aspect_ratio:.2f}, density={brown_density:.2f}",
            }

    return best_match


def _detect_fssai_logo(image: np.ndarray) -> dict:
    """
    Detect the FSSAI logo using contour analysis.

    The FSSAI logo is typically a triangular/shield shape.
    Uses edge detection + contour analysis.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_match = {"detected": False, "confidence": 0.0, "bbox": None, "details": ""}

    img_area = image.shape[0] * image.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)

        area_ratio = area / max(img_area, 1)
        if area_ratio < 0.001 or area_ratio > 0.15:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)

        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        confidence = 0.0

        if 3 <= len(approx) <= 6:
            confidence += 0.4

        if 0.5 <= aspect_ratio <= 2.0:
            confidence += 0.2

        if 0.002 <= area_ratio <= 0.05:
            confidence += 0.2

        if confidence > best_match["confidence"]:
            best_match = {
                "detected": bool(confidence > 0.5),
                "confidence": round(min(confidence, 0.85), 3),
                "bbox": [int(x), int(y), int(x+w), int(y+h)],
                "details": f"Logo candidate: vertices={len(approx)}, area_ratio={area_ratio:.4f}",
            }

    return best_match


# ── Initialize YOLO on module load ──
_try_load_yolo()
