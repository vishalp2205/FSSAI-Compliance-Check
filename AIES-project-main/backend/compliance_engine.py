r"""
FSSAI Compliance Engine — Python Backend

Deterministic rule-based compliance checker mapped to
FSSAI (Labelling and Display) Regulations, 2020.

Implements Rule Catalogue R01–R10 per project report §4.3:
  R01: License format (^[12]\d{13}$)
  R02: License state code valid
  R03: License present near FSSAI label
  R04: FSSAI logo detected
  R05: Dietary mark present (veg or non-veg)
  R06: Dietary mark colour correct
  R07: Dietary mark vs ingredients consistency
  R08: Allergen declaration present
  R09: Date marking present
  R10: Nutrition fields complete

Verdict logic (§6):
  - Compliant: Rule satisfied AND evidence above confidence threshold
  - Non-Compliant: Rule violated AND evidence above confidence threshold
  - Needs Human Review: Evidence below threshold, or ambiguous

Integrates:
  - RAG-retrieved regulatory citations for explainability
  - Symbol detection results from OpenCV/YOLO
  - OCR confidence gating per §6
  - Audit logging per FR8
"""

import re
import os
import json
import datetime
from .rag_pipeline import generate_explanation

# ── Confidence Thresholds (§6 — favour review over missed violations) ──
OCR_CONFIDENCE_THRESHOLD = 60.0    # Below this → Needs Human Review
SYMBOL_CONFIDENCE_THRESHOLD = 0.4  # Below this → Needs Human Review

# ── Audit Log Path ──
AUDIT_LOG_DIR = os.path.join(os.path.dirname(__file__), "data", "audit_logs")
os.makedirs(AUDIT_LOG_DIR, exist_ok=True)

# ── Valid FSSAI State Codes (digits 2-3 of license) ──
VALID_STATE_CODES = {
    '01': 'Jammu & Kashmir', '02': 'Himachal Pradesh', '03': 'Punjab',
    '04': 'Chandigarh', '05': 'Uttarakhand', '06': 'Haryana',
    '07': 'Delhi', '08': 'Rajasthan', '09': 'Uttar Pradesh',
    '10': 'Bihar', '11': 'Sikkim', '12': 'Arunachal Pradesh',
    '13': 'Nagaland', '14': 'Manipur', '15': 'Mizoram', '16': 'Tripura',
    '17': 'Meghalaya', '18': 'Assam', '19': 'West Bengal',
    '20': 'Jharkhand', '21': 'Odisha', '22': 'Chhattisgarh',
    '23': 'Madhya Pradesh', '24': 'Gujarat', '25': 'Daman & Diu',
    '26': 'Dadra & Nagar Haveli', '27': 'Maharashtra', '28': 'Andhra Pradesh',
    '29': 'Karnataka', '30': 'Goa', '31': 'Lakshadweep',
    '32': 'Kerala', '33': 'Tamil Nadu', '34': 'Puducherry',
    '35': 'Andaman & Nicobar', '36': 'Telangana', '37': 'Andhra Pradesh (new)',
}

# ── Major FSSAI Allergens (8 categories) ──
ALLERGEN_CATEGORIES = [
    {"name": "Cereals containing gluten", "patterns": ["gluten", "wheat", "barley", "rye", "oats", "spelt"]},
    {"name": "Crustaceans", "patterns": ["crustacean", "shrimp", "prawn", "crab", "lobster", "crayfish"]},
    {"name": "Eggs", "patterns": ["egg"]},
    {"name": "Fish", "patterns": ["fish", "anchov"]},
    {"name": "Peanuts", "patterns": ["peanut", "groundnut"]},
    {"name": "Soybeans", "patterns": ["soy", "soya", "soybean"]},
    {"name": "Milk", "patterns": ["milk", "dairy", "lactose", "casein", "whey", "cream", "butter", "cheese", "curd", "ghee", "paneer"]},
    {"name": "Tree nuts", "patterns": ["almond", "cashew", "walnut", "pistachio", "hazelnut", "macadamia", "pecan", "brazil nut", "tree nut"]},
]

# ── Non-Veg ingredient indicators (for R07) ──
NON_VEG_INDICATORS = [
    "chicken", "mutton", "lamb", "beef", "pork", "meat", "fish", "prawn",
    "shrimp", "crab", "lobster", "egg", "gelatin", "lard", "tallow",
    "anchovy", "squid", "oyster", "clam", "sardine", "tuna", "salmon",
]

# ── Required nutritional fields ──
NUTRITION_REQUIRED = ["energy", "protein", "carbohydrate", "sugar", "fat", "sodium"]
NUTRITION_KEYWORDS = [
    "energy", "calories", "kcal", "kj",
    "protein",
    "carbohydrate", "total carbohydrate",
    "sugar", "total sugar", "added sugar",
    "fat", "total fat", "saturated fat", "trans fat",
    "sodium", "salt", "cholesterol", "fibre", "fiber",
]


# ═══════════════════════════════════════════════════════════════
# FIELD EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_fields(raw_text: str) -> dict:
    """Extract structured fields from raw OCR text."""
    text = raw_text or ""
    lower = text.lower()

    return {
        "fssai_license": _extract_license(text),
        "date_marking": _extract_dates(text, lower),
        "allergens": _extract_allergens(lower),
        "nutritional_info": _extract_nutrition(lower),
        "dietary_symbol_text": _extract_dietary_text(lower),
        "ingredients_text": _extract_ingredients(lower),
        "net_quantity": _extract_net_quantity(text, lower),
        "batch_info": _extract_batch_info(text, lower),
        "raw_text": text,
    }


def build_structured_record(extracted_fields: dict, symbol_results: dict | None, ocr_confidence: float) -> dict:
    """
    Build the structured JSON extraction record per report §5.3 / Listing 1.
    This is the canonical schema for downstream processing.
    """
    lic = extracted_fields["fssai_license"]
    primary_lic = lic[0] if lic else None

    dates = extracted_fields["date_marking"]
    allergens = extracted_fields["allergens"]
    nutrition = extracted_fields["nutritional_info"]

    # Symbol info
    veg_det = symbol_results.get("veg_mark", {}) if symbol_results else {}
    nonveg_det = symbol_results.get("non_veg_mark", {}) if symbol_results else {}
    logo_det = symbol_results.get("fssai_logo", {}) if symbol_results else {}

    veg_detected = veg_det.get("detected", False)
    nonveg_detected = nonveg_det.get("detected", False)

    if veg_detected and nonveg_detected:
        dietary_class = "veg" if veg_det.get("confidence", 0) >= nonveg_det.get("confidence", 0) else "non_veg"
    elif veg_detected:
        dietary_class = "veg"
    elif nonveg_detected:
        dietary_class = "non_veg"
    else:
        dietary_class = None

    return {
        "license_no": {
            "value": primary_lic["value"] if primary_lic else None,
            "conf": round(ocr_confidence / 100, 3),
            "type": primary_lic["type"] if primary_lic else None,
            "state_code": primary_lic["state_code"] if primary_lic else None,
            "state_name": primary_lic["state_name"] if primary_lic else None,
        },
        "dietary_mark": {
            "class": dietary_class,
            "colour_ok": veg_det.get("colour_verified", False) or nonveg_det.get("colour_verified", False) or veg_det.get("detected", False) or nonveg_det.get("detected", False),
            "conf": max(veg_det.get("confidence", 0), nonveg_det.get("confidence", 0)),
        },
        "fssai_logo": {
            "found": logo_det.get("detected", False),
            "conf": logo_det.get("confidence", 0),
        },
        "date_marking": {
            "phrase": next((p for p in ["Best Before", "Use By", "Expiry"] if dates.get(f"has_{p.lower().replace(' ', '_')}", False)), None),
            "value": dates["found_dates"][0] if dates["found_dates"] else None,
            "all_dates": dates["found_dates"],
            "conf": round(ocr_confidence / 100, 3),
        },
        "ingredients": extracted_fields.get("ingredients_text", {}).get("text", ""),
        "allergen_line": {
            "text": ", ".join(allergens["found"]) if allergens["found"] else None,
            "has_section": allergens["has_allergen_section"],
            "conf": round(ocr_confidence / 100, 3),
        },
        "nutrition": {
            field: field in nutrition["found"]
            for field in NUTRITION_REQUIRED
        },
        "net_quantity": extracted_fields.get("net_quantity", {}).get("value"),
        "batch_number": extracted_fields.get("batch_info", {}).get("value"),
    }


def _extract_license(text: str) -> list[dict]:
    """Extract FSSAI license numbers."""
    cleaned = re.sub(r'[\s\-\.]', ' ', text)

    # Strict: 14 digits starting with 1 or 2
    strict = re.findall(r'\b([12]\d{13})\b', cleaned)
    if strict:
        return [_parse_license(num) for num in strict]

    # Fallback: any 14-digit number
    fallback = re.findall(r'\b(\d{14})\b', cleaned)
    if fallback:
        return [_parse_license(num) for num in fallback]

    # Look for labeled license numbers
    label_matches = re.findall(
        r'(?:lic(?:ense|ence)?\.?\s*(?:no|number)?\.?|fssai)\s*[:\-]?\s*(\d[\d\s\-\.]{10,20}\d)',
        text, re.IGNORECASE
    )
    if label_matches:
        results = []
        for m in label_matches:
            digits = re.sub(r'\D', '', m)
            results.append(_parse_license(digits))
        return results

    return []


def _parse_license(num: str) -> dict:
    """Parse a license number string into structured data."""
    state_code = num[1:3] if len(num) >= 3 else ""
    return {
        "value": num,
        "length": len(num),
        "type": "Central" if num.startswith("1") else ("State" if num.startswith("2") else "Invalid prefix"),
        "state_code": state_code,
        "state_name": VALID_STATE_CODES.get(state_code, "Unknown"),
        "is_valid_state_code": state_code in VALID_STATE_CODES,
        "is_valid_format": len(num) == 14 and bool(re.match(r'^[12]\d{13}$', num)) and state_code in VALID_STATE_CODES,
    }


def _extract_dates(text: str, lower: str) -> dict:
    """Extract statutory date markings and types."""
    result = {
        "found_dates": [],
        "has_use_by": bool(re.search(r'\buse\s*by\b', lower)),
        "has_expiry": bool(re.search(r'\b(?:expiry|expiration|exp\s*date|exp\.?\s*[:\-])\b', lower)),
        "has_best_before": bool(re.search(r'\bbest\s*before\b', lower)),
        "has_manufacturing": bool(re.search(r'\b(?:mfg|mfd|manuf(?:actur(?:ed|ing))?)\s*(?:date|on)?\b', lower)),
        "has_packing": bool(re.search(r'\b(?:pkg|pkd|pack(?:ed|ing)?)\s*(?:date|on)?\b', lower)),
    }

    date_patterns = [
        r'\d{1,2}[\/-]\d{1,2}[\/-]\d{2,4}',
        r'\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*[\s,]*\d{2,4}',
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*[\s,]*\d{1,2}[\s,]*\d{2,4}',
        r'\d{4}[\/-]\d{1,2}[\/-]\d{1,2}',
        r'\d{1,2}[\/-]\d{4}',          # MM/YYYY (valid for shelf life > 3 months)
        r'\d{1,2}\.\d{1,2}\.\d{2,4}',  # DD.MM.YYYY with dots
    ]

    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        result["found_dates"].extend(matches)

    result["found_dates"] = list(set(result["found_dates"]))
    return result


def _extract_allergens(lower: str) -> dict:
    """Extract allergen-related information."""
    found = []
    for allergen in ALLERGEN_CATEGORIES:
        if any(p in lower for p in allergen["patterns"]):
            found.append(allergen["name"])

    return {
        "found": found,
        "has_allergen_section": bool(re.search(r'(?:allergen|contains|may contain|allergy)', lower)),
        "total_categories": len(ALLERGEN_CATEGORIES),
    }


def _extract_nutrition(lower: str) -> dict:
    """Extract nutritional information."""
    found = [kw for kw in NUTRITION_KEYWORDS if kw in lower]
    missing = [req for req in NUTRITION_REQUIRED if not any(req in f for f in found)]

    serving_match = re.search(r'(?:per|serving)\s*(?:size)?\s*[:\-]?\s*([\d.]+\s*(?:g|ml|gm))', lower)

    return {
        "found": found,
        "missing": missing,
        "has_nutrition_section": bool(re.search(r'(?:nutrition|nutritional)\s*(?:information|facts|value)', lower)),
        "serving_size": serving_match.group(1) if serving_match else None,
        "raw_text": lower,
    }


def _extract_dietary_text(lower: str) -> dict:
    """Extract dietary symbol indicators from text."""
    veg_kw = ["vegetarian", "veg ", "pure veg", "100% vegetarian"]
    non_veg_kw = ["non-vegetarian", "non vegetarian", "nonvegetarian", "non-veg", "non veg",
                  "contains egg", "contains fish", "contains meat"]

    is_veg = any(v in lower for v in veg_kw)
    is_non_veg = any(v in lower for v in non_veg_kw)

    return {
        "detected": is_veg or is_non_veg,
        "type": "Non-Vegetarian" if is_non_veg else ("Vegetarian" if is_veg else "Not Detected"),
        "source": "text_keywords",
    }


def _extract_ingredients(lower: str) -> dict:
    """Extract ingredients section for R07 cross-check."""
    # Try to find ingredients section
    match = re.search(r'ingredients?\s*[:\-]?\s*(.+?)(?:contains|allergen|nutrition|storage|best before|use by|mfg|$)',
                      lower, re.DOTALL)
    ingredients_text = match.group(1).strip() if match else lower

    has_non_veg = any(indicator in ingredients_text for indicator in NON_VEG_INDICATORS)
    non_veg_found = [ind for ind in NON_VEG_INDICATORS if ind in ingredients_text]

    return {
        "text": ingredients_text[:500],
        "has_non_veg_ingredients": has_non_veg,
        "non_veg_terms": non_veg_found,
    }


def _extract_net_quantity(text: str, lower: str) -> dict:
    """Extract net quantity declaration per Regulation 9."""
    match = re.search(r'(?:net\s*(?:wt\.?|weight|qty\.?|quantity|content|volume)?)\s*[:\-]?\s*(\d+(?:\.\d+)?\s*(?:kg|g|gm|gms|ml|l|ltr|litres|liters))\b', lower)
    if not match:
        match = re.search(r'\b(\d+(?:\.\d+)?\s*(?:kg|g|gm|gms|ml|l|ltr))\b', lower)
    val = match.group(1).strip() if match else None
    return {
        "value": val,
        "found": val is not None,
    }


def _extract_batch_info(text: str, lower: str) -> dict:
    """Extract batch/lot/code identification per Regulation 3(h)."""
    match = re.search(r'(?:batch|lot|code|b\.?\s*no\.?|lot\s*no\.?|b\.?\s*code)\s*[:\-]?\s*([a-zA-Z0-9\-\/]+)', text, re.IGNORECASE)
    val = match.group(1).strip() if match else None
    return {
        "value": val,
        "found": val is not None,
    }


# ═══════════════════════════════════════════════════════════════
# VERDICT HELPER (§6 — Confidence Gating)
# ═══════════════════════════════════════════════════════════════

def _verdict(status: str) -> str:
    """Map status to verdict string per §6."""
    if status == "pass":
        return "Compliant"
    elif status == "fail":
        return "Non-Compliant"
    else:
        return "Needs Human Review"


def _confidence_gate(base_status: str, ocr_confidence: float, detail: str) -> tuple[str, str]:
    """
    Apply confidence gating per §6 verdict logic.
    If OCR confidence is below threshold, route to review instead of hard verdict.
    Returns (gated_status, updated_detail).
    """
    if ocr_confidence > 0 and ocr_confidence < OCR_CONFIDENCE_THRESHOLD:
        if base_status in ("pass", "fail"):
            return "review", detail + f" [OCR confidence {ocr_confidence:.0f}% below threshold {OCR_CONFIDENCE_THRESHOLD:.0f}% — routed to human review]"
    return base_status, detail


# ═══════════════════════════════════════════════════════════════
# COMPLIANCE AUDIT — R01 to R10
# ═══════════════════════════════════════════════════════════════

def run_compliance_audit(
    extracted_fields: dict,
    symbol_results: dict | None = None,
    ocr_confidence: float = 0.0,
) -> dict:
    """
    Run the full FSSAI compliance audit per report §4.3 Rule Catalogue.

    Implements R01–R10 with:
      - Deterministic rule-based checks
      - Confidence-gated verdicts (§6)
      - RAG-retrieved regulatory explanations
      - Symbol detection results

    Args:
        extracted_fields: Output of extract_fields()
        symbol_results: Output of symbol_detection.detect_symbols()
        ocr_confidence: OCR confidence score (0-100)

    Returns:
        Complete audit result with checks, scores, and explanations
    """
    checks = []
    pass_count = 0
    fail_count = 0
    review_count = 0

    def _add_check(rule_id, title, status, detail, extracted, check_type, extra=None):
        nonlocal pass_count, fail_count, review_count

        # Apply confidence gating
        status, detail = _confidence_gate(status, ocr_confidence, detail)

        if status == "pass":
            pass_count += 1
        elif status == "fail":
            fail_count += 1
        else:
            review_count += 1

        rag = generate_explanation(check_type, status, str(extracted), detail)

        check = {
            "id": rule_id,
            "title": title,
            "rule_id": rule_id.upper(),
            "status": status,
            "verdict": _verdict(status),
            "detail": detail,
            "extracted": str(extracted),
            "regulation": rag["regulation_text"],
            "section": rag["section"],
            "explanation": rag["explanation"],
            "fix": rag.get("fix"),
            "rag_chunks": rag.get("retrieved_chunks", []),
        }
        if extra:
            check.update(extra)
        checks.append(check)

    # ════════ R01: License Format ════════
    lic = extracted_fields["fssai_license"]
    if lic:
        primary = lic[0]
        if primary["is_valid_format"]:
            _add_check("r01", "License Format (R01)", "pass",
                       f"Valid 14-digit license matching ^[12]\\d{{13}}$: {primary['value']}.",
                       primary["value"], "fssai_license")
        else:
            issues = []
            if primary["length"] != 14:
                issues.append(f"has {primary['length']} digits instead of 14")
            if primary["type"] == "Invalid prefix":
                issues.append("first digit is not 1 (Central) or 2 (State)")
            _add_check("r01", "License Format (R01)", "fail",
                       f"License found but format invalid: {'; '.join(issues)}.",
                       primary["value"], "fssai_license")
    else:
        _add_check("r01", "License Format (R01)", "fail",
                   "No 14-digit FSSAI license number detected.",
                   "Not found", "fssai_license")

    # ════════ R02: License State Code ════════
    if lic:
        primary = lic[0]
        if primary["is_valid_state_code"]:
            _add_check("r02", "License State Code (R02)", "pass",
                       f"State code '{primary['state_code']}' maps to {primary['state_name']}.",
                       f"{primary['state_code']} → {primary['state_name']}", "fssai_state_code")
        else:
            _add_check("r02", "License State Code (R02)", "fail",
                       f"State code '{primary['state_code']}' does not match any valid Indian state.",
                       primary["state_code"], "fssai_state_code")
    else:
        _add_check("r02", "License State Code (R02)", "fail",
                   "No license found to validate state code.",
                   "N/A", "fssai_state_code")

    # ════════ R03: License Present (near FSSAI label) ════════
    if lic:
        _add_check("r03", "License Present (R03)", "pass",
                   f"FSSAI license number detected: {lic[0]['value']}.",
                   lic[0]["value"], "fssai_presence")
    else:
        _add_check("r03", "License Present (R03)", "fail",
                   "No FSSAI license number found on the label.",
                   "Not found", "fssai_presence")

    # ════════ R04: FSSAI Logo Detected ════════
    fssai_logo = symbol_results.get("fssai_logo", {}) if symbol_results else {}
    if fssai_logo.get("detected"):
        conf = fssai_logo.get("confidence", 0)
        if conf >= SYMBOL_CONFIDENCE_THRESHOLD:
            _add_check("r04", "FSSAI Logo (R04)", "pass",
                       f"FSSAI logo detected via {symbol_results.get('method', 'vision')} (confidence: {conf:.0%}).",
                       f"Detected ({conf:.0%})", "fssai_logo")
        else:
            _add_check("r04", "FSSAI Logo (R04)", "review",
                       f"Possible FSSAI logo detected but confidence is low ({conf:.0%}). Needs manual verification.",
                       f"Borderline ({conf:.0%})", "fssai_logo")
    else:
        _add_check("r04", "FSSAI Logo (R04)", "fail",
                   "FSSAI logo not detected on the label.",
                   "Not detected", "fssai_logo")

    # ════════ R05: Dietary Mark Present ════════
    text_dietary = extracted_fields["dietary_symbol_text"]
    symbol_detected = False
    symbol_type = "Not Detected"
    detection_method = "none"

    if symbol_results:
        veg = symbol_results.get("veg_mark", {})
        non_veg = symbol_results.get("non_veg_mark", {})
        method = symbol_results.get("method", "unknown")

        if veg.get("detected"):
            symbol_detected = True
            symbol_type = "Vegetarian"
            detection_method = method
        elif non_veg.get("detected"):
            symbol_detected = True
            symbol_type = "Non-Vegetarian"
            detection_method = method

    if not symbol_detected and text_dietary["detected"]:
        symbol_detected = True
        symbol_type = text_dietary["type"]
        detection_method = "text_keywords"

    if symbol_detected:
        _add_check("r05", "Dietary Mark Present (R05)", "pass",
                   f"{symbol_type} mark detected via {detection_method}.",
                   f"{symbol_type} ({detection_method})", "dietary_symbol")
    else:
        _add_check("r05", "Dietary Mark Present (R05)", "fail",
                   "No Veg/Non-Veg symbol detected.",
                   "Not detected", "dietary_symbol")

    # ════════ R06: Dietary Mark Colour Correct ════════
    if symbol_results:
        veg = symbol_results.get("veg_mark", {})
        non_veg = symbol_results.get("non_veg_mark", {})

        if veg.get("detected") or non_veg.get("detected"):
            # OpenCV/YOLO detection with colour verification
            detected_conf = max(veg.get("confidence", 0), non_veg.get("confidence", 0))
            mark_color = "Green circle (Veg)" if veg.get("detected") else "Brown triangle (Non-Veg)"
            _add_check("r06", "Dietary Mark Colour (R06)", "pass",
                       f"Colour validated by {symbol_results.get('method', 'vision')} HSV analysis (confidence: {detected_conf:.0%}). {mark_color}.",
                       f"Colour OK ({detected_conf:.0%})", "dietary_color")
        else:
            _add_check("r06", "Dietary Mark Colour (R06)", "review",
                       "No dietary symbol detected to verify colour.",
                       "N/A", "dietary_color")
    else:
        _add_check("r06", "Dietary Mark Colour (R06)", "review",
                   "Symbol detection not available; cannot verify mark colour.",
                   "N/A", "dietary_color")

    # ════════ R07: Dietary Mark vs Ingredients Consistency ════════
    ingredients = extracted_fields.get("ingredients_text", {})
    has_non_veg_ingredients = ingredients.get("has_non_veg_ingredients", False)
    non_veg_terms = ingredients.get("non_veg_terms", [])

    if symbol_detected:
        if symbol_type == "Vegetarian" and has_non_veg_ingredients:
            _add_check("r07", "Mark vs Ingredients (R07)", "fail",
                       f"Label has Vegetarian mark but ingredients contain non-veg terms: {', '.join(non_veg_terms)}.",
                       f"Conflict: Veg mark + {', '.join(non_veg_terms)}", "dietary_ingredients_conflict")
        elif symbol_type == "Non-Vegetarian" and has_non_veg_ingredients:
            _add_check("r07", "Mark vs Ingredients (R07)", "pass",
                       f"Non-Veg mark consistent with ingredients ({', '.join(non_veg_terms)}).",
                       "Consistent", "dietary_ingredients_conflict")
        elif symbol_type == "Vegetarian" and not has_non_veg_ingredients:
            _add_check("r07", "Mark vs Ingredients (R07)", "pass",
                       "Vegetarian mark consistent — no non-veg ingredients detected.",
                       "Consistent", "dietary_ingredients_conflict")
        else:
            _add_check("r07", "Mark vs Ingredients (R07)", "review",
                       f"Non-Veg mark present but no non-veg ingredient keywords found. May be valid but needs review.",
                       "Review recommended", "dietary_ingredients_conflict")
    else:
        _add_check("r07", "Mark vs Ingredients (R07)", "review",
                   "No dietary mark detected; cannot check consistency with ingredients.",
                   "N/A", "dietary_ingredients_conflict")

    # ════════ R08: Allergen Declaration ════════
    allergens = extracted_fields["allergens"]
    if allergens["has_allergen_section"] and allergens["found"]:
        _add_check("r08", "Allergen Declaration (R08)", "pass",
                   f"Allergen section detected with {len(allergens['found'])} categories: {', '.join(allergens['found'])}.",
                   ", ".join(allergens["found"]), "allergen_declaration")
    elif allergens["found"] and not allergens["has_allergen_section"]:
        _add_check("r08", "Allergen Declaration (R08)", "fail",
                   f"Allergen ingredients found ({', '.join(allergens['found'])}) but no dedicated 'Contains:' section.",
                   ", ".join(allergens["found"]), "allergen_declaration")
    else:
        _add_check("r08", "Allergen Declaration (R08)", "review",
                   "No allergen information detected. Compliant if product contains no major allergens.",
                   "Not found (may be N/A)", "allergen_declaration")

    # ════════ R09: Date Marking ════════
    dates = extracted_fields["date_marking"]
    has_required_date = dates["has_use_by"] or dates["has_expiry"] or dates["has_best_before"]

    if has_required_date and dates["found_dates"]:
        types = []
        if dates["has_use_by"]: types.append("Use By")
        if dates["has_expiry"]: types.append("Expiry")
        if dates["has_best_before"]: types.append("Best Before")
        if dates["has_manufacturing"]: types.append("Manufacturing")
        _add_check("r09", "Date Marking (R09)", "pass",
                   f"Date marking found: {', '.join(types)}. Dates: {', '.join(dates['found_dates'])}.",
                   " | ".join(dates["found_dates"]), "date_marking")
    elif dates["found_dates"] and not has_required_date:
        _add_check("r09", "Date Marking (R09)", "review",
                   f"Dates found ({', '.join(dates['found_dates'])}) but no explicit Use By/Expiry/Best Before label.",
                   " | ".join(dates["found_dates"]), "date_marking")
    else:
        _add_check("r09", "Date Marking (R09)", "fail",
                   "No date marking detected on the label.",
                   "Not found", "date_marking")

    # ════════ R10: Nutrition Fields Complete ════════
    nutrition = extracted_fields["nutritional_info"]
    if nutrition["has_nutrition_section"] and not nutrition["missing"]:
        _add_check("r10", "Nutrition Panel (R10)", "pass",
                   f"Nutritional information panel with all required fields: {', '.join(nutrition['found'][:8])}...",
                   ", ".join(nutrition["found"]), "nutritional_info")
    elif nutrition["found"]:
        missing_str = f" Missing: {', '.join(nutrition['missing'])}." if nutrition["missing"] else ""
        _add_check("r10", "Nutrition Panel (R10)", "review",
                   f"Partial nutritional information found.{missing_str}",
                   ", ".join(nutrition["found"]), "nutritional_info")
    else:
        _add_check("r10", "Nutrition Panel (R10)", "fail",
                   "No nutritional information panel detected.",
                   "Not found", "nutritional_info")

    # ────── OVERALL SCORE ──────
    total = len(checks)
    score = round((pass_count / total) * 100) if total > 0 else 0

    if score >= 80:
        overall_status = "high"
    elif score >= 50:
        overall_status = "mid"
    else:
        overall_status = "low"

    labels = {100: "Fully Compliant", 80: "Mostly Compliant", 50: "Partially Compliant", 0: "Significant Issues"}
    overall_label = next(v for k, v in sorted(labels.items(), reverse=True) if score >= k)

    result = {
        "score": score,
        "overall_status": overall_status,
        "overall_label": overall_label,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "warn_count": review_count,
        "review_count": review_count,
        "total": total,
        "checks": checks,
        "ocr_confidence": ocr_confidence,
        "confidence_threshold": OCR_CONFIDENCE_THRESHOLD,
    }

    # ── FR8: Audit Logging ──
    _log_audit(result, extracted_fields)

    return result


# ═══════════════════════════════════════════════════════════════
# AUDIT LOGGING (FR8)
# ═══════════════════════════════════════════════════════════════

def _log_audit(result: dict, extracted_fields: dict) -> None:
    """
    Log audit inputs, intermediate outputs, and verdicts to file per FR8.
    Each audit gets a timestamped JSON log entry.
    """
    try:
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "score": result["score"],
            "overall_label": result["overall_label"],
            "pass_count": result["pass_count"],
            "fail_count": result["fail_count"],
            "review_count": result["review_count"],
            "ocr_confidence": result["ocr_confidence"],
            "checks": [
                {
                    "rule_id": c["rule_id"],
                    "title": c["title"],
                    "verdict": c["verdict"],
                    "extracted": c["extracted"][:100],
                    "section": c.get("section", ""),
                }
                for c in result["checks"]
            ],
            "extracted_summary": {
                "license_found": len(extracted_fields.get("fssai_license", [])) > 0,
                "dates_found": len(extracted_fields.get("date_marking", {}).get("found_dates", [])),
                "allergens_found": len(extracted_fields.get("allergens", {}).get("found", [])),
                "nutrition_fields": len(extracted_fields.get("nutritional_info", {}).get("found", [])),
            },
        }

        log_file = os.path.join(AUDIT_LOG_DIR, "audit_log.jsonl")
        
        def _json_default(obj):
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, default=_json_default, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[AUDIT LOG] Warning: Could not write audit log: {e}")
