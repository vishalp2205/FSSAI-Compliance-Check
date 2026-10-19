import requests
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IMAGE_PATH = r"C:\Users\Manisha Khurana\Downloads\AIES Mini Project\public\test_label.jpg"
API_URL = "http://localhost:8000/api/analyze"

print("=" * 60)
print("  FSSAI Label Auditor — Full Pipeline Test (R01-R10)")
print("=" * 60)
print()

try:
    with open(IMAGE_PATH, "rb") as f:
        print(f"Uploading: {IMAGE_PATH}")
        print("Running analysis...")
        resp = requests.post(API_URL, files={"file": ("test_label.jpg", f, "image/jpeg")}, timeout=180)

    data = resp.json()

    if not data.get("success"):
        print("ERROR:", data)
        sys.exit(1)

    audit = data["audit"]
    print(f"\n{'=' * 60}")
    print(f"  RESULT: {audit['overall_label']} — {audit['score']}%")
    print(f"  Pass: {audit['pass_count']} | Fail: {audit['fail_count']} | Review: {audit['review_count']}")
    print(f"  Total Rules: {audit['total']} | OCR Confidence: {audit['ocr_confidence']:.1f}%")
    print(f"  Total Time: {data['total_time_ms']}ms")
    print(f"{'=' * 60}")

    print("\n--- Pipeline Timing ---")
    for step in data["pipeline"]:
        detail = step["detail"][:70]
        print(f"  {step['step']:<50} {step['duration_ms']:>6}ms  {detail}")

    print("\n--- Compliance Checks (R01-R10) ---")
    for check in audit["checks"]:
        icon = {"pass": "PASS", "fail": "FAIL", "review": "REVIEW"}.get(check["status"], "???")
        print(f"\n  [{icon}] {check['rule_id']}: {check['title']}")
        print(f"    Verdict: {check['verdict']}")
        print(f"    Extracted: {check['extracted'][:100]}")
        print(f"    RAG Section: {check.get('section', 'N/A')[:80]}")
        if check.get("fix"):
            print(f"    Fix: {check['fix'][:100]}")

    print("\n--- Structured Record (§5.3) ---")
    sr = data.get("structured_record", {})
    print(f"  License: {sr.get('license_no', {})}")
    print(f"  Dietary Mark: {sr.get('dietary_mark', {})}")
    print(f"  FSSAI Logo: {sr.get('fssai_logo', {})}")
    print(f"  Date Marking: {sr.get('date_marking', {})}")
    print(f"  Allergen: {sr.get('allergen_line', {})}")

    print("\n--- Symbol Detection ---")
    sd = data.get("symbol_detection", {})
    for key in ["veg_mark", "non_veg_mark", "fssai_logo"]:
        det = sd.get(key, {})
        status = f"Detected ({det.get('confidence', 0):.0%})" if det.get("detected") else "Not found"
        print(f"  {key}: {status}")

    print(f"\n\nFull pipeline test PASSED!")

except requests.exceptions.ConnectionError:
    print("ERROR: Backend server not running at http://localhost:8000")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
