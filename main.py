# main.py
import os
import re
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="Aadhar Data Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_MAP = {
    "0": "AADHAR_NUMBER",
    "1": "DATE_OF_BIRTH",
    "2": "GENDER",
    "3": "NAME",
    "4": "NAME",
}

MODEL_PATH = "yolov8_trained_model.pt"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Train first using train.py.")

model = YOLO(MODEL_PATH)
print(f"✅ Model loaded. Raw classes: {model.names}")

CONFIDENCE_THRESHOLD = 0.4

NAME_BLACKLIST_WORDS = [
    "aadhaar", "aadhar", "proof", "identity", "citizenship",
    "date of birth", "verification", "authentication", "scanning",
    "offline", "online", "xml", "qr", "government", "india",
    "भारत", "सरकार", "मेरा", "पहचान", "आधार", "should", "used",
    "birth", "citizen", "नहीं", "प्रमाण", "नागरिकता"
]


def is_valid_name(text: str) -> bool:
    if not text or len(text.strip()) < 2:
        return False
    text_lower = text.lower()
    for word in NAME_BLACKLIST_WORDS:
        if word in text_lower:
            return False
    if len(text.strip()) > 40:
        return False
    if len(text.strip().split()) > 4:
        return False
    special_chars = sum(1 for c in text if not c.isalnum() and c not in " .',-")
    if special_chars > 2:
        return False
    return True


def clean_aadhar_number(text: str) -> str:
    digits = re.sub(r'\D', '', text)
    if len(digits) == 12:
        return f"{digits[:4]} {digits[4:8]} {digits[8:]}"
    return digits if digits else text.strip()


def clean_dob(text: str) -> str:
    match = re.search(r'\b\d{2}[\/\-]\d{2}[\/\-]\d{4}\b', text)
    return match.group(0) if match else text.strip()


def clean_gender(text: str) -> str:
    text_lower = text.lower()
    if "female" in text_lower:
        return "Female"
    elif "male" in text_lower:
        return "Male"
    elif "transgender" in text_lower or "other" in text_lower:
        return "Transgender/Other"
    if "पुरुष" in text or "पुरष" in text:
        return "Male"
    if "महिला" in text or "स्त्री" in text:
        return "Female"
    return ""


def extract_name_from_text(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        if re.match(r'^[A-Za-z][A-Za-z\s\.]{1,38}$', line):
            if is_valid_name(line):
                return line.strip()
    return ""


def run_ocr(cropped_img: np.ndarray, label: str) -> str:
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    # Scale up small crops — use 3x for all for best accuracy
    scale = max(3, int(300 / w)) if w < 300 else 3
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if label == "NAME":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    if label == "AADHAR_NUMBER":
        # Extra sharpening for digit clarity
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if label == "DATE_OF_BIRTH":
        config = "--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789/-"
    elif label == "AADHAR_NUMBER":
        config = "--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789 "
    elif label == "GENDER":
        config = "--psm 6 --oem 1"
    elif label == "NAME":
        config = "--psm 6 --oem 1"
    else:
        config = "--psm 6 --oem 1"

    text = pytesseract.image_to_string(thresh, config=config).strip()
    return text


def ocr_top_half_for_name(image: np.ndarray) -> str:
    """Crop only top 55% of card and right 65% to find name (skip photo area)."""
    h, w = image.shape[:2]
    top_region = image[int(h * 0.15): int(h * 0.55), int(w * 0.35): w]

    gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_text = ""
    for psm in [6, 3, 4]:
        config = f"--psm {psm} --oem 1"
        text = pytesseract.image_to_string(thresh, config=config).strip()
        print(f"🔍 Top-half OCR (psm={psm}) for name:\n", text)
        if len(text) > len(best_text):
            best_text = text

    return best_text


def ocr_gender_region(image: np.ndarray) -> str:
    """Try multiple horizontal strips to find Male/Female."""
    h, w = image.shape[:2]
    combined_text = ""

    regions = [
        image[int(h * 0.30): int(h * 0.50), int(w * 0.30): int(w * 0.80)],
        image[int(h * 0.40): int(h * 0.60), int(w * 0.30): int(w * 0.80)],
        image[int(h * 0.20): int(h * 0.55), int(w * 0.30): int(w * 0.80)],
    ]

    for i, region in enumerate(regions):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config="--psm 6 --oem 1").strip()
        print(f"🔍 Gender region {i+1} OCR:\n", text)
        combined_text += " " + text

        if re.search(r'\b(male|female|transgender)\b', text, re.IGNORECASE):
            break
        if "पुरुष" in text or "पुरष" in text or "महिला" in text:
            break

    return combined_text


def ocr_aadhar_region(image: np.ndarray) -> str:
    """
    Specifically scan the bottom half of the card for the 12-digit Aadhar number.
    The number is usually printed large in the lower-center of the card.
    """
    h, w = image.shape[:2]

    # Bottom 40% of card, center area
    regions = [
        image[int(h * 0.55): int(h * 0.85), int(w * 0.10): int(w * 0.90)],
        image[int(h * 0.60): int(h * 0.90), int(w * 0.05): int(w * 0.95)],
        image[int(h * 0.50): int(h * 0.80), int(w * 0.15): int(w * 0.85)],
    ]

    for i, region in enumerate(regions):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # Use 3x upscale for large digit text
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(
            thresh,
            config="--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789 "
        ).strip()
        print(f"🔍 Aadhar region {i+1} OCR:\n", text)

        # Check if we found 12 digits
        digits = re.sub(r'\D', '', text)
        if len(digits) == 12:
            return f"{digits[:4]} {digits[4:8]} {digits[8:]}"

        # Also try without whitelist to catch spaces between groups
        text2 = pytesseract.image_to_string(
            thresh, config="--psm 6 --oem 1"
        ).strip()
        match = re.search(r'\b(\d{4}\s\d{4}\s\d{4})\b', text2)
        if match:
            return match.group(1)

    return ""


def full_card_ocr_fallback(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    full_text = pytesseract.image_to_string(thresh, config="--psm 6 --oem 1")

    print("🔍 Full card OCR text:\n", full_text)

    result = {"NAME": None, "DATE_OF_BIRTH": None, "GENDER": None, "AADHAR_NUMBER": None}

    # --- Aadhar Number ---
    aadhar_match = re.search(r'\b(\d{4}\s\d{4}\s\d{4})\b', full_text)
    if aadhar_match:
        result["AADHAR_NUMBER"] = aadhar_match.group(1)

    # Also try finding 12 consecutive digits
    if not result["AADHAR_NUMBER"]:
        digits_match = re.search(r'\b(\d{12})\b', re.sub(r'\s', '', full_text))
        if digits_match:
            d = digits_match.group(1)
            result["AADHAR_NUMBER"] = f"{d[:4]} {d[4:8]} {d[8:]}"

    # --- DOB ---
    dob_match = re.search(
        r'DOB\s*[:\-/]?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4})',
        full_text, re.IGNORECASE
    )
    if dob_match:
        result["DATE_OF_BIRTH"] = dob_match.group(1)
    else:
        dob_match2 = re.search(r'\b(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b', full_text)
        if dob_match2:
            result["DATE_OF_BIRTH"] = dob_match2.group(1)

    # --- Gender from full text ---
    if re.search(r'\bfemale\b', full_text, re.IGNORECASE):
        result["GENDER"] = "Female"
    elif re.search(r'\bmale\b', full_text, re.IGNORECASE):
        result["GENDER"] = "Male"
    elif re.search(r'\btransgender\b', full_text, re.IGNORECASE):
        result["GENDER"] = "Transgender/Other"
    elif re.search(r'mal[e3]', full_text, re.IGNORECASE):
        result["GENDER"] = "Male"
    elif re.search(r'femal[e3]', full_text, re.IGNORECASE):
        result["GENDER"] = "Female"

    # --- Hindi gender words ---
    if not result["GENDER"]:
        if "पुरुष" in full_text or "पुरष" in full_text:
            result["GENDER"] = "Male"
        elif "महिला" in full_text or "स्त्री" in full_text:
            result["GENDER"] = "Female"

    # --- Gender fallback: targeted region ---
    if not result["GENDER"]:
        gender_text = ocr_gender_region(image)
        if re.search(r'\bfemale\b', gender_text, re.IGNORECASE):
            result["GENDER"] = "Female"
        elif re.search(r'\bmale\b', gender_text, re.IGNORECASE):
            result["GENDER"] = "Male"
        elif re.search(r'mal[e3]', gender_text, re.IGNORECASE):
            result["GENDER"] = "Male"
        elif re.search(r'femal[e3]', gender_text, re.IGNORECASE):
            result["GENDER"] = "Female"
        elif "पुरुष" in gender_text or "पुरष" in gender_text:
            result["GENDER"] = "Male"
        elif "महिला" in gender_text:
            result["GENDER"] = "Female"

    # --- Name: scan top half of card ---
    top_text = ocr_top_half_for_name(image)
    top_lines = [l.strip() for l in top_text.splitlines() if l.strip()]

    for line in top_lines:
        if (re.match(r'^[A-Z][a-zA-Z\s\.]{2,38}$', line)
                and is_valid_name(line)
                and not re.search(r'\d', line)
                and line.lower() not in ["government of india", "india", "male",
                                         "female", "government", "of india"]):
            result["NAME"] = line.strip()
            break

    # Broader fallback across full text if still no name
    if not result["NAME"]:
        all_lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        found_gov = False
        for line in all_lines:
            if re.search(r'government\s+of\s+india', line, re.IGNORECASE):
                found_gov = True
                continue
            if found_gov:
                if (re.match(r'^[A-Z][a-zA-Z\s\.]{2,38}$', line)
                        and is_valid_name(line)
                        and not re.search(r'\d', line)):
                    result["NAME"] = line.strip()
                    break

    return result


@app.get("/")
def root():
    return {"message": "Aadhar Extractor API is running. POST an image to /extract"}


@app.post("/extract")
async def extract_aadhar_data(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    # ✅ Keep original full-resolution image for high-res OCR crops
    original = image.copy()
    orig_h, orig_w = original.shape[:2]
    print(f"📐 Original image size: {orig_w}×{orig_h}")

    # Resize only for YOLO inference
    resized = cv2.resize(image, (640, 640))
    results = model.predict(source=resized, conf=CONFIDENCE_THRESHOLD, verbose=False)

    extracted = {
        "NAME": None,
        "DATE_OF_BIRTH": None,
        "GENDER": None,
        "AADHAR_NUMBER": None,
    }

    best_confidence = {
        "NAME": 0,
        "DATE_OF_BIRTH": 0,
        "GENDER": 0,
        "AADHAR_NUMBER": 0,
    }

    detections = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            confidence = float(confs[i])

            raw_label = result.names[int(labels[i])]
            label = CLASS_MAP.get(str(raw_label), raw_label)

            # ✅ Skip NAME detections in lower half (disclaimer zone)
            if label == "NAME" and y1 > 320:
                print(f"⚠️ Skipping NAME bbox at y1={y1} — likely disclaimer region")
                detections.append({
                    "label": "NAME_SKIPPED",
                    "text": "Skipped (disclaimer zone)",
                    "raw_text": "",
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                })
                continue

            pad = 5
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(639, x2 + pad)
            y2p = min(639, y2 + pad)

            # ✅ Scale bbox back to original image resolution for high-res OCR
            scale_x = orig_w / 640
            scale_y = orig_h / 640
            ox1 = max(0, int(x1p * scale_x))
            oy1 = max(0, int(y1p * scale_y))
            ox2 = min(orig_w, int(x2p * scale_x))
            oy2 = min(orig_h, int(y2p * scale_y))

            crop = original[oy1:oy2, ox1:ox2]
            if crop.size == 0:
                continue

            print(f"   Crop size for {label}: {crop.shape[1]}×{crop.shape[0]}px")

            raw_text = run_ocr(crop, label)

            if label == "AADHAR_NUMBER":
                cleaned = clean_aadhar_number(raw_text)
            elif label == "DATE_OF_BIRTH":
                cleaned = clean_dob(raw_text)
            elif label == "GENDER":
                cleaned = clean_gender(raw_text)
            elif label == "NAME":
                cleaned = extract_name_from_text(raw_text)
                if not cleaned:
                    cleaned = raw_text.strip()
                if not is_valid_name(cleaned):
                    cleaned = ""
            else:
                cleaned = raw_text.strip()

            if label in extracted and cleaned and confidence > best_confidence.get(label, 0):
                extracted[label] = cleaned
                best_confidence[label] = confidence

            detections.append({
                "label": label,
                "text": cleaned,
                "raw_text": raw_text,
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
            })

    # ✅ Fallback: run full card OCR for any missing fields
    missing = [k for k, v in extracted.items() if not v]
    if missing:
        print(f"⚠️ Missing fields {missing}, running full card OCR fallback...")

        # ✅ Use original resolution for fallback too
        fallback = full_card_ocr_fallback(original)
        print(f"📋 Fallback result: {fallback}")

        for field in missing:
            if fallback.get(field):
                extracted[field] = fallback[field]
                print(f"   ✅ Fallback filled {field}: {fallback[field]}")

        # ✅ Special targeted fallback for Aadhar number
        if not extracted["AADHAR_NUMBER"]:
            print("🔍 Running targeted Aadhar region OCR...")
            aadhar_result = ocr_aadhar_region(original)
            if aadhar_result:
                extracted["AADHAR_NUMBER"] = aadhar_result
                print(f"   ✅ Aadhar number found: {aadhar_result}")

    return JSONResponse(content={
        "success": True,
        "extracted": extracted,
        "all_detections": detections,
        "model_classes": list(model.names.values()),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)