"""
pipeline.py
Cloud-side OCR pipeline — Render Free Tier Optimized.

- EasyOCR initialized ONCE at module load (not lazy, not inside request handlers).
- OCR runs on saved JPEG file path (more reliable than raw numpy arrays).
- Added OCR timeout via concurrent.futures to prevent hanging on corrupt images.
- No YOLO. No ultralytics. No torch on cloud.
"""

import os
import logging
import concurrent.futures
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger("Pipeline")

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Global OCR reader (initialized ONCE at import, reused for all requests) ──
import easyocr
logger.info("Loading EasyOCR reader (CPU)...")
_ocr_reader = easyocr.Reader(["en"], gpu=False)
logger.info("EasyOCR reader ready.")


def _read_plate(image_path: str) -> str | None:
    """
    Run EasyOCR on a saved image file.
    Uses a 30-second timeout via thread pool to prevent hanging.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_ocr_reader.readtext, image_path)
        try:
            ocr_result = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            logger.error("OCR timed out on %s", image_path)
            return None

    candidates = []
    for _bbox, text, score in ocr_result:
        clean = "".join(c for c in text if c.isalnum()).upper()
        if len(clean) > 3 and score > 0.2:
            candidates.append({"text": clean, "confidence": score})

    if not candidates:
        return None

    best = max(candidates, key=lambda x: x["confidence"])
    return best["text"]


class OCRPipeline:
    """
    Minimal cloud pipeline:
      1. Save uploaded image to static/uploads/.
      2. Run EasyOCR on the saved file.
      3. Return structured result for challan generation.
    """

    def process_upload(self, image_np: np.ndarray, traffic_level: str = "UNKNOWN") -> dict:
        timestamp  = datetime.now().strftime("%Y%m%d%H%M%S")
        wide_fname = f"wide_{timestamp}.jpg"
        wide_path  = os.path.join(UPLOAD_DIR, wide_fname)
        wide_url   = f"/static/uploads/{wide_fname}"

        cv2.imwrite(wide_path, image_np)

        action          = "OCR Failed"
        plate_text      = None
        plate_url       = None
        decision_status = "Running OCR..."

        try:
            plate_text = _read_plate(wide_path)

            if plate_text:
                action      = "Challan Generated"
                plate_fname = f"plate_{plate_text}_{timestamp}.jpg"
                plate_path  = os.path.join(UPLOAD_DIR, plate_fname)
                plate_url   = f"/static/uploads/{plate_fname}"
                cv2.imwrite(plate_path, image_np)
                decision_status = f"Plate detected: {plate_text}. Challan generated."
            else:
                action          = "OCR Failed"
                decision_status = "No readable plate found."

        except Exception as exc:
            logger.exception("OCR error: %s", exc)
            action          = "OCR Failed"
            decision_status = f"Pipeline error: {exc}"

        return {
            "status":          decision_status,
            "action":          action,
            "plate_number":    plate_text,
            "wide_image_url":  wide_url,
            "plate_image_url": plate_url,
            "traffic_level":   traffic_level,
            "density":         0.0,
            "free_space":      0,
        }


pipeline_engine = OCRPipeline()
