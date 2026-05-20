"""
pipeline.py
Cloud-side OCR pipeline — Render Free Tier Optimized.

- EasyOCR initialized EXPLICITLY via init_ocr(), NOT at module import time.
  This keeps import-time memory low so Render's 512MB free tier doesn't OOM.
- Call pipeline_engine.init_ocr() from app startup to warm up OCR eagerly.
- OCR runs on in-memory numpy array (no filesystem dependency), falls back to file path.
- gc.collect() after each OCR run to keep memory low.
- No YOLO. No ultralytics. No torch on cloud.
"""

import os
import gc
import logging
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger("Pipeline")

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── OCR reader (lazily initialized; call init_ocr() to warm up) ──
_ocr_reader = None


def init_ocr():
    """Initialize the global EasyOCR reader. Call once at app startup."""
    global _ocr_reader
    if _ocr_reader is not None:
        return
    import easyocr
    logger.info("Loading EasyOCR reader (CPU)...")
    _ocr_reader = easyocr.Reader(["en"], gpu=False)
    logger.info("EasyOCR reader ready.")


def _ensure_ocr():
    """Ensure OCR is initialized (lazy fallback if init_ocr() wasn't called)."""
    if _ocr_reader is None:
        init_ocr()


def _read_plate(image_np: np.ndarray, image_path: str | None = None) -> str | None:
    """
    Run EasyOCR on a numpy array (in-memory), falling back to file path.
    Runs synchronously — no extra thread pools (avoids memory overhead).
    """
    _ensure_ocr()

    result = None

    # Strategy 1: numpy array directly (no filesystem dependency)
    try:
        result = _ocr_reader.readtext(image_np)
    except Exception as exc:
        logger.warning("OCR via numpy failed: %s", exc)

    # Strategy 2: fallback to saved file path
    if not result and image_path and os.path.isfile(image_path):
        logger.info("Falling back to file-path OCR: %s", image_path)
        try:
            result = _ocr_reader.readtext(image_path)
        except Exception as exc:
            logger.warning("OCR via file path also failed: %s", exc)

    gc.collect()

    if not result:
        return None

    candidates = []
    for _bbox, text, score in result:
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

    def init_ocr(self):
        """Warm up EasyOCR. Call at app startup."""
        init_ocr()

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
            plate_text = _read_plate(image_np, wide_path)

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
