"""
pipeline.py
Cloud-side OCR pipeline (v5 — Render Free Tier Optimized).

Key changes from v4:
  - Removed YOLOv8 / ultralytics completely (was using 300MB+ RAM).
  - EasyOCR runs directly on the full uploaded image.
  - OCR reader is LAZY-LOADED on first request (not at startup).
  - Startup RAM: ~50MB. OCR RAM: ~300MB (loaded only when first image arrives).
  - Render free tier (512MB) can handle this comfortably.
"""

import os
import cv2
import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger("Pipeline")

# ── Upload directory ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Lazy OCR loader ───────────────────────────────────────────────────────────
_ocr_reader = None  # Not loaded at startup — loaded on first request

def _get_ocr_reader():
    """
    Returns a shared EasyOCR reader instance.
    Loads it on first call (lazy loading) to keep startup RAM low.
    """
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        logger.info("Lazy-loading EasyOCR reader (CPU)...")
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
        logger.info("EasyOCR reader ready.")
    return _ocr_reader


# ── OCR on full image ─────────────────────────────────────────────────────────
def _read_plate(image_np: np.ndarray) -> str | None:
    """
    Run EasyOCR on the full uploaded image.
    Passes the raw BGR image directly (EasyOCR handles its own preprocessing).
    Single-channel images cause 'model does not support image input' errors.
    """
    reader     = _get_ocr_reader()
    ocr_result = reader.readtext(image_np)

    candidates = []
    for _bbox, text, score in ocr_result:
        clean = "".join(c for c in text if c.isalnum()).upper()
        if len(clean) > 3:
            candidates.append({"text": clean, "confidence": score})

    if not candidates:
        return None

    best = max(candidates, key=lambda x: x["confidence"])
    return best["text"]


# ── Main pipeline callable ────────────────────────────────────────────────────
class OCRPipeline:
    """
    Minimal cloud pipeline:
      1. Save uploaded image to static/uploads/.
      2. Run EasyOCR on the image.
      3. Return structured result for challan generation.
    No YOLO. No ultralytics. No torch. Just EasyOCR + OpenCV.
    """

    def process_upload(self, image_np: np.ndarray, traffic_level: str = "UNKNOWN") -> dict:
        """
        Process a single frame uploaded by the Raspberry Pi.

        Args:
            image_np:      Decoded OpenCV image (BGR).
            traffic_level: 'LOW' | 'MEDIUM' (HIGH frames are never uploaded by Pi).

        Returns:
            dict with status, action, plate_number, wide_image_url, plate_image_url
        """
        timestamp  = datetime.now().strftime("%Y%m%d%H%M%S")
        wide_fname = f"wide_{timestamp}.jpg"
        wide_path  = os.path.join(UPLOAD_DIR, wide_fname)
        wide_url   = f"/static/uploads/{wide_fname}"

        # Save raw uploaded image
        cv2.imwrite(wide_path, image_np)
        logger.info("Saved uploaded frame → %s", wide_path)

        action          = "OCR Failed"
        plate_text      = None
        plate_url       = None
        decision_status = f"Traffic: {traffic_level}. Running OCR on uploaded image."

        try:
            plate_text = _read_plate(image_np)

            if plate_text:
                action      = "Challan Generated"
                plate_fname = f"plate_{plate_text}_{timestamp}.jpg"
                plate_path  = os.path.join(UPLOAD_DIR, plate_fname)
                plate_url   = f"/static/uploads/{plate_fname}"
                cv2.imwrite(plate_path, image_np)
                decision_status = (
                    f"Traffic: {traffic_level}. "
                    f"Plate detected: {plate_text}. Challan generated."
                )
                logger.info("OCR success — plate: %s", plate_text)
            else:
                action          = "OCR Failed"
                decision_status = f"Traffic: {traffic_level}. No readable plate found in image."
                logger.warning("OCR returned no readable plate.")

        except Exception as exc:
            logger.exception("OCR pipeline error: %s", exc)
            action          = "OCR Failed"
            decision_status = f"Pipeline error: {exc}"

        return {
            "status":          decision_status,
            "action":          action,
            "plate_number":    plate_text,
            "wide_image_url":  wide_url,
            "plate_image_url": plate_url,
            "traffic_level":   traffic_level,
            # Legacy fields kept for DB column compatibility
            "density":         0.0,
            "free_space":      0,
        }


# ── Global singleton ──────────────────────────────────────────────────────────
# Lightweight at startup — OCR loads lazily on first /api/upload request.
pipeline_engine = OCRPipeline()
