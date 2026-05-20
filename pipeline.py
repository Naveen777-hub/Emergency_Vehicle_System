"""
pipeline.py
Cloud-side fallback pipeline — NO EasyOCR / PyTorch on Render.

OCR now runs on the Raspberry Pi (edge). The cloud pipeline is only
a fallback for legacy Pi clients that don't send plate_text.

- Saves uploaded image to static/uploads/.
- Returns structured result (OCR is done on Pi).
- No EasyOCR. No torch. No YOLO. Minimal memory footprint.
"""

import os
import logging
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger("Pipeline")

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class OCRPipeline:
    """Minimal cloud pipeline (fallback only — OCR runs on Pi)."""

    def process_upload(self, image_np: np.ndarray, traffic_level: str = "UNKNOWN") -> dict:
        timestamp  = datetime.now().strftime("%Y%m%d%H%M%S")
        wide_fname = f"wide_{timestamp}.jpg"
        wide_path  = os.path.join(UPLOAD_DIR, wide_fname)
        wide_url   = f"/static/uploads/{wide_fname}"

        cv2.imwrite(wide_path, image_np)

        return {
            "status":          "No Pi plate_text — cloud processing skipped",
            "action":          "OCR Failed",
            "plate_number":    None,
            "wide_image_url":  wide_url,
            "plate_image_url": None,
            "traffic_level":   traffic_level,
            "density":         0.0,
            "free_space":      0,
        }


pipeline_engine = OCRPipeline()
