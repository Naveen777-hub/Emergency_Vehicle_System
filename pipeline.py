"""
pipeline.py
Cloud-side fallback pipeline — NO processing on Render.

All AI/OCR now runs on the Raspberry Pi (edge).
The cloud pipeline is a no-op passthrough kept for structural compatibility.
"""

import logging

logger = logging.getLogger("Pipeline")


class OCRPipeline:
    """No-op pipeline — all processing happens on the Raspberry Pi."""

    def process_upload(self, image_np=None, traffic_level="UNKNOWN"):
        return {
            "status":          "Legacy fallback — Pi should send plate_number directly",
            "action":          "OCR Failed",
            "plate_number":    None,
            "wide_image_url":  None,
            "plate_image_url": None,
            "traffic_level":   traffic_level,
            "density":         0.0,
            "free_space":      0,
        }


pipeline_engine = OCRPipeline()
