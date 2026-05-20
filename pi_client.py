"""
pi_client.py — Raspberry Pi Edge Node Client
Emergency Vehicle Priority System — v5 (Edge OCR)

Run this script on the Raspberry Pi.

Responsibilities:
  1. Capture image from Pi Camera (or USB webcam).
  2. Run YOLOv8n ONNX locally to detect traffic level.
  3. Run EasyOCR locally to read license plate.
  4. Upload image + traffic_level + plate_text to Flask backend.

Requirements (on Pi):
  pip install requests opencv-python onnxruntime numpy easyocr

ONNX model export (run once on a machine with ultralytics):
  yolo export model=yolov8n.pt format=onnx imgsz=640
"""

import cv2
import time
import logging
import requests
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
SERVER_URL        = "https://evps-backend.onrender.com/api/upload"  # ← Your Render URL
ONNX_MODEL_PATH   = "yolov8n.onnx"     # Local ONNX model on Pi
CAPTURE_INTERVAL  = 5                  # Seconds between captures
CAMERA_INDEX      = 0                  # 0 = default webcam / Pi cam
CONFIDENCE_THRESH = 0.4
VEHICLE_CLASSES   = {2, 3, 5, 7}       # COCO: car, motorcycle, bus, truck
HIGH_TRAFFIC_THRESHOLD    = 8          # Vehicles count → HIGH
MEDIUM_TRAFFIC_THRESHOLD  = 3          # Vehicles count → MEDIUM (below = LOW)
UPLOAD_TIMEOUT    = 80                 # Seconds to wait for cloud response

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Pi-Client")


# ── ONNX Traffic Detector ─────────────────────────────────────────────────────

class LocalTrafficDetector:
    """Runs YOLOv8n ONNX on-device for traffic classification."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        logger.info("Loading ONNX model: %s", model_path)
        self.session    = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        logger.info("ONNX model loaded.")

    def preprocess(self, img_bgr: np.ndarray, input_size: int = 640) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (input_size, input_size))
        blob    = img_res.astype(np.float32) / 255.0
        blob    = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # NCHW
        return blob

    def detect_vehicles(self, img_bgr: np.ndarray) -> int:
        """Returns count of vehicles detected."""
        blob    = self.preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        preds   = outputs[0][0]          # shape: (8400, 85) for YOLOv8n
        vehicle_count = 0

        for det in preds:
            conf = det[4]
            if conf < CONFIDENCE_THRESH:
                continue
            class_scores = det[5:]
            class_id     = int(np.argmax(class_scores))
            if class_id in VEHICLE_CLASSES:
                vehicle_count += 1

        return vehicle_count

    def classify_traffic(self, img_bgr: np.ndarray) -> str:
        """Returns 'LOW', 'MEDIUM', or 'HIGH'."""
        count = self.detect_vehicles(img_bgr)
        logger.info("Vehicles detected: %d", count)

        if count >= HIGH_TRAFFIC_THRESHOLD:
            return "HIGH"
        elif count >= MEDIUM_TRAFFIC_THRESHOLD:
            return "MEDIUM"
        else:
            return "LOW"


# ── Local OCR Plate Reader ────────────────────────────────────────────────────

class LocalPlateReader:
    """Runs EasyOCR on-device for license plate recognition."""

    def __init__(self):
        import easyocr
        logger.info("Loading EasyOCR reader (Pi CPU)...")
        self.reader = easyocr.Reader(["en"], gpu=False)
        logger.info("EasyOCR reader ready.")

    def read_plate(self, img_bgr: np.ndarray) -> str | None:
        """Returns best plate text from image, or None."""
        try:
            result = self.reader.readtext(img_bgr)
        except Exception as exc:
            logger.warning("OCR failed: %s", exc)
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


# ── Upload to Flask backend ───────────────────────────────────────────────────

def upload_frame(img_bgr: np.ndarray, traffic_level: str, plate_text: str | None) -> bool:
    """
    Encode image as JPEG and POST to Flask backend.
    Sends plate_text if available (edge OCR); cloud skips OCR if plate_text is present.
    """
    success, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        logger.error("Failed to encode image as JPEG.")
        return False

    img_bytes = encoded.tobytes()
    post_data = {"traffic_level": traffic_level}
    if plate_text:
        post_data["plate_text"] = plate_text

    try:
        response = requests.post(
            SERVER_URL,
            files  = {"image": ("frame.jpg", img_bytes, "image/jpeg")},
            data   = post_data,
            timeout = UPLOAD_TIMEOUT,
        )
        if response.status_code == 200:
            result = response.json()
            logger.info(
                "Upload OK — action: %s | plate: %s | challan: %s",
                result.get("action"),
                result.get("plate_number"),
                result.get("challan_number"),
            )
            return True
        else:
            logger.warning("Server returned %d: %s", response.status_code, response.text)
            return False

    except requests.exceptions.RequestException as exc:
        logger.error("Upload failed: %s", exc)
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    detector = LocalTrafficDetector(ONNX_MODEL_PATH)

    try:
        plate_reader = LocalPlateReader()
    except Exception as exc:
        logger.error("Failed to init EasyOCR (out of memory?). Running without OCR: %s", exc)
        plate_reader = None

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        logger.error("Cannot open camera (index %d). Check connection.", CAMERA_INDEX)
        return

    logger.info("Pi client started. Capturing every %ds. Server: %s", CAPTURE_INTERVAL, SERVER_URL)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame capture failed. Retrying...")
            time.sleep(2)
            continue

        traffic_level = detector.classify_traffic(frame)
        logger.info("Traffic level: %s", traffic_level)

        if traffic_level == "HIGH":
            logger.info("HIGH traffic detected — skipping upload.")
            time.sleep(CAPTURE_INTERVAL)
            continue

        # Read plate on Pi before uploading
        plate_text = None
        if plate_reader is not None:
            try:
                plate_text = plate_reader.read_plate(frame)
                logger.info("Plate detected: %s", plate_text or "None")
            except Exception as exc:
                logger.warning("Plate read error: %s", exc)

        logger.info("Uploading frame (traffic: %s, plate: %s)...", traffic_level, plate_text)
        upload_frame(frame, traffic_level, plate_text)

        time.sleep(CAPTURE_INTERVAL)

    cap.release()


if __name__ == "__main__":
    main()
