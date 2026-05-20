"""
pi_client.py — Raspberry Pi Edge Node Client
Emergency Vehicle Priority System — v6 (Edge OCR + Plate Crop)

Run this script on the Raspberry Pi.

Responsibilities:
  1. Capture image from Pi Camera (or USB webcam).
  2. Run YOLOv8n ONNX locally to detect traffic level.
  3. Run EasyOCR locally to read license plate + get bounding box.
  4. Crop number plate region from the frame.
  5. Upload vehicle_image + plate_image + plate_number + traffic_level.

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
SERVER_URL        = "https://evps-backend.onrender.com/api/upload"
ONNX_MODEL_PATH   = "yolov8n.onnx"
CAPTURE_INTERVAL  = 5
CAMERA_INDEX      = 0
CONFIDENCE_THRESH = 0.4
VEHICLE_CLASSES   = {2, 3, 5, 7}
HIGH_TRAFFIC_THRESHOLD    = 8
MEDIUM_TRAFFIC_THRESHOLD  = 3
UPLOAD_TIMEOUT    = 80

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
        blob    = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
        return blob

    def detect_vehicles(self, img_bgr: np.ndarray) -> int:
        blob    = self.preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        preds   = outputs[0][0]
        vehicle_count = 0
        for det in preds:
            if det[4] < CONFIDENCE_THRESH:
                continue
            class_id = int(np.argmax(det[5:]))
            if class_id in VEHICLE_CLASSES:
                vehicle_count += 1
        return vehicle_count

    def classify_traffic(self, img_bgr: np.ndarray) -> str:
        count = self.detect_vehicles(img_bgr)
        logger.info("Vehicles detected: %d", count)
        if count >= HIGH_TRAFFIC_THRESHOLD:
            return "HIGH"
        elif count >= MEDIUM_TRAFFIC_THRESHOLD:
            return "MEDIUM"
        else:
            return "LOW"


# ── Local OCR Plate Reader + Crop ─────────────────────────────────────────────

class LocalPlateReader:
    """Runs EasyOCR on-device; returns plate text + cropped plate image."""

    def __init__(self):
        import easyocr
        logger.info("Loading EasyOCR reader (Pi CPU)...")
        self.reader = easyocr.Reader(["en"], gpu=False)
        logger.info("EasyOCR reader ready.")

    def read_plate(self, img_bgr: np.ndarray):
        """
        Returns (plate_text, cropped_plate_bgr) or (None, None).
        Crops the plate region from the original frame using EasyOCR's bounding box.
        """
        try:
            result = self.reader.readtext(img_bgr)
        except Exception as exc:
            logger.warning("OCR failed: %s", exc)
            return None, None

        candidates = []
        for bbox, text, score in result:
            clean = "".join(c for c in text if c.isalnum()).upper()
            if len(clean) > 3 and score > 0.2:
                candidates.append({"text": clean, "confidence": score, "bbox": bbox})

        if not candidates:
            return None, None

        best = max(candidates, key=lambda x: x["confidence"])
        pts = np.array(best["bbox"], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        # Add small padding
        pad = 4
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img_bgr.shape[1] - x, w + pad * 2)
        h = min(img_bgr.shape[0] - y, h + pad * 2)
        cropped = img_bgr[y:y+h, x:x+w]

        return best["text"], cropped


# ── Upload to Flask backend ───────────────────────────────────────────────────

def upload_frame(img_bgr: np.ndarray, traffic_level: str,
                 plate_text: str | None, plate_crop: np.ndarray | None) -> bool:
    """
    Upload vehicle_image + plate_image + plate_number + traffic_level.
    Cloud backend no longer runs any OCR — it receives pre-processed data.
    """
    success_veh, encoded_veh = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success_veh:
        logger.error("Failed to encode vehicle image.")
        return False

    files = {
        "vehicle_image": ("vehicle.jpg", encoded_veh.tobytes(), "image/jpeg"),
    }

    if plate_crop is not None:
        success_pl, encoded_pl = cv2.imencode(".jpg", plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success_pl:
            files["plate_image"] = ("plate.jpg", encoded_pl.tobytes(), "image/jpeg")

    post_data = {"traffic_level": traffic_level}
    if plate_text:
        post_data["plate_number"] = plate_text

    try:
        response = requests.post(
            SERVER_URL,
            files=files,
            data=post_data,
            timeout=UPLOAD_TIMEOUT,
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

        plate_text  = None
        plate_crop  = None
        if plate_reader is not None:
            try:
                plate_text, plate_crop = plate_reader.read_plate(frame)
                logger.info("Plate: %s | cropped: %s",
                            plate_text or "None",
                            f"{plate_crop.shape[1]}x{plate_crop.shape[0]}" if plate_crop is not None else "None")
            except Exception as exc:
                logger.warning("Plate read error: %s", exc)

        logger.info("Uploading — traffic: %s, plate: %s", traffic_level, plate_text)
        upload_frame(frame, traffic_level, plate_text, plate_crop)

        time.sleep(CAPTURE_INTERVAL)

    cap.release()


if __name__ == "__main__":
    main()
