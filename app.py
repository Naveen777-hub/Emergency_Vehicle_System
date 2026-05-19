"""
app.py
Flask application entry point — v5 (Edge-Cloud Architecture, Render Optimized).

Architecture:
  - Traffic detection runs on Raspberry Pi (edge node) via YOLOv8n ONNX.
  - Pi uploads images ONLY for LOW/MEDIUM traffic; HIGH frames are skipped locally.
  - Cloud backend runs EasyOCR directly on uploaded images — NO YOLO/ultralytics/torch.
  - OCR is lazy-loaded on first request (not at startup) to keep RAM low (~50MB idle).
  - /api/upload accepts a single image + traffic_level metadata.
  - Render-ready: SECRET_KEY and DATABASE_URL read from environment variables.
  - PostgreSQL supported via DATABASE_URL env var (SQLite fallback for local dev).
"""

import os
import logging
from datetime import datetime

from flask import Flask, request, jsonify, redirect, url_for, render_template
from flask_login import LoginManager, current_user
import cv2
import numpy as np

from database import db, User, Vehicle, Challan
from pipeline import pipeline_engine

# ── Logging (single root config — pipeline.py uses child loggers) ──────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("PIL").setLevel(logging.WARNING)   # Pillow noise
logging.getLogger("easyocr").setLevel(logging.WARNING)  # EasyOCR noise
logging.getLogger("torch").setLevel(logging.WARNING)    # Torch noise
logger = logging.getLogger("App")

# ── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Configuration (Render-safe) ───────────────────────────────────────────────
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "evps-dev-key-change-in-production")

# ── Production session security ─────────────────────────────────────────────
app.config["SESSION_COOKIE_HTTPONLY"]  = True
app.config["SESSION_COOKIE_SAMESITE"]  = "Lax"
app.config["SESSION_COOKIE_SECURE"]    = bool(os.environ.get("SESSION_COOKIE_SECURE", "True").lower() == "true")
app.config["PERMANENT_SESSION_LIFETIME"] = 86400  # 24 hours

# PostgreSQL on Render via DATABASE_URL env var; SQLite fallback for local dev.
_raw_db_url = os.environ.get("DATABASE_URL", "")
if _raw_db_url.startswith("postgres://"):
    _raw_db_url = _raw_db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    _raw_db_url
    if _raw_db_url
    else f"sqlite:///{os.path.join(DATA_DIR, 'evs.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"]             = 16 * 1024 * 1024  # 16 MB

# ── Connection pooling for Render free tier (limited connections) ──────────
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_size": 5,
    "pool_recycle": 3600,
    "pool_pre_ping": True,
}

# ── Extensions ────────────────────────────────────────────────────────────────
db.init_app(app)

# ── Auto-migration: add missing columns to existing tables ──────────────────
def _auto_migrate():
    """Add columns to existing tables that exist in models but not in DB.
    Skips missing tables gracefully (for first-time init via init_db.py)."""
    from sqlalchemy import inspect
    with app.app_context():
        inspector = inspect(db.engine)
        existing_tables = set(inspector.get_table_names())
        for table_name, table in db.metadata.tables.items():
            if table_name not in existing_tables:
                continue
            existing_cols = {c["name"] for c in inspector.get_columns(table_name)}
            for col in table.columns:
                if col.name not in existing_cols:
                    col_type = col.type.compile(db.engine.dialect)
                    nullable = "NULL" if col.nullable else "NOT NULL"
                    chunks = f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type} {nullable}"
                    if col.default is not None:
                        chunks += f" DEFAULT {col.default.arg}"
                    db.session.execute(chunks)
                    logger.info("Auto-migrated: added column '%s' to table '%s'", col.name, table_name)
        db.session.commit()

_auto_migrate()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view             = "auth.login"
login_manager.login_message          = "Please log in to access this page."
login_manager.login_message_category = "warning"


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ── Blueprints ────────────────────────────────────────────────────────────────
from auth import auth_bp
from blueprints.admin import admin_bp
from blueprints.user import user_bp

app.register_blueprint(auth_bp)
app.register_blueprint(admin_bp)
app.register_blueprint(user_bp)


# ── Jinja2 global context ─────────────────────────────────────────────────────
@app.context_processor
def inject_globals():
    now = datetime.now()
    return {
        "current_date": now.strftime("%d %B %Y"),
        "current_year": now.year,
    }


# ── Root redirect ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.role == "admin":
            return redirect(url_for("admin.dashboard"))
        return redirect(url_for("user.dashboard"))
    return redirect(url_for("auth.login"))


# ── Shared live-feed state ────────────────────────────────────────────────────
# Stores the most recent pipeline result for the live dashboard to poll.
latest_result_store: dict = {"timestamp": None, "data": None}


# ── Raspberry Pi upload endpoint ──────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def process_pipeline():
    """
    Raspberry Pi upload endpoint (v5 — Render-optimized, synchronous).

    The Pi has already:
      1. Captured the image.
      2. Run local traffic detection (YOLOv8n ONNX).
      3. Determined traffic level (LOW / MEDIUM / HIGH).
      4. Decided to upload (only LOW or MEDIUM reach this endpoint).

    This endpoint:
      1. Reads the uploaded image.
      2. Reads the traffic_level metadata.
      3. Runs EasyOCR directly on the image (lazy-loaded, no YOLO on cloud).
      4. Saves challan to database.
      5. Returns result JSON.

    Expected multipart fields:
      image         — JPEG/PNG image file
      traffic_level — string: "LOW" | "MEDIUM"
    """
    # ── Validate input ────────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty file uploaded."}), 400

    traffic_level = request.form.get("traffic_level", "UNKNOWN").upper().strip()
    if traffic_level not in {"LOW", "MEDIUM", "UNKNOWN"}:
        # Accept but warn — Pi may be misconfigured
        logger.warning("Unexpected traffic_level value received: '%s'", traffic_level)

    # ── Decode image ──────────────────────────────────────────────────────────
    try:
        img_bytes = file.read()
        img_np    = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_np is None:
            return jsonify({"error": "Could not decode image. Send a valid JPEG/PNG."}), 400
    except Exception as exc:
        logger.exception("Image decode error: %s", exc)
        return jsonify({"error": f"Image decode failed: {exc}"}), 500

    # ── Run OCR pipeline ──────────────────────────────────────────────────────
    try:
        result = pipeline_engine.process_upload(img_np, traffic_level)
        logger.info("Pipeline result: action=%s plate=%s traffic=%s",
                    result.get("action"), result.get("plate_number"), traffic_level)
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        return jsonify({"error": f"Pipeline failed: {exc}"}), 500

    # ── Persist challan to DB ─────────────────────────────────────────────────
    challan_number = None
    if result.get("action") in {"Challan Generated", "OCR Attempted", "OCR Failed"}:
        try:
            plate_number = result.get("plate_number") or "UNKNOWN"
            vehicle      = Vehicle.query.filter_by(plate_number=plate_number).first()
            challan_number = f"EVS{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:19]}"

            challan = Challan(
                challan_number  = challan_number,
                plate_number    = plate_number,
                vehicle_id      = vehicle.id if vehicle else None,
                action          = result.get("action"),
                traffic_level   = traffic_level,
                density_pct     = result.get("density", 0),
                free_space_px   = result.get("free_space", 0),
                pipeline_status = result.get("status"),
                wide_image_url  = result.get("wide_image_url"),
                plate_image_url = result.get("plate_image_url"),
                status          = "Pending",
                amount          = 500.0,
            )
            db.session.add(challan)
            db.session.commit()
            logger.info("Challan %s saved to database.", challan_number)
        except Exception as exc:
            logger.exception("DB write error: %s", exc)
            db.session.rollback()

    # ── Update live-feed state ────────────────────────────────────────────────
    import time
    latest_result_store["timestamp"] = time.time()
    latest_result_store["data"]      = result

    return jsonify({
        "status":          "processed",
        "action":          result.get("action"),
        "plate_number":    result.get("plate_number"),
        "challan_number":  challan_number,
        "traffic_level":   traffic_level,
        "wide_image_url":  result.get("wide_image_url"),
        "plate_image_url": result.get("plate_image_url"),
        "pipeline_status": result.get("status"),
    }), 200


# ── Live-feed polling endpoint ────────────────────────────────────────────────
@app.route("/api/latest", methods=["GET"])
def get_latest_result():
    """Returns the most recent pipeline result as JSON (polled by admin live monitor)."""
    return jsonify(latest_result_store)


# ── Health check (Render / uptime monitors) ───────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health_check():
    """Simple health endpoint. Returns 200 if the app is alive."""
    return jsonify({"status": "ok", "service": "EVPS Flask Backend"}), 200


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return render_template("errors/404.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception("Internal server error: %s", e)
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error"}), 500
    return render_template("errors/500.html"), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


# ── Periodic old upload cleanup (runs once at startup) ──────────────────────
def _cleanup_old_uploads(max_age_hours: int = 24):
    """Remove uploaded images older than max_age_hours to prevent disk bloat."""
    import time
    now = time.time()
    cutoff = now - (max_age_hours * 3600)
    removed = 0
    if os.path.isdir(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            if fname == ".gitkeep":
                continue
            fpath = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                os.remove(fpath)
                removed += 1
    if removed:
        logger.info("Cleaned %d old upload(s) (threshold: %dh)", removed, max_age_hours)


_cleanup_old_uploads()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=False)
