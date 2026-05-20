"""
app.py
Flask application entry point — Render Free Tier Optimized.

Architecture:
  - Raspberry Pi runs traffic detection (YOLOv8n ONNX) locally.
  - Pi uploads images ONLY for LOW/MEDIUM traffic.
  - Cloud backend: EasyOCR (initialized after import to keep startup RAM low)
                  → challan DB → dashboard.
  - /api/upload accepts image + traffic_level, runs OCR, saves challan.
  - Render-ready: SECRET_KEY / DATABASE_URL from env vars.
  - SQLite local / PostgreSQL on Render.
"""

import os
import time
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

# ── Auto-setup: create tables, migrate columns, seed admin ────────────────
def _ensure_database():
    """Idempotent startup initializer:
    1. Create all tables if they don't exist (handles fresh PostgreSQL on Render).
    2. Add missing columns to existing tables.
    3. Seed default admin user if not present.
    """
    from sqlalchemy import inspect
    with app.app_context():
        inspector = inspect(db.engine)
        existing_tables = set(inspector.get_table_names())
        model_tables    = set(db.metadata.tables.keys())

        # 1. Create missing tables
        if not existing_tables:
            db.create_all()
            logger.info("Created all database tables.")
            existing_tables = model_tables
        elif missing := model_tables - existing_tables:
            db.create_all()
            logger.info("Created missing tables: %s", missing)
            existing_tables |= missing

        # 2. Add missing columns to existing tables
        for table_name, table in db.metadata.tables.items():
            if table_name not in existing_tables:
                continue
            db_cols = {c["name"] for c in inspector.get_columns(table_name)}
            for col in table.columns:
                if col.name not in db_cols:
                    col_type = col.type.compile(db.engine.dialect)
                    sql = f"ALTER TABLE {table_name} ADD COLUMN {col.name} {col_type}"
                    if col.default is not None:
                        sql += f" DEFAULT {col.default.arg}"
                    db.session.execute(sql)
                    logger.info("Auto-migrated: added column '%s' to '%s'", col.name, table_name)

        # 3. Seed admin user if missing
        if not User.query.filter_by(username="admin").first():
            admin = User(
                username="admin", full_name="System Administrator",
                email="admin@evps.gov.in", role="admin", is_active=True,
            )
            admin.set_password("admin123")
            db.session.add(admin)
            logger.info("Seeded default admin user.")

        db.session.commit()

_db_ready = False
_ocr_ready = False
try:
    _ensure_database()
    _db_ready = True
except Exception as exc:
    logger.warning("Database init deferred (will retry on first request): %s", exc)

try:
    pipeline_engine.init_ocr()
    _ocr_ready = True
except Exception as exc:
    logger.warning("OCR init deferred (will init on first OCR request): %s", exc)


@app.before_request
def _ensure_db_and_ocr_on_request():
    """Retry database and OCR init on first request if deferred at startup."""
    global _db_ready, _ocr_ready
    if not _db_ready:
        try:
            _ensure_database()
            _db_ready = True
        except Exception as exc:
            logger.error("Database still unavailable: %s", exc)
    if not _ocr_ready:
        try:
            pipeline_engine.init_ocr()
            _ocr_ready = True
        except Exception as exc:
            logger.warning("OCR still unavailable: %s", exc)


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
    Raspberry Pi upload endpoint.

    Pi already:
      1. Captured image.
      2. Ran local YOLOv8n ONNX traffic detection.
      3. Determined traffic level (LOW / MEDIUM / HIGH).
      4. Uploads only LOW or MEDIUM frames.

    This endpoint:
      1. Reads + decodes image.
      2. Runs EasyOCR (initialized at startup, not inside request handlers).
      3. Saves challan to database.
      4. Returns result JSON.

    Multipart fields:
      image         — JPEG/PNG file
      traffic_level — "LOW" | "MEDIUM"
    """
    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file field."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty file uploaded."}), 400

    traffic_level = request.form.get("traffic_level", "UNKNOWN").upper().strip()

    try:
        img_bytes = file.read()
        img_np    = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_np is None:
            return jsonify({"error": "Could not decode image."}), 400
    except Exception as exc:
        return jsonify({"error": f"Image decode failed: {exc}"}), 500

    try:
        result = pipeline_engine.process_upload(img_np, traffic_level)
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        return jsonify({"error": f"Pipeline failed: {exc}"}), 500

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
        except Exception as exc:
            logger.exception("DB write error: %s", exc)
            db.session.rollback()

    latest_result_store["timestamp"] = time.time()
    latest_result_store["data"]      = result

    elapsed = time.time() - start
    logger.info("Upload processed in %.2fs — action=%s plate=%s", elapsed,
                result.get("action"), result.get("plate_number"))

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
        logger.info("Cleaned %d old upload(s)", removed)


_cleanup_old_uploads()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0", use_reloader=False)
