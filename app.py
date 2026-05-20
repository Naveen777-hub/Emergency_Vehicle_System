"""
app.py
Flask application entry point — Render Free Tier Optimized.

Architecture:
  - Raspberry Pi runs ALL processing locally:
      YOLOv8n ONNX (traffic) + EasyOCR (plate recognition + crop).
  - Pi uploads vehicle_image + plate_image + plate_number + traffic_level.
  - Cloud backend: saves evidence, creates challan record, serves dashboard.
  - No EasyOCR / PyTorch / YOLO on cloud — keeps Render 512MB happy.
  - /api/upload accepts vehicle_image, plate_image, plate_number, traffic_level.
  - Render-ready: SECRET_KEY / DATABASE_URL from env vars.
  - SQLite local / PostgreSQL on Render.
"""

import os
import time
import logging
from datetime import datetime

from flask import Flask, request, jsonify, redirect, url_for, render_template
from flask_login import LoginManager, current_user

from database import db, User, Vehicle, Challan

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger("App")

# ── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR   = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "evps-dev-key-change-in-production")

app.config["SESSION_COOKIE_HTTPONLY"]  = True
app.config["SESSION_COOKIE_SAMESITE"]  = "Lax"
app.config["SESSION_COOKIE_SECURE"]    = bool(os.environ.get("SESSION_COOKIE_SECURE", "True").lower() == "true")
app.config["PERMANENT_SESSION_LIFETIME"] = 86400

_raw_db_url = os.environ.get("DATABASE_URL", "")
if _raw_db_url.startswith("postgres://"):
    _raw_db_url = _raw_db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    _raw_db_url
    if _raw_db_url
    else f"sqlite:///{os.path.join(DATA_DIR, 'evs.db')}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"]             = 16 * 1024 * 1024

app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_size": 5,
    "pool_recycle": 3600,
    "pool_pre_ping": True,
}

# ── Extensions ────────────────────────────────────────────────────────────────
db.init_app(app)


# ── Auto-setup: tables, migration, admin seed ────────────────────────────────
def _ensure_database():
    from sqlalchemy import inspect
    with app.app_context():
        inspector = inspect(db.engine)
        existing_tables = set(inspector.get_table_names())
        model_tables    = set(db.metadata.tables.keys())

        if not existing_tables:
            db.create_all()
            logger.info("Created all database tables.")
            existing_tables = model_tables
        elif missing := model_tables - existing_tables:
            db.create_all()
            logger.info("Created missing tables: %s", missing)
            existing_tables |= missing

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
try:
    _ensure_database()
    _db_ready = True
except Exception as exc:
    logger.warning("Database init deferred (will retry on first request): %s", exc)


@app.before_request
def _ensure_db_on_request():
    global _db_ready
    if not _db_ready:
        try:
            _ensure_database()
            _db_ready = True
        except Exception as exc:
            logger.error("Database still unavailable: %s", exc)


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


# ── Jinja2 globals ────────────────────────────────────────────────────────────
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
latest_result_store: dict = {"timestamp": None, "data": None}


# ── Raspberry Pi upload endpoint ──────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def process_pipeline():
    """
    Raspberry Pi upload endpoint (v6 — Pi does all processing).

    Pi already:
      1. Captured image.
      2. Ran YOLOv8n ONNX traffic detection.
      3. Ran EasyOCR plate recognition + crop.
      4. Uploads vehicle_image + plate_image + plate_number + traffic_level.

    This endpoint:
      1. Saves vehicle_image and plate_image to disk.
      2. Creates challan with Pi-provided plate_number.
      3. Returns result JSON.

    Multipart fields:
      vehicle_image — JPEG full camera frame
      plate_image   — JPEG cropped number plate (optional)
      plate_number  — Pi-recognized plate text (optional)
      traffic_level — "LOW" | "MEDIUM"
    """
    start = time.time()

    # ── Validate ──────────────────────────────────────────────────────────
    if "vehicle_image" not in request.files:
        return jsonify({"error": "Missing 'vehicle_image' file field."}), 400

    veh_file = request.files["vehicle_image"]
    if veh_file.filename == "":
        return jsonify({"error": "Empty vehicle_image uploaded."}), 400

    plate_file   = request.files.get("plate_image")
    plate_number = request.form.get("plate_number", "").strip().upper()
    plate_number = plate_number if plate_number else None
    traffic_level = request.form.get("traffic_level", "UNKNOWN").upper().strip()

    # ── Generate secure filenames ─────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_plate = (plate_number or "UNKNOWN").replace(" ", "_").replace("/", "_")
    veh_fname  = f"vehicle_{safe_plate}_{timestamp}.jpg"
    veh_path   = os.path.join(UPLOAD_DIR, veh_fname)
    veh_url    = f"/static/uploads/{veh_fname}"

    # ── Save vehicle image ────────────────────────────────────────────────
    try:
        veh_file.save(veh_path)
    except Exception as exc:
        logger.error("Failed to save vehicle_image: %s", exc)
        return jsonify({"error": "Failed to save vehicle image."}), 500

    # ── Save plate image (if provided) ────────────────────────────────────
    plate_url = None
    if plate_file and plate_file.filename:
        pl_fname = f"plate_{safe_plate}_{timestamp}.jpg"
        pl_path  = os.path.join(UPLOAD_DIR, pl_fname)
        plate_url = f"/static/uploads/{pl_fname}"
        try:
            plate_file.save(pl_path)
        except Exception as exc:
            logger.error("Failed to save plate_image: %s", exc)

    # ── Build result ──────────────────────────────────────────────────────
    if plate_number:
        action          = "Challan Generated"
        pipeline_status = f"Plate from Pi: {plate_number}"
    else:
        action          = "OCR Failed"
        pipeline_status = "No plate_number received from Pi."

    result = {
        "plate_number":    plate_number,
        "action":          action,
        "wide_image_url":  veh_url,
        "plate_image_url": plate_url,
        "status":          pipeline_status,
        "traffic_level":   traffic_level,
        "density":         0.0,
        "free_space":      0,
    }

    # ── Persist challan ───────────────────────────────────────────────────
    challan_number = None
    try:
        vehicle = Vehicle.query.filter_by(plate_number=plate_number).first() if plate_number else None
        challan_number = f"EVS{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:19]}"

        challan = Challan(
            challan_number  = challan_number,
            plate_number    = plate_number or "UNKNOWN",
            vehicle_id      = vehicle.id if vehicle else None,
            action          = action,
            traffic_level   = traffic_level,
            density_pct     = 0.0,
            free_space_px   = 0,
            pipeline_status = pipeline_status,
            wide_image_url  = veh_url,
            plate_image_url = plate_url,
            status          = "Pending",
            amount          = 500.0,
        )
        db.session.add(challan)
        db.session.commit()
    except Exception as exc:
        logger.exception("DB write error: %s", exc)
        db.session.rollback()

    # ── Update live feed ──────────────────────────────────────────────────
    latest_result_store["timestamp"] = time.time()
    latest_result_store["data"]      = result

    elapsed = time.time() - start
    logger.info("Upload processed in %.2fs — plate=%s", elapsed, plate_number)

    return jsonify({
        "status":          "processed",
        "action":          action,
        "plate_number":    plate_number,
        "challan_number":  challan_number,
        "traffic_level":   traffic_level,
        "wide_image_url":  veh_url,
        "plate_image_url": plate_url,
        "pipeline_status": pipeline_status,
    }), 200


# ── Live-feed polling endpoint ────────────────────────────────────────────────
@app.route("/api/latest", methods=["GET"])
def get_latest_result():
    return jsonify(latest_result_store)


# ── Health check ──────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health_check():
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


# ── Periodic cleanup ─────────────────────────────────────────────────────────
def _cleanup_old_uploads(max_age_hours: int = 24):
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
