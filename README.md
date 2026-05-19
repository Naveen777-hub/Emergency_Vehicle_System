# Intelligent Noise-Free Emergency Vehicle Priority Framework

**Capstone Project — CAP769 | MCA Semester 4**

An edge-cloud AI system that automates right-of-way enforcement for emergency vehicles using computer vision. A Raspberry Pi captures camera feeds, runs local YOLOv8n ONNX traffic detection, and uploads images to a Flask cloud server. The cloud backend runs EasyOCR for number plate recognition and generates automated challans.

---

## System Architecture

```
[Raspberry Pi]                         [Cloud Server — Flask/Render]
  ├── Camera Feed ──── HTTP POST ────► ├── EasyOCR (Number Plate OCR)
  │    (LOW/MEDIUM traffic only)       ├── PostgreSQL / SQLite Database
  │    Pi runs YOLOv8n ONNX locally    └── Role-Based Web Portal
  └── HIGH traffic → skip upload
```

**Edge Node (Raspberry Pi):** Runs YOLOv8n ONNX locally for traffic density classification. Uploads images only when traffic is LOW or MEDIUM to save bandwidth and cloud compute.

**Cloud Node (Flask):** Runs EasyOCR directly on uploaded images (lazy-loaded, no YOLO on cloud). Persists enforcement records to database. Serves a government-styled web portal for administrators and vehicle owners. Optimized for Render free tier (512MB RAM).

---

## Project Structure

```
Emergency_Vehicle_System/
│
├── app.py                        # Flask application — routes, error handlers, config
├── pipeline.py                   # OCR pipeline — EasyOCR (lazy-loaded, no YOLO)
├── auth.py                       # Authentication blueprint — login / logout / register
├── database.py                   # SQLAlchemy models — User, Vehicle, Challan
├── init_db.py                    # One-time database initialisation and admin seed
├── reset_db.py                   # Wipes data (keeps admin) — works with SQLite & PostgreSQL
│
├── blueprints/
│   ├── admin.py                  # Admin routes — dashboard, challans, users, vehicles
│   └── user.py                   # Vehicle owner routes — challan view, PDF receipt
│
├── utils/
│   └── pdf_receipt.py            # ReportLab PDF e-challan receipt generator
│
├── pi_client.py                  # Raspberry Pi edge client (runs on Pi, not on cloud)
│
├── static/
│   ├── css/gov.css               # Indian Government portal design system
│   └── uploads/                  # Uploaded images (auto-cleaned after 24h)
│
├── templates/
│   ├── base.html                 # Master layout — govt header, nav, footer
│   ├── errors/                   # Error pages (404, 500)
│   ├── auth/                     # Login, register templates
│   ├── admin/                    # Admin templates
│   └── user/                     # Vehicle owner templates
│
├── data/
│   └── evs.db                    # SQLite database (auto-created on init)
│
├── requirements.txt              # Pinned dependencies — Render free tier optimized
├── render.yaml                   # Render deployment config
└── Procfile                      # Gunicorn start command
```

---

## AI Pipeline (`pipeline.py`)

A single `OCRPipeline` class handles all cloud-side AI:

1. **Image Preprocessing** — Grayscale, Gaussian blur, adaptive threshold
2. **EasyOCR** — Lazy-loaded on first request (not at startup). Runs directly on the full uploaded image.
3. **Challan Decision** — If OCR extracts a plate number > 3 chars, a challan is generated.

No YOLO, no ultralytics, no torch on the cloud backend. All traffic detection runs on the Raspberry Pi edge node.

---

## Database Schema (`database.py`)

| Table      | Key Columns |
|------------|-------------|
| `users`    | id, username, password_hash, full_name, email, phone, role, is_active, created_at |
| `vehicles` | id, plate_number, vehicle_type, owner_name, user_id (FK → users) |
| `challans` | id, challan_number, plate_number, vehicle_id (FK), traffic_level, action, density_pct, free_space_px, pipeline_status, wide_image_url, plate_image_url, status, amount, created_at, paid_at |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | **Raspberry Pi endpoint** — accepts single image + traffic_level, runs OCR, writes challan |
| `GET`  | `/api/latest` | Latest IoT telemetry as JSON (polled by live monitor) |
| `GET`  | `/api/health` | Health check for Render uptime monitors |
| `GET`  | `/` | Redirect to role dashboard or login |

All admin and user routes are under `/admin/` and `/user/` prefixes respectively.

---

## Setup and Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialise the Database

```bash
python init_db.py
```

Default admin credentials:
```
Username : admin
Password : admin123
```

### 3. Start the Server

```bash
python app.py
```

Server starts on `http://0.0.0.0:5000`.

### 4. Open the Portal

Navigate to `http://localhost:5000` — you will be redirected to the login page.

### 5. Simulate an Edge Device Upload

```python
import requests
files = {'image': open('test_vehicle.jpg', 'rb')}
data  = {'traffic_level': 'LOW'}
requests.post('http://localhost:5000/api/upload', files=files, data=data)
```

---

## Deploy to Render

1. Push this repo to GitHub.
2. In Render dashboard: **New +** → **Blueprint** → connect your repo.
3. `render.yaml` is auto-detected. Render creates the web service + PostgreSQL database.
4. No manual config needed. The free tier (512MB RAM) is sufficient.

> ⚠️ EasyOCR loads on first `/api/upload` request (~300MB transient). Startup RAM is ~50MB. If you see OOM kills, add a warmup request during the health check grace period.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Web Framework | Flask 3.x |
| Authentication | Flask-Login |
| Database ORM | Flask-SQLAlchemy (SQLite / PostgreSQL) |
| OCR | EasyOCR (lazy-loaded, CPU mode) |
| Image Processing | OpenCV, NumPy |
| PDF Generation | ReportLab |
| Edge Detection | YOLOv8n ONNX (on Raspberry Pi only) |
| Frontend | Vanilla HTML / CSS (Indian Government design system) |
