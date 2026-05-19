"""
blueprints/user.py
Routes for authenticated vehicle owners.
"""

from functools import wraps
from flask import (
    Blueprint, render_template, redirect, url_for,
    flash, make_response
)
from flask_login import login_required, current_user

from database import db, Vehicle, Challan
from utils.pdf_receipt import generate_receipt_pdf

user_bp = Blueprint("user", __name__, url_prefix="/user")


# ── Access guard ──────────────────────────────────────────────────────────────

def vehicle_owner_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "vehicle_owner":
            flash("Access denied. This section is for registered vehicle owners only.", "danger")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


def _owner_challans(user):
    """Return all challans belonging to the current user's registered vehicles."""
    plate_numbers = [v.plate_number for v in user.vehicles]
    if not plate_numbers:
        return []
    return (
        Challan.query
        .filter(Challan.plate_number.in_(plate_numbers))
        .order_by(Challan.created_at.desc())
        .all()
    )


# ── Dashboard ─────────────────────────────────────────────────────────────────

@user_bp.route("/dashboard")
@login_required
@vehicle_owner_required
def dashboard():
    vehicles = current_user.vehicles.all()
    challans = _owner_challans(current_user)

    stats = {
        "total":   len(challans),
        "pending": sum(1 for c in challans if c.status == "Pending"),
        "paid":    sum(1 for c in challans if c.status == "Paid"),
        "amount_due": sum(c.amount for c in challans if c.status == "Pending"),
    }

    return render_template(
        "user/dashboard.html",
        vehicles=vehicles,
        challans=challans,
        stats=stats,
        active_nav="dashboard",
    )


# ── My Challans (full list) ───────────────────────────────────────────────────

@user_bp.route("/challans")
@login_required
@vehicle_owner_required
def my_challans():
    challans = _owner_challans(current_user)
    return render_template(
        "user/my_challans.html",
        challans=challans,
        active_nav="challans",
    )


# ── Challan Detail ────────────────────────────────────────────────────────────

@user_bp.route("/challan/<int:cid>")
@login_required
@vehicle_owner_required
def challan_detail(cid):
    challan = Challan.query.get_or_404(cid)

    # Authorisation: ensure this challan belongs to the current user
    user_plates = [v.plate_number for v in current_user.vehicles]
    if challan.plate_number not in user_plates:
        flash("You are not authorised to view this challan.", "danger")
        return redirect(url_for("user.dashboard"))

    vehicle = Vehicle.query.get(challan.vehicle_id) if challan.vehicle_id else None
    return render_template(
        "user/challan_detail.html",
        challan=challan,
        vehicle=vehicle,
        active_nav="challans",
    )


# ── Download PDF Receipt ──────────────────────────────────────────────────────

@user_bp.route("/challan/<int:cid>/receipt.pdf")
@login_required
@vehicle_owner_required
def download_receipt(cid):
    challan = Challan.query.get_or_404(cid)

    # Authorisation check
    user_plates = [v.plate_number for v in current_user.vehicles]
    if challan.plate_number not in user_plates:
        flash("You are not authorised to download this receipt.", "danger")
        return redirect(url_for("user.dashboard"))

    pdf_bytes = generate_receipt_pdf(challan)

    filename = f"Challan_{challan.challan_number}.pdf"

    response = make_response(pdf_bytes)
    response.headers["Content-Type"]        = "application/pdf"
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    response.headers["Content-Length"]      = len(pdf_bytes)
    return response
