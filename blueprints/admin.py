"""
blueprints/admin.py
All admin-only routes for the Emergency Vehicle Priority System.
"""

from datetime import datetime, date, timedelta
from functools import wraps

from flask import (
    Blueprint, render_template, redirect, url_for,
    request, flash, abort
)
from flask_login import login_required, current_user

from database import db, User, Vehicle, Challan

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


# ── Access guard ──────────────────────────────────────────────────────────────

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            flash("Access denied. This section is restricted to administrators.", "danger")
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated


# ── Dashboard ─────────────────────────────────────────────────────────────────

@admin_bp.route("/")
@login_required
@admin_required
def dashboard():
    today = date.today()
    month_start = today.replace(day=1)

    stats = {
        "today_challans":   Challan.query.filter(
                                db.func.date(Challan.created_at) == today
                            ).count(),
        "month_challans":   Challan.query.filter(
                                Challan.created_at >= datetime.combine(month_start, datetime.min.time())
                            ).count(),
        "pending_challans": Challan.query.filter_by(status="Pending").count(),
        "total_challans":   Challan.query.count(),
        "total_vehicles":   Vehicle.query.count(),
        "total_users":      User.query.filter_by(role="vehicle_owner").count(),
    }

    recent_challans = (
        Challan.query
        .order_by(Challan.created_at.desc())
        .limit(8)
        .all()
    )

    return render_template(
        "admin/dashboard.html",
        stats=stats,
        recent_challans=recent_challans,
        active_nav="dashboard",
    )


# ── Live Monitor ──────────────────────────────────────────────────────────────

@admin_bp.route("/live")
@login_required
@admin_required
def live_monitor():
    return render_template("admin/live_monitor.html", active_nav="live")


# ── Challan Management ────────────────────────────────────────────────────────

@admin_bp.route("/challans")
@login_required
@admin_required
def challans():
    page         = request.args.get("page", 1, type=int)
    status_f     = request.args.get("status", "").strip()
    plate_f      = request.args.get("plate", "").strip()
    date_from_f  = request.args.get("date_from", "").strip()
    date_to_f    = request.args.get("date_to", "").strip()

    query = Challan.query

    if status_f:
        query = query.filter(Challan.status == status_f)
    if plate_f:
        query = query.filter(Challan.plate_number.ilike(f"%{plate_f}%"))
    if date_from_f:
        try:
            dt = datetime.strptime(date_from_f, "%Y-%m-%d")
            query = query.filter(Challan.created_at >= dt)
        except ValueError:
            pass
    if date_to_f:
        try:
            dt = datetime.strptime(date_to_f, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(Challan.created_at < dt)
        except ValueError:
            pass

    pagination = (
        query.order_by(Challan.created_at.desc())
             .paginate(page=page, per_page=15, error_out=False)
    )

    return render_template(
        "admin/challans.html",
        challans=pagination.items,
        pagination=pagination,
        status_f=status_f,
        plate_f=plate_f,
        date_from_f=date_from_f,
        date_to_f=date_to_f,
        active_nav="challans",
    )


@admin_bp.route("/challan/<int:cid>")
@login_required
@admin_required
def challan_detail(cid):
    challan = Challan.query.get_or_404(cid)
    vehicle = Vehicle.query.get(challan.vehicle_id) if challan.vehicle_id else None
    registered_owner = None
    if vehicle and vehicle.user_id:
        registered_owner = User.query.get(vehicle.user_id)

    return render_template(
        "admin/challan_detail.html",
        challan=challan,
        vehicle=vehicle,
        registered_owner=registered_owner,
        active_nav="challans",
    )


@admin_bp.route("/challan/<int:cid>/update-status", methods=["POST"])
@login_required
@admin_required
def update_challan_status(cid):
    challan = Challan.query.get_or_404(cid)
    new_status = request.form.get("status", "").strip()

    allowed = {"Pending", "Paid", "Disputed"}
    if new_status not in allowed:
        flash("Invalid status value.", "danger")
        return redirect(url_for("admin.challan_detail", cid=cid))

    challan.status = new_status
    if new_status == "Paid":
        challan.paid_at = datetime.utcnow()
    else:
        challan.paid_at = None

    db.session.commit()
    flash(f"Challan status updated to '{new_status}'.", "success")
    return redirect(url_for("admin.challan_detail", cid=cid))


# ── User Management ───────────────────────────────────────────────────────────

@admin_bp.route("/users")
@login_required
@admin_required
def users():
    search = request.args.get("search", "").strip()
    query = User.query

    if search:
        like = f"%{search}%"
        query = query.filter(
            db.or_(
                User.username.ilike(like),
                User.full_name.ilike(like),
                User.email.ilike(like),
            )
        )

    all_users = query.order_by(User.created_at.desc()).all()
    return render_template(
        "admin/users.html",
        users=all_users,
        search=search,
        active_nav="users",
    )


@admin_bp.route("/users/add", methods=["GET", "POST"])
@login_required
@admin_required
def add_user():
    if request.method == "POST":
        full_name        = request.form.get("full_name", "").strip()
        username         = request.form.get("username", "").strip()
        email            = request.form.get("email", "").strip() or None
        phone            = request.form.get("phone", "").strip() or None
        role             = request.form.get("role", "vehicle_owner")
        password         = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = []
        if not full_name:
            errors.append("Full name is required.")
        if not username:
            errors.append("Username is required.")
        elif User.query.filter_by(username=username).first():
            errors.append(f'Username "{username}" is already taken.')
        if email and User.query.filter_by(email=email).first():
            errors.append(f'Email "{email}" is already registered.')
        if not password:
            errors.append("Password is required.")
        elif len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm_password:
            errors.append("Passwords do not match.")
        if role not in ("admin", "vehicle_owner"):
            errors.append("Invalid role selected.")

        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template("admin/add_user.html", active_nav="users",
                                   form_data=request.form)

        user = User(
            full_name=full_name,
            username=username,
            email=email,
            phone=phone,
            role=role,
            is_active=True,
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash(f'User account for "{full_name}" created successfully.', "success")
        return redirect(url_for("admin.users"))

    return render_template("admin/add_user.html", active_nav="users", form_data={})


@admin_bp.route("/users/<int:uid>/toggle", methods=["POST"])
@login_required
@admin_required
def toggle_user(uid):
    user = User.query.get_or_404(uid)
    if user.id == current_user.id:
        flash("You cannot deactivate your own account.", "warning")
        return redirect(url_for("admin.users"))

    user.is_active = not user.is_active
    db.session.commit()
    state = "activated" if user.is_active else "deactivated"
    flash(f'Account for "{user.username}" has been {state}.', "info")
    return redirect(url_for("admin.users"))


@admin_bp.route("/users/<int:uid>/edit", methods=["GET", "POST"])
@login_required
@admin_required
def edit_user(uid):
    user = User.query.get_or_404(uid)

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email     = request.form.get("email", "").strip() or None
        phone     = request.form.get("phone", "").strip() or None
        password  = request.form.get("password", "")

        if not full_name:
            flash("Full name is required.", "danger")
            return render_template("admin/edit_user.html", user=user, active_nav="users")

        if email and email != user.email:
            existing = User.query.filter_by(email=email).first()
            if existing and existing.id != uid:
                flash(f'Email "{email}" is already in use.', "danger")
                return render_template("admin/edit_user.html", user=user, active_nav="users")

        user.full_name = full_name
        user.email     = email
        user.phone     = phone

        if password:
            if len(password) < 6:
                flash("Password must be at least 6 characters.", "danger")
                return render_template("admin/edit_user.html", user=user, active_nav="users")
            user.set_password(password)

        db.session.commit()
        flash(f'User "{user.username}" updated successfully.', "success")
        return redirect(url_for("admin.users"))

    return render_template("admin/edit_user.html", user=user, active_nav="users")


# ── Vehicle Registry ──────────────────────────────────────────────────────────

@admin_bp.route("/vehicles")
@login_required
@admin_required
def vehicles():
    search = request.args.get("search", "").strip()
    query  = Vehicle.query

    if search:
        like = f"%{search}%"
        query = query.filter(
            db.or_(
                Vehicle.plate_number.ilike(like),
                Vehicle.owner_name.ilike(like),
            )
        )

    all_vehicles = query.order_by(Vehicle.registered_at.desc()).all()
    vehicle_owners = User.query.filter_by(role="vehicle_owner", is_active=True).all()

    return render_template(
        "admin/vehicles.html",
        vehicles=all_vehicles,
        vehicle_owners=vehicle_owners,
        search=search,
        active_nav="vehicles",
    )


@admin_bp.route("/vehicles/add", methods=["POST"])
@login_required
@admin_required
def add_vehicle():
    plate_number = request.form.get("plate_number", "").strip().upper()
    vehicle_type = request.form.get("vehicle_type", "").strip()
    owner_name   = request.form.get("owner_name", "").strip()
    user_id_str  = request.form.get("user_id", "").strip()

    if not plate_number:
        flash("Registration plate number is required.", "danger")
        return redirect(url_for("admin.vehicles"))

    if Vehicle.query.filter_by(plate_number=plate_number).first():
        flash(f'Vehicle with plate "{plate_number}" is already registered.', "danger")
        return redirect(url_for("admin.vehicles"))

    user_id = int(user_id_str) if user_id_str.isdigit() else None

    vehicle = Vehicle(
        plate_number=plate_number,
        vehicle_type=vehicle_type or None,
        owner_name=owner_name or None,
        user_id=user_id,
    )
    db.session.add(vehicle)

    # Link any existing challans with this plate to the vehicle
    db.session.flush()
    Challan.query.filter_by(plate_number=plate_number, vehicle_id=None).update(
        {"vehicle_id": vehicle.id}
    )

    db.session.commit()
    flash(f'Vehicle "{plate_number}" registered successfully.', "success")
    return redirect(url_for("admin.vehicles"))


@admin_bp.route("/vehicles/<int:vid>/link", methods=["POST"])
@login_required
@admin_required
def link_vehicle(vid):
    vehicle     = Vehicle.query.get_or_404(vid)
    user_id_str = request.form.get("user_id", "").strip()

    if not user_id_str or not user_id_str.isdigit():
        vehicle.user_id = None
        db.session.commit()
        flash(f'Vehicle "{vehicle.plate_number}" unlinked from user.', "info")
        return redirect(url_for("admin.vehicles"))

    user = User.query.get(int(user_id_str))
    if not user or user.role != "vehicle_owner":
        flash("Invalid user selected.", "danger")
        return redirect(url_for("admin.vehicles"))

    vehicle.user_id = user.id
    db.session.commit()
    flash(f'Vehicle "{vehicle.plate_number}" linked to user "{user.username}".', "success")
    return redirect(url_for("admin.vehicles"))
