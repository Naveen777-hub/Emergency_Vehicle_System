"""
auth.py
Authentication blueprint: login, logout, and self-registration for vehicle owners.
"""

from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user
from database import db, User, Vehicle

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return _role_redirect(current_user)

    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            error = "Invalid username or password. Please try again."
        elif not user.is_active:
            error = "Your account has been deactivated. Contact the administrator."
        else:
            login_user(user, remember=False)
            return _role_redirect(user)

    return render_template("auth/login.html", error=error)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("auth.login"))


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Self-registration for vehicle owners."""
    if current_user.is_authenticated:
        return _role_redirect(current_user)

    error = None

    if request.method == "POST":
        full_name    = request.form.get("full_name", "").strip()
        username     = request.form.get("username", "").strip().lower()
        email        = request.form.get("email", "").strip() or None
        phone        = request.form.get("phone", "").strip() or None
        plate_number = request.form.get("plate_number", "").strip().upper() or None
        password     = request.form.get("password", "")
        confirm_pw   = request.form.get("confirm_password", "")

        # ── Validations ──────────────────────────────────────────────────────
        if not full_name or not username or not password:
            error = "Full name, username, and password are required."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm_pw:
            error = "Passwords do not match."
        elif User.query.filter_by(username=username).first():
            error = f"Username '{username}' is already taken. Please choose another."
        elif email and User.query.filter_by(email=email).first():
            error = "An account with this email already exists."
        else:
            # ── Create user ──────────────────────────────────────────────────
            user = User(
                username=username,
                full_name=full_name,
                email=email,
                phone=phone,
                role="vehicle_owner",
                is_active=True,
            )
            user.set_password(password)
            db.session.add(user)
            db.session.flush()  # get user.id before commit

            # ── Auto-link plate if provided and found ────────────────────────
            linked_plate = None
            if plate_number:
                vehicle = Vehicle.query.filter_by(plate_number=plate_number).first()
                if vehicle:
                    if vehicle.user_id is None:
                        vehicle.user_id = user.id
                        linked_plate = plate_number
                    else:
                        flash(
                            f"Plate {plate_number} is already linked to another account. "
                            "Contact the administrator to resolve this.",
                            "warning"
                        )
                else:
                    flash(
                        f"Plate '{plate_number}' was not found in the vehicle registry. "
                        "Your account was created — ask the admin to link your plate.",
                        "warning"
                    )

            db.session.commit()

            msg = "Account created successfully! You can now log in."
            if linked_plate:
                msg += f" Your vehicle ({linked_plate}) has been linked to your account."
            flash(msg, "success")
            return redirect(url_for("auth.login"))

    return render_template("auth/register.html", error=error)


def _role_redirect(user):
    """Redirect a user to their role-appropriate dashboard."""
    if user.role == "admin":
        return redirect(url_for("admin.dashboard"))
    return redirect(url_for("user.dashboard"))
