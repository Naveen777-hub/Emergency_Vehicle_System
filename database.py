"""
database.py
SQLAlchemy models for the Emergency Vehicle Priority System.
Tables: User, Vehicle, Challan
"""

import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """
    Represents a system user.
    Roles:
        - admin         : Full system access
        - vehicle_owner : Can view own challans and download receipts
    """
    __tablename__ = "users"

    id          = db.Column(db.Integer, primary_key=True)
    username    = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name   = db.Column(db.String(150), nullable=False)
    email       = db.Column(db.String(150), unique=True, nullable=True)
    phone       = db.Column(db.String(20), nullable=True)
    role        = db.Column(db.String(20), nullable=False, default="vehicle_owner")  # 'admin' | 'vehicle_owner'
    is_active   = db.Column(db.Boolean, default=True, nullable=False)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    vehicles = db.relationship("Vehicle", backref="owner", lazy="dynamic")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


class Vehicle(db.Model):
    """
    A registered vehicle linked to an owner (User).
    Challans may exist for a plate before the vehicle is registered.
    """
    __tablename__ = "vehicles"

    id            = db.Column(db.Integer, primary_key=True)
    plate_number  = db.Column(db.String(20), unique=True, nullable=False, index=True)
    vehicle_type  = db.Column(db.String(50), nullable=True)   # Car, Motorcycle, Bus, Truck
    owner_name    = db.Column(db.String(150), nullable=True)   # Display name of registered owner
    user_id       = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Challans issued against this vehicle
    challans = db.relationship("Challan", backref="vehicle_ref", lazy="dynamic",
                               foreign_keys="Challan.vehicle_id")

    def __repr__(self):
        return f"<Vehicle {self.plate_number}>"


class Challan(db.Model):
    """
    A traffic violation record produced by the AI pipeline.
    """
    __tablename__ = "challans"

    id               = db.Column(db.Integer, primary_key=True)
    challan_number   = db.Column(db.String(30), unique=True, nullable=False, index=True)
    plate_number     = db.Column(db.String(20), nullable=False, index=True)
    vehicle_id       = db.Column(db.Integer, db.ForeignKey("vehicles.id"), nullable=True)

    # AI Pipeline output
    action           = db.Column(db.String(50), nullable=False)       # Challan Generated / OCR Failed
    traffic_level    = db.Column(db.String(10), nullable=True)        # LOW | MEDIUM (from Raspberry Pi)
    density_pct      = db.Column(db.Float, default=0.0)              # Legacy — kept for DB compat
    free_space_px    = db.Column(db.Integer, default=0)              # Legacy — kept for DB compat
    pipeline_status  = db.Column(db.Text, nullable=True)              # Verbose decision reason
    wide_image_url   = db.Column(db.Text, nullable=True)
    plate_image_url  = db.Column(db.Text, nullable=True)

    # Enforcement fields
    status           = db.Column(db.String(20), default="Pending")    # Pending | Paid | Disputed
    amount           = db.Column(db.Float, default=500.0)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    paid_at          = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return f"<Challan {self.challan_number} [{self.status}]>"
