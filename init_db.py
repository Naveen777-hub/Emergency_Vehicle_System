"""
init_db.py
Run this script ONCE to create all database tables and seed the default admin user.
Usage:  python init_db.py
"""

import os
import sys

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from database import db, User, Vehicle, Challan


def init_database():
    with app.app_context():
        # Create data directory if it does not exist
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)

        # Create all tables
        db.create_all()
        print("[OK] Database tables created.")

        # Seed default admin if not already present
        if not User.query.filter_by(username="admin").first():
            admin = User(
                username="admin",
                full_name="System Administrator",
                email="admin@evps.gov.in",
                phone="1800-000-0000",
                role="admin",
                is_active=True,
            )
            admin.set_password("admin123")
            db.session.add(admin)
            db.session.commit()
            print("[OK] Default admin user created  →  username: admin  |  password: admin123")
        else:
            print("[--] Admin user already exists. Skipping seed.")

        total_users    = User.query.count()
        total_vehicles = Vehicle.query.count()
        total_challans = Challan.query.count()
        print(f"\nDatabase Summary:")
        print(f"  Users    : {total_users}")
        print(f"  Vehicles : {total_vehicles}")
        print(f"  Challans : {total_challans}")
        print("\nDatabase initialisation complete.")


if __name__ == "__main__":
    init_database()
