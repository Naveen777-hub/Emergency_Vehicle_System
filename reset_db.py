"""
reset_db.py
Wipes all challans, vehicles, and non-admin users from the database.
Works with both SQLite (local dev) and PostgreSQL (Render).
Usage: python reset_db.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from database import db, User, Vehicle, Challan


def reset():
    with app.app_context():
        before = {
            "challans": Challan.query.count(),
            "vehicles": Vehicle.query.count(),
            "users":    User.query.filter(User.role != "admin").count(),
        }

        print("Before reset:")
        for k, v in before.items():
            print(f"  {k.capitalize()}: {v}")

        Challan.query.delete()
        Vehicle.query.delete()
        User.query.filter(User.role != "admin").delete()
        db.session.commit()

        print()
        print("[OK] All challans deleted.")
        print("[OK] All vehicles deleted.")
        print("[OK] All non-admin users deleted.")
        print()
        print("Database is now brand new. Admin login untouched.")


if __name__ == "__main__":
    reset()
