"""
seed_users.py
Creates vehicle owner accounts for HariKumar and Naveen Kumar,
then links them to their registered vehicles.
Usage: python seed_users.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from database import db, User, Vehicle

def seed():
    with app.app_context():
        users_to_create = [
            {
                "username":   "harikumar",
                "password":   "hari1234",
                "full_name":  "HariKumar",
                "email":      "harikumar@evps.gov.in",
                "phone":      "9000000001",
                "plate":      "KL31R2002",
            },
            {
                "username":   "naveenkumar",
                "password":   "naveen1234",
                "full_name":  "Naveen Kumar",
                "email":      "naveenkumar@evps.gov.in",
                "phone":      "9000000002",
                "plate":      "KL31S0690",
            },
        ]

        for u in users_to_create:
            # Create user if not exists
            existing = User.query.filter_by(username=u["username"]).first()
            if not existing:
                user = User(
                    username=u["username"],
                    full_name=u["full_name"],
                    email=u["email"],
                    phone=u["phone"],
                    role="vehicle_owner",
                    is_active=True,
                )
                user.set_password(u["password"])
                db.session.add(user)
                db.session.flush()  # get the new user.id
                print(f"[OK] Created user: {u['username']}  |  password: {u['password']}")
            else:
                user = existing
                print(f"[--] User already exists: {u['username']}")

            # Link vehicle to this user
            vehicle = Vehicle.query.filter_by(plate_number=u["plate"]).first()
            if vehicle:
                vehicle.user_id = user.id
                print(f"[OK] Linked plate {u['plate']} → {u['username']}")
            else:
                print(f"[!!] Vehicle {u['plate']} not found in registry — skipping link")

        db.session.commit()
        print("\n── Summary ──────────────────────────────────")
        for u in users_to_create:
            print(f"  Username : {u['username']:<20} Password : {u['password']}")
        print("─────────────────────────────────────────────")
        print("Done! Both accounts can now log in and view their challans.")

if __name__ == "__main__":
    seed()
