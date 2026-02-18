#!/usr/bin/env python3
"""
Script to initialize users.json with an admin account.
Run this once before first deployment:
    cd backend && python generate_passwords.py
"""
import json
import getpass
from pathlib import Path
from api.utils.security import hash_password

USERS_DB_PATH = Path(__file__).parent / "data" / "users.json"


def main():
    if USERS_DB_PATH.exists():
        overwrite = input(f"users.json already exists at {USERS_DB_PATH}. Overwrite? (y/N): ")
        if overwrite.lower() != 'y':
            print("Aborted.")
            return

    print("--- Create admin account ---")
    admin_password = getpass.getpass("Enter admin password: ")
    confirm = getpass.getpass("Confirm admin password: ")

    if admin_password != confirm:
        print("Passwords do not match. Aborted.")
        return

    if len(admin_password) < 8:
        print("Password must be at least 8 characters. Aborted.")
        return

    users = [
        {
            "id": "user-001",
            "username": "admin",
            "email": "admin@mcq-eval.com",
            "role": "admin",
            "created_at": "2024-01-01T00:00:00Z",
            "hashed_password": hash_password(admin_password)
        }
    ]

    USERS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_DB_PATH, 'w') as f:
        json.dump(users, f, indent=2)

    print(f"users.json created at {USERS_DB_PATH}")


if __name__ == "__main__":
    main()
