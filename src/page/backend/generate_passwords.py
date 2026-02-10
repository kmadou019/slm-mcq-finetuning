#!/usr/bin/env python3
"""
Script to generate password hashes for users
"""
from api.utils.security import get_password_hash

# Generate hashes for our test users
admin_hash = get_password_hash("admin123")
evaluator_hash = get_password_hash("eval123")

print("Password hashes generated:")
print(f"\nAdmin (admin123):")
print(f'"{admin_hash}"')
print(f"\nEvaluator (eval123):")
print(f'"{evaluator_hash}"')
