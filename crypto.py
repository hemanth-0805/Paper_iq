import hashlib
from typing import Optional


def _salt_for(username: str) -> str:
    # Deterministic salt to avoid separate salt storage in the schema.
    return hashlib.sha256(f"paperiq::{username.lower()}".encode("utf-8")).hexdigest()


def hash_password(password: str, username: Optional[str] = None) -> str:
    # If username provided, include per-user salt. Otherwise still hashed.
    salt = _salt_for(username) if username else "paperiq::static"
    return hashlib.sha256((salt + "::" + password).encode("utf-8")).hexdigest()

