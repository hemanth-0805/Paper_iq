from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st

try:
    from pymongo.errors import DuplicateKeyError, PyMongoError

    _PYMONGO_ERRORS_AVAILABLE = True
except Exception:
    DuplicateKeyError = Exception  # type: ignore
    PyMongoError = Exception  # type: ignore
    _PYMONGO_ERRORS_AVAILABLE = False

from crypto import hash_password
from db import get_mongo_db, _get_users_col, _next_sequence


def signup_user(username: str, password: str, role: str = "user") -> Tuple[bool, str]:
    username = (username or "").strip()
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."
    if role not in {"user", "admin"}:
        return False, "Invalid role."

    try:
        db = get_mongo_db()
        users_col = _get_users_col(db)
        user_id = _next_sequence(db, "users")
        users_col.insert_one(
            {
                "user_id": user_id,
                "username": username,
                "password_hash": hash_password(password, username=username),
                "role": role,
            }
        )
        return True, "Account created. Please log in."
    except DuplicateKeyError:
        return False, "Username already exists."
    except PyMongoError:
        return False, "Database error. Please try again."


def login_user(
    username: str,
    password: str,
    required_role: Optional[str] = None,
) -> Tuple[bool, str, Optional[dict]]:
    username = (username or "").strip()

    try:
        db = get_mongo_db()
        users_col = _get_users_col(db)
        row = users_col.find_one({"username": username})
        if not row:
            return False, "Invalid username or password.", None

        expected_hash = row["password_hash"]
        provided_hash = hash_password(password, username=username)
        if provided_hash != expected_hash:
            return False, "Invalid username or password.", None

        if required_role and row["role"] != required_role:
            return False, f"Access denied: requires {required_role} role.", None

        return True, "Login successful.", {"user_id": int(row["user_id"]), "username": row["username"], "role": row["role"]}
    except PyMongoError:
        return False, "Database error. Please try again.", None


def logout() -> None:
    st.session_state.pop("auth", None)
    st.session_state.pop("last_analysis", None)
    st.session_state.pop("last_analysis_meta", None)

