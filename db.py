from __future__ import annotations

from typing import Optional

import streamlit as st

try:
    from pymongo import MongoClient, ReturnDocument
    from pymongo.errors import PyMongoError, DuplicateKeyError

    PYMONGO_AVAILABLE = True
except Exception:
    MongoClient = None  # type: ignore
    ReturnDocument = None  # type: ignore
    PyMongoError = Exception  # type: ignore
    DuplicateKeyError = Exception  # type: ignore
    PYMONGO_AVAILABLE = False

from config import (
    MONGODB_URI,
    MONGODB_DB_NAME,
    MONGODB_USERS_COLLECTION,
    MONGODB_ANALYSIS_COLLECTION,
    MONGODB_COUNTERS_COLLECTION,
)
from crypto import hash_password


@st.cache_resource(show_spinner=False)
def get_mongo_db():
    """
    Returns a pymongo database handle.

    Cached via Streamlit to avoid opening a new connection on every rerun.
    """
    if not PYMONGO_AVAILABLE or MongoClient is None:
        raise RuntimeError("pymongo is not installed. Run `pip install pymongo` and restart the app.")

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Validate connection early.
    client[MONGODB_DB_NAME].command("ping")
    return client[MONGODB_DB_NAME]


def _get_users_col(db):
    return db[MONGODB_USERS_COLLECTION]

def _get_analysis_col(db):
    return db[MONGODB_ANALYSIS_COLLECTION]

def _get_counters_col(db):
    return db[MONGODB_COUNTERS_COLLECTION]

def _next_sequence(db, key: str) -> int:
    """Simple atomic counter (so we can keep numeric ids like the sqlite version)."""
    counters = _get_counters_col(db)
    doc = counters.find_one_and_update(
        {"_id": key},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return int(doc["seq"])

def create_database() -> None:
    """Ensure MongoDB collections + indexes exist and create the default admin."""
    db = get_mongo_db()
    users_col = _get_users_col(db)
    analysis_col = _get_analysis_col(db)

    # Indexes
    users_col.create_index("username", unique=True)
    users_col.create_index("user_id", unique=True)
    analysis_col.create_index("analysis_id", unique=True)
    analysis_col.create_index("user_id")

    # Ensure a default admin exists (admin / admin123).
    username = "admin"
    default_password = "admin123"
    correct_hash = hash_password(default_password, username=username)

    existing_admin = users_col.find_one({"username": username}, {"_id": 0, "user_id": 1})
    if existing_admin:
        users_col.update_one({"username": username}, {"$set": {"password_hash": correct_hash, "role": "admin"}})
    else:
        user_id = _next_sequence(db, "users")
        users_col.update_one(
            {"username": username},
            {"$set": {"user_id": user_id, "password_hash": correct_hash, "role": "admin"}},
            upsert=True,
        )

