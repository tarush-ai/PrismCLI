"""SQLite database layer for Prism."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from prism.models import Experiment, Investigation, Paper

PRISM_DIR = Path.home() / ".prism"
DB_PATH = PRISM_DIR / "prism.db"


def _get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    PRISM_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize the database and create tables if they don't exist."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                topic TEXT NOT NULL,
                phase TEXT NOT NULL DEFAULT 'observation',
                user_context TEXT DEFAULT '',
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                code TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                result TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (investigation_id) REFERENCES investigations(id)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def create_investigation(inv: Investigation) -> None:
    """Save a new investigation to the database."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO investigations (id, title, topic, phase, user_context, data, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inv.id,
                inv.title,
                inv.topic,
                inv.phase,
                inv.user_context,
                inv.model_dump_json(),
                inv.created_at,
                inv.updated_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_investigation(id: str) -> Investigation | None:
    """Retrieve an investigation by ID."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM investigations WHERE id = ?", (id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return Investigation.model_validate_json(row["data"])
    finally:
        conn.close()


def list_investigations() -> list[dict[str, Any]]:
    """List all investigations with summary info."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, title, topic, phase, created_at
            FROM investigations
            ORDER BY updated_at DESC
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def update_investigation(inv: Investigation) -> None:
    """Update an existing investigation in the database."""
    inv.touch()
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE investigations
            SET title = ?, topic = ?, phase = ?, user_context = ?, data = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                inv.title,
                inv.topic,
                inv.phase,
                inv.user_context,
                inv.model_dump_json(),
                inv.updated_at,
                inv.id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def save_experiment(exp: Experiment) -> None:
    """Save a new experiment to the database."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO experiments
            (id, investigation_id, title, description, code, status, result, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exp.id,
                exp.investigation_id,
                exp.title,
                exp.description,
                exp.code,
                exp.status,
                exp.result,
                exp.created_at,
                exp.updated_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_experiments(investigation_id: str) -> list[Experiment]:
    """Get all experiments for an investigation."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM experiments WHERE investigation_id = ?
            ORDER BY created_at ASC
            """,
            (investigation_id,),
        )
        rows = cursor.fetchall()
        return [
            Experiment(
                id=row["id"],
                investigation_id=row["investigation_id"],
                title=row["title"],
                description=row["description"],
                code=row["code"] or "",
                status=row["status"],
                result=row["result"] or "",
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]
    finally:
        conn.close()


def update_experiment(exp: Experiment) -> None:
    """Update an existing experiment."""
    exp.updated_at = datetime.utcnow().isoformat()
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE experiments
            SET title = ?, description = ?, code = ?, status = ?, result = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                exp.title,
                exp.description,
                exp.code,
                exp.status,
                exp.result,
                exp.updated_at,
                exp.id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def delete_investigation(id: str) -> bool:
    """Delete an investigation and its experiments."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE investigation_id = ?", (id,))
        cursor.execute("DELETE FROM investigations WHERE id = ?", (id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()
