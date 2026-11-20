import sqlite3
import json
from .config import DATABASE_PATH


def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                transcript TEXT NOT NULL,
                created_at TEXT NOT NULL,
                summary TEXT,
                action_items TEXT
            )
        """)
        # Migration: ensure summary and action_items exist in legacy databases
        cursor = conn.execute("PRAGMA table_info(notes)")
        existing_cols = [col[1] for col in cursor.fetchall()]
        if "summary" not in existing_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN summary TEXT")
        if "action_items" not in existing_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN action_items TEXT")


def save_note(note_id, title, transcript, created_at, summary="", action_items=None):
    if action_items is None:
        action_items = []
    action_items_str = json.dumps(action_items) if isinstance(action_items, list) else str(action_items)

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO notes
            (id, title, transcript, created_at, summary, action_items)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (note_id, title, transcript, created_at, summary, action_items_str))


def get_all_notes():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM notes
            ORDER BY created_at DESC
        """).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            try:
                d["action_items"] = json.loads(d.get("action_items") or "[]")
            except Exception:
                d["action_items"] = []
            results.append(d)
        return results


def delete_note(note_id):
    with get_connection() as conn:
        cursor = conn.execute("""
            DELETE FROM notes
            WHERE id = ?
        """, (note_id,))
        return cursor.rowcount > 0
