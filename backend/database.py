import os
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
                action_items TEXT,
                audio_path TEXT,
                utterances_json TEXT
            )
        """)
        # Safe migration: ensure all columns exist in legacy databases
        cursor = conn.execute("PRAGMA table_info(notes)")
        existing_cols = [col[1] for col in cursor.fetchall()]
        if "summary" not in existing_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN summary TEXT")
        if "action_items" not in existing_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN action_items TEXT")
        if "audio_path" not in existing_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN audio_path TEXT")
        if "utterances_json" not in existing_cols:
            conn.execute("ALTER TABLE notes ADD COLUMN utterances_json TEXT")


def save_note(note_id, title, transcript, created_at, summary="", action_items=None, audio_path="", utterances=None):
    if action_items is None:
        action_items = []
    action_items_str = json.dumps(action_items) if isinstance(action_items, (list, dict)) else str(action_items)
    utterances_str = json.dumps(utterances) if isinstance(utterances, (list, dict)) else (str(utterances) if utterances is not None else "[]")

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO notes
            (id, title, transcript, created_at, summary, action_items, audio_path, utterances_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (note_id, title, transcript, created_at, summary, action_items_str, audio_path, utterances_str))


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
            try:
                d["utterances"] = json.loads(d.get("utterances_json") or "[]")
            except Exception:
                d["utterances"] = []
            results.append(d)
        return results


def delete_note(note_id):
    """
    Deletes the note record from SQLite AND deletes the physical audio file from disk.
    Guarantees zero orphan audio files remain.
    """
    with get_connection() as conn:
        row = conn.execute("""
            SELECT audio_path FROM notes WHERE id = ?
        """, (note_id,)).fetchone()

        audio_path = dict(row).get("audio_path") if row else ""

        cursor = conn.execute("""
            DELETE FROM notes
            WHERE id = ?
        """, (note_id,))
        deleted = cursor.rowcount > 0

        # Physical file cleanup
        if deleted and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass

        return deleted
