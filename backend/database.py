
import sqlite3

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
                created_at TEXT NOT NULL
            )
        """)


def save_note(note_id, title, transcript, created_at):

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO notes
            (id, title, transcript, created_at)
            VALUES (?, ?, ?, ?)
        """, (note_id, title, transcript, created_at))


def get_all_notes():

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT * FROM notes
            ORDER BY created_at DESC
        """).fetchall()

        return [dict(row) for row in rows]


def delete_note(note_id):

    with get_connection() as conn:

        cursor = conn.execute("""
            DELETE FROM notes
            WHERE id = ?
        """, (note_id,))

        return cursor.rowcount > 0
