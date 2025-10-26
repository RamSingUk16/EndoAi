import sqlite3
from typing import Optional, Iterable, Dict
from .config import settings


def get_conn(path: Optional[str] = None) -> sqlite3.Connection:
    db_path = path or settings.DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash BLOB NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            slide_id TEXT UNIQUE NOT NULL,
            patient_id TEXT NOT NULL,
            age INTEGER,
            clinical_history TEXT,
            image_data BLOB NOT NULL,
            filename TEXT,
            model TEXT DEFAULT 'program3',
            gradcam_requested TEXT DEFAULT 'auto',
            gradcam_data BLOB,
            status TEXT DEFAULT 'pending',
            prediction TEXT,
            confidence REAL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY,
            case_id INTEGER NOT NULL,
            artifact_type TEXT,
            filename TEXT,
            blob BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS shares (
            id INTEGER PRIMARY KEY,
            case_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY,
            case_id INTEGER NOT NULL,
            author_id INTEGER NOT NULL,
            body TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            edited_at TIMESTAMP,
            FOREIGN KEY(case_id) REFERENCES cases(id) ON DELETE CASCADE,
            FOREIGN KEY(author_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )

    conn.commit()


def _get_table_columns(conn: sqlite3.Connection, table: str) -> Dict[str, dict]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = {}
    for cid, name, ctype, notnull, dflt, pk in cur.fetchall():
        cols[name] = {
            'cid': cid,
            'type': ctype,
            'notnull': notnull,
            'default': dflt,
            'pk': pk,
        }
    return cols


def _ensure_column(conn: sqlite3.Connection, table: str, column_def: str, column_name: Optional[str] = None):
    name = column_name or column_def.split()[0]
    cols = _get_table_columns(conn, table)
    if name not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
        conn.commit()


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Apply lightweight, idempotent migrations for older DBs.

    - Add missing columns to existing tables without destructive changes.
    - Seed user_id for existing cases to the admin user if missing.
    """
    create_tables(conn)
    # Ensure users table has an admin user (id used to backfill user_id)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
    row = cur.fetchone()
    admin_id = None
    if row:
        admin_id = row[0]

    # Migrate cases table columns (older DBs may lack user_id and newer fields)
    try:
        _ensure_column(conn, 'cases', 'user_id INTEGER')
        _ensure_column(conn, 'cases', 'slide_id TEXT')
        _ensure_column(conn, 'cases', 'patient_id TEXT')
        _ensure_column(conn, 'cases', 'age INTEGER')
        _ensure_column(conn, 'cases', 'filename TEXT')
        _ensure_column(conn, 'cases', 'clinical_history TEXT')
        _ensure_column(conn, 'cases', 'image_data BLOB')
        _ensure_column(conn, 'cases', "model TEXT DEFAULT 'program3'", 'model')
        _ensure_column(conn, 'cases', 'uploaded_at TIMESTAMP')
        _ensure_column(conn, 'cases', "gradcam_requested TEXT DEFAULT 'auto'")
        _ensure_column(conn, 'cases', 'gradcam_data BLOB')
        _ensure_column(conn, 'cases', "status TEXT DEFAULT 'pending'")
        _ensure_column(conn, 'cases', 'prediction TEXT')
        _ensure_column(conn, 'cases', 'confidence REAL')
        _ensure_column(conn, 'cases', 'processed_at TIMESTAMP')

        # Backfill user_id to admin for rows where it's NULL
        if admin_id is not None:
            cur.execute("UPDATE cases SET user_id = ? WHERE user_id IS NULL", (admin_id,))
            conn.commit()
    except sqlite3.OperationalError:
        # If table doesn't exist yet, create_tables already ensured it; ignore.
        pass

    # Sessions table should exist; nothing to migrate yet.


if __name__ == "__main__":
    # quick test
    c = get_conn()
    create_tables(c)
    print("Tables created in", settings.DB_PATH)
