import sqlite3
from typing import Optional
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


if __name__ == "__main__":
    # quick test
    c = get_conn()
    create_tables(c)
    print("Tables created in", settings.DB_PATH)
