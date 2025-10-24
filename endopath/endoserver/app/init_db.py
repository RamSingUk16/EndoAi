"""Initialize the SQLite database and seed initial users.

Run: python endopath/endoserver/app/init_db.py
"""
import bcrypt
import os
from .db import get_conn, create_tables
from .config import settings


def seed_users(conn):
    users = ["admin", "NikhilPratul", "RupaliArora"]
    cur = conn.cursor()
    for username in users:
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone():
            print(f"user '{username}' already exists, skipping")
            continue

        password = username.encode("utf-8")
        pw_hash = bcrypt.hashpw(password, bcrypt.gensalt())
        is_admin = 1 if username == "admin" else 0
        cur.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            (username, pw_hash, is_admin),
        )
        print(f"seeded user: {username}")

    conn.commit()


def main():
    # Ensure DB dir exists
    db_dir = os.path.dirname(settings.DB_PATH) or "."
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = get_conn()
    create_tables(conn)
    seed_users(conn)
    conn.close()
    print("Database initialized at:", settings.DB_PATH)


if __name__ == "__main__":
    main()
