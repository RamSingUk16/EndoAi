from fastapi import APIRouter, Depends, Request, Response, HTTPException
from pydantic import BaseModel
from .db import get_conn
from .config import settings
import bcrypt
from datetime import datetime, timedelta
import uuid
from typing import Optional

router = APIRouter()

# Simple in-memory rate limiter for login attempts by IP
_login_attempts = {}
ATTEMPT_WINDOW_SECONDS = 60
MAX_ATTEMPTS = 10


class LoginIn(BaseModel):
    username: str
    password: str


def _clean_attempts(ip: str):
    now = datetime.utcnow().timestamp()
    lst = _login_attempts.get(ip, [])
    _login_attempts[ip] = [t for t in lst if now - t < ATTEMPT_WINDOW_SECONDS]


@router.post('/auth/login')
def login(payload: LoginIn, request: Request, response: Response):
    ip = request.client.host if request.client else 'local'
    _clean_attempts(ip)
    if len(_login_attempts.get(ip, [])) >= MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail='Too many login attempts, try later')

    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT id, username, password_hash, is_admin FROM users WHERE username = ?', (payload.username,))
    row = cur.fetchone()
    if not row:
        _login_attempts.setdefault(ip, []).append(datetime.utcnow().timestamp())
        raise HTTPException(status_code=401, detail='Invalid credentials')

    stored_hash = row['password_hash']
    try:
        ok = bcrypt.checkpw(payload.password.encode('utf-8'), stored_hash)
    except Exception:
        ok = False

    if not ok:
        _login_attempts.setdefault(ip, []).append(datetime.utcnow().timestamp())
        raise HTTPException(status_code=401, detail='Invalid credentials')

    # create session
    token = uuid.uuid4().hex
    expires = datetime.utcnow() + timedelta(minutes=settings.SESSION_TTL_MINUTES)
    cur.execute('INSERT INTO sessions (user_id, session_token, expires_at) VALUES (?, ?, ?)',
                (row['id'], token, expires.isoformat()))
    conn.commit()

    # set httpOnly cookie
    max_age = settings.SESSION_TTL_MINUTES * 60
    response.set_cookie('session_token', token, httponly=True, max_age=max_age, samesite='lax')
    return {'username': row['username'], 'is_admin': bool(row['is_admin'])}


@router.post('/auth/logout')
def logout(request: Request, response: Response):
    token = request.cookies.get('session_token')
    if token:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('DELETE FROM sessions WHERE session_token = ?', (token,))
        conn.commit()

    # clear cookie
    response.delete_cookie('session_token')
    return Response(status_code=204)


def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get('session_token')
    if not token:
        return None
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute('SELECT s.user_id, u.username, u.is_admin FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.session_token = ? AND s.expires_at > ?', (token, now))
    row = cur.fetchone()
    if not row:
        return None

    # renew session expiry
    expires = datetime.utcnow() + timedelta(minutes=settings.SESSION_TTL_MINUTES)
    cur.execute('UPDATE sessions SET expires_at = ? WHERE session_token = ?', (expires.isoformat(), token))
    conn.commit()
    return {'id': row['user_id'], 'username': row['username'], 'is_admin': bool(row['is_admin'])}


@router.get('/auth/session')
def get_session(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Not authenticated')
    return {'username': user['username'], 'is_admin': user['is_admin']}


def require_auth(request: Request) -> dict:
    """
    Dependency that requires authentication.
    Raises 401 if not authenticated.
    Returns the current user dict.
    """
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required')
    return user


def require_admin(request: Request) -> dict:
    """
    Dependency that requires admin authentication.
    Raises 401 if not authenticated, 403 if not admin.
    Returns the current user dict.
    """
    user = require_auth(request)
    if not user['is_admin']:
        raise HTTPException(status_code=403, detail='Admin access required')
    return user


def session_info(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail='No active session')
    return user


__all__ = ['router', 'get_current_user', 'require_auth', 'require_admin']
