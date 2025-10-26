from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .inference_strict import ModelLoader
from .auth import router as auth_router
from .cases import router as cases_router
from .db import get_conn, create_tables, migrate_schema
import os
import sys
import platform

app = FastAPI(title="EndoServer")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": settings.VERSION}


@app.get("/model/info")
def model_info():
    """Report which model will be used and any available metadata.

    Prefers Program 3 model if available, otherwise Program 1, then MODEL_DIR.
    Returns the resolved model key and path; accuracy is included only if metadata is present.
    """
    import os
    import json
    import glob

    ml = ModelLoader()
    # Prefer program3, then program1
    p3 = ml._discover_model_path('program3')
    p1 = ml._discover_model_path('program1')

    active_key = None
    active_path = None
    if p3:
        active_key, active_path = 'program3', p3
    elif p1:
        active_key, active_path = 'program1', p1
    else:
        # Last fallback: any model in MODEL_DIR
        model_dir = settings.MODEL_DIR
        any_models = []
        if os.path.isdir(model_dir):
            any_models.extend(glob.glob(os.path.join(model_dir, "*.keras")))
            any_models.extend(glob.glob(os.path.join(model_dir, "*.h5")))
        if any_models:
            any_models = sorted(any_models, key=os.path.getmtime, reverse=True)
            active_key, active_path = 'default', any_models[0]

    # Try to find metadata next to the active model (best effort)
    val_accuracy = None
    status = 'unknown'
    try:
        if active_path:
            base = os.path.dirname(active_path)
            metadata_files = sorted(
                glob.glob(os.path.join(base, "*_metadata.json")),
                key=os.path.getmtime,
                reverse=True
            )
            if metadata_files:
                with open(metadata_files[0], 'r') as f:
                    metadata = json.load(f)
                history = metadata.get('training', {}).get('history', {})
                # Accept multiple naming variants
                for k in [
                    'val_main_accuracy',
                    'val_main_output_accuracy',
                    'val_accuracy'
                ]:
                    if k in history and history[k]:
                        val_accuracy = history[k][-1] * 100
                        break
    except Exception:
        pass

    if val_accuracy is not None:
        if val_accuracy >= 85:
            status = 'excellent'
        elif val_accuracy >= 80:
            status = 'good'
        elif val_accuracy >= 70:
            status = 'moderate'
        else:
            status = 'low'

    return {
        'model_key': active_key,
        'model_path': active_path,
        'status': status,
        'validation_accuracy': round(val_accuracy, 1) if val_accuracy is not None else None,
    }


@app.get("/debug/tf")
def debug_tf():
    """Report TensorFlow/Keras/Python environment details used by this server process."""
    tf_version = None
    keras_version = None
    tf_keras_version = None
    try:
        import tensorflow as tf  # type: ignore
        tf_version = getattr(tf, "__version__", None)
        try:
            tf_keras_version = getattr(tf.keras, "__version__", None)
        except Exception:
            tf_keras_version = None
    except Exception:
        tf_version = None
    try:
        import keras  # type: ignore
        keras_version = getattr(keras, "__version__", None)
    except Exception:
        keras_version = None

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "tensorflow": tf_version,
        "tf.keras": tf_keras_version,
        "keras": keras_version,
    }


# Ensure database schema (and at least one admin user) exists on startup
@app.on_event("startup")
def _startup_db():
    conn = get_conn()
    create_tables(conn)
    # Seed a default admin if none exists (idempotent)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username = ?", ("admin",))
    if not cur.fetchone():
        try:
            import bcrypt  # optional until login is used
            pw_hash = bcrypt.hashpw(b"admin", bcrypt.gensalt())
            cur.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                ("admin", pw_hash, 1),
            )
            conn.commit()
        except Exception:
            # Don't block server start if bcrypt isn't ready; login will error clearly
            pass
    conn.close()
    # Run non-destructive migrations to keep older DBs compatible
    conn = get_conn()
    migrate_schema(conn)
    conn.close()


# Include API routes
app.include_router(auth_router)
app.include_router(cases_router)

# Mount static files (frontend) at root - MUST BE LAST
# This catches all unmatched routes and serves static files
if os.path.exists(settings.STATIC_DIR):
    app.mount("/", StaticFiles(directory=settings.STATIC_DIR, html=True), name="static")

