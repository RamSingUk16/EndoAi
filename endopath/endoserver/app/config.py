import os
from pydantic_settings import BaseSettings


# Compute base directories once
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # endopath/endoserver

# Determine a sensible default model directory:
# 1) Prefer models produced by trainer under endopath/program1-trainer/models
# 2) Fallback to endopath/endoserver/models
_DEFAULT_MODEL_DIR = os.environ.get("MODEL_DIR")
if not _DEFAULT_MODEL_DIR:
    trainer_models = os.path.normpath(os.path.join(_BASE_DIR, "..", "program1-trainer", "models"))
    server_models = os.path.join(_BASE_DIR, "models")
    if os.path.exists(trainer_models):
        _DEFAULT_MODEL_DIR = trainer_models
    else:
        _DEFAULT_MODEL_DIR = server_models


class Settings(BaseSettings):
    PORT: int = 8080
    DB_PATH: str = "./endometrial.db"
    MODEL_DIR: str = _DEFAULT_MODEL_DIR
    SESSION_TTL_MINUTES: int = 60
    # Default static dir: endopath/endoserver/static (can be changed via env)
    BASE_DIR: str = _BASE_DIR
    STATIC_DIR: str = os.environ.get("STATIC_DIR", os.path.join(_BASE_DIR, "static"))
    VERSION: str = os.environ.get("APP_VERSION", "0.1.0")


settings = Settings()
