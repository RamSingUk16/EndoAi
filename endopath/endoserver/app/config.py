import os
from pydantic import BaseSettings


class Settings(BaseSettings):
    PORT: int = 8080
    DB_PATH: str = "./endometrial.db"
    MODEL_DIR: str = "./models"
    SESSION_TTL_MINUTES: int = 60
    # Default static dir: endopath/endoserver/static (can be changed via env)
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    STATIC_DIR: str = os.environ.get("STATIC_DIR", os.path.join(BASE_DIR, "static"))
    VERSION: str = os.environ.get("APP_VERSION", "0.1.0")


settings = Settings()
