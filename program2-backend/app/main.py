from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .config import settings

app = FastAPI(title="Program2 Backend")


# Mount static files (frontend) at root
app.mount("/", StaticFiles(directory=settings.STATIC_DIR, html=True), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"version": settings.VERSION}
