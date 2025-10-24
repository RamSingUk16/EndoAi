from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .auth import router as auth_router
from .cases import router as cases_router
import os

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


# Include API routes
app.include_router(auth_router)
app.include_router(cases_router)

# Mount static files (frontend) at root - MUST BE LAST
# This catches all unmatched routes and serves static files
if os.path.exists(settings.STATIC_DIR):
    app.mount("/", StaticFiles(directory=settings.STATIC_DIR, html=True), name="static")

