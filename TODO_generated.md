# TODO (Generated) — Programs 2 & 3 (Backend + Frontend)

This file was generated from `prompt_plan_backend_frontend.md` and `todo_backend_frontend.md`.

## How to use
- Edit items and mark them with `- [x]` when done.
- Tasks are grouped by area and ordered roughly by implementation sequence.

---

## Backend — Scaffolding & Core
- [ ] Create project tree `endopath/endoserver/` with `app/`, `models/`, `requirements.txt`, `README.md`.
- [ ] Add `.env` with PORT, DB_PATH, MODEL_DIR, thresholds, SESSION_TTL_MINUTES.
- [ ] Implement `endopath/endoserver/app/main.py` mounting `static/` and exposing `/health` and `/version`.
- [ ] Create `config.py` to load `.env`.
- [ ] Create `endopath/endoserver/app/init_db.py` to create SQLite DB and seed users.
- [ ] Verify: run server and confirm `/health` -> {"status":"ok"}.

## Backend — Database & Auth
- [ ] Implement `db.py` schema: users, sessions, cases, artifacts, shares, comments.
- [ ] Seed users: `admin`, `NikhilPratul`, `RupaliArora` (bcrypt hashed; password=username).
- [ ] Implement `auth.py` with `POST /auth/login` (httpOnly cookie) and `POST /auth/logout` and a session renewal dependency (TTL 60m).
- [ ] Add basic login rate-limiting.

## Backend — Upload, Inference & Worker
- [ ] Implement `POST /cases` (multipart) to accept JPEG ≤10 MB, store image BLOB, metadata, set `status=processing` and return `{id}`.
- [ ] Add model directory `endopath/endoserver/models/` and copy latest `.h5` there.
- [ ] Implement `inference.py` to load newest `.h5` at startup (singleton), preprocess to 224×224, predict classes/subtypes, compute EH↔EA delta/ratio, and apply thresholds.
- [ ] Implement `worker.py` (BackgroundTask) to run inference asynchronously and update case status to `ready`.

## Backend — Artifacts & Quality Metrics
- [ ] Generate Grad‑CAM PNG overlay and store in `artifacts` (type=`gradcam`).
- [ ] Compute data-quality metrics (blur, brightness/contrast, color-cast proxy, JPEG artefact proxy, texture density) and save JSON to `cases.data_quality_json`.
- [ ] Implement endpoints: `GET /cases/{id}/image`, `GET /cases/{id}/gradcam`.

## Backend — Retrieval, Sharing & Comments
- [ ] Implement `GET /cases` list with filters (status, limit, offset) returning own+shared cases (admin sees all).
- [ ] Implement `GET /cases/{id}` detail endpoint and enforce access control (404 when unauthorized).
- [ ] Implement sharing endpoints (`GET/POST/DELETE /cases/{id}/share`).
- [ ] Implement comments endpoints (`POST/PUT/DELETE /comments`) with permission rules.

## Backend — PDF Reports
- [ ] Implement `pdf.py` to render a clinical PDF report (WeasyPrint) named `Slide<ID>_Report.pdf` and endpoint `GET /cases/{id}/report`.

## Frontend — Setup & Tooling
- [ ] Create `endoui/` with `login.html`, `upload.html`, `results.html`, `admin.html` and assets (`css/`, `js/`).
- [ ] Add local `css/bootstrap.min.css` and `js/lib/chart.umd.min.js`.
- [ ] Configure `.eslintrc.json` and `vitest.config.js` for linting and tests.
- [ ] Implement `js/utils.js` and `js/api.js` (fetch wrappers with credentials + toasts).

## Frontend — Auth & Upload
- [ ] Implement `login.html` and `js/auth.js` to call `POST /auth/login`, handle cookies, and redirect to `upload.html` on success.
- [ ] Implement `upload.html` and `js/upload.js` to upload JPEGs (≤10 MB) with metadata; show toast with Slide# and link to results.

## Frontend — Results & Detail
- [ ] Implement `results.html` and `js/results.js` to show summary panel, filters, and card grid with polling.
- [ ] Implement detail view for a case: original image, Grad‑CAM overlay with opacity slider, per-class and subtype charts, flags, metadata, notes, and PDF download.
- [ ] Implement comments UI and sharing UI with permission-aware actions.

## QA, Accessibility & Tests
- [ ] Add ARIA labels, alt text, and keyboard navigation checks.
- [ ] Ensure responsive design and toasts for user actions.
- [ ] Add Vitest tests for API wrappers and core UI flows; ensure ESLint passes.

---

## Next steps (suggested)
1. Run `python program2-backend/app/init_db.py` to create DB and seed users.
2. Start backend: `uvicorn program2-backend.app.main:app --host 127.0.0.1 --port 8080` and verify `/health`.
3. Implement `POST /cases` and the inference worker; test upload + background processing.

---

_File created automatically from workspace Markdown on Oct 24, 2025._
