# PathoPulse — Implementation Blueprint & Prompt Plan (Programs 2 & 3: Backend + Frontend)

**Scope:** This file contains only the plan and prompts needed to implement **Program 2 (FastAPI backend)** and **Program 3 (Frontend)** on a separate machine from the trainer. It is self‑contained and independently runnable once a `.h5` model is copied into `program2-backend/models/`.

---

## Part A — Step‑by‑Step Blueprint (Programs 2 & 3)

### Phase 0 — Local Scaffolding (Backend + Frontend Foundation)
**Objective:** Prepare the backend server that serves static frontend files and exposes a health check.

- Create the following tree (you can use this as the repo root on the backend machine):
```
program2-backend/
├─ app/
│  ├─ main.py
│  ├─ auth.py
│  ├─ db.py
│  ├─ schemas.py
│  ├─ inference.py
│  ├─ worker.py
│  ├─ quality.py
│  ├─ pdf.py
│  ├─ config.py
│  ├─ init_db.py
│  └─ static/                # will contain the frontend files
├─ models/                   # copy latest .h5 here; auto-load newest on startup
├─ requirements.txt
└─ README.md

program3-frontend/
├─ login.html  upload.html  results.html  admin.html
├─ css/bootstrap.min.css  css/styles.css
├─ js/lib/chart.umd.min.js
├─ js/api.js  js/auth.js  js/upload.js  js/results.js  js/charts.js  js/admin.js  js/utils.js
├─ .eslintrc.json  vitest.config.js
└─ tests/api.test.js  tests/ui.test.js
```
- Add `.env` with `PORT=8080`, `DB_PATH=./endometrial.db`, `MODEL_DIR=./models`, threshold variables, and `SESSION_TTL_MINUTES=60`.
- **requirements.txt:** `fastapi`, `uvicorn[standard]`, `python-multipart`, `bcrypt`, `python-dotenv`, `pydantic`, `jinja2`, `weasyprint`, `Pillow`, `numpy`, `tensorflow==2.15.*` (CPU).

**Acceptance & Verification**
- `python program2-backend/app/init_db.py` creates SQLite DB and seeds users.
- `uvicorn program2-backend.app.main:app --host 127.0.0.1 --port 8080`
- `http://localhost:8080/health` → `{"status":"ok"}`
- `http://localhost:8080/login.html` loads (static).

---

### Phase 2 — Backend (Auth → DB → Upload → Async Inference → Artifacts → PDF)

**Step 2.1 — FastAPI Skeleton & Static Serving**
- `main.py`: mount `static/` at `/`; add `/health` and `/version`.
- `config.py`: load `.env` values and expose constants.

**Verify**
- Health endpoint returns ok; console prints env config.

**Step 2.2 — SQLite Schema & Init Script**
- `db.py` creates tables: `users`, `sessions`, `cases`, `artifacts`, `shares`, `comments` per the spec.
- `init_db.py` seeds users `admin`, `NikhilPratul`, `RupaliArora` with passwords equal to usernames (bcrypt‑hashed).

**Verify**
- DB exists; tables & users present.

**Step 2.3 — Session Auth**
- `auth.py`: `POST /auth/login` (sets httpOnly cookie), `POST /auth/logout` (clears). TTL 60 min; renew via dependency on protected routes. Basic login rate limiting in memory.

**Verify**
- Valid login creates cookie; invalid returns 401; logout 204.

**Step 2.4 — Upload Endpoint & BLOB Storage**
- `POST /cases` (multipart): accepts JPEG ≤10 MB + optional metadata fields. Stores original JPEG as BLOB in `cases.image_blob`, sets `status=processing`, returns `{id}`.

**Verify**
- Postman upload returns id; row exists; image bytes stored; status=processing.

**Step 2.5 — Async Worker & Model Loader**
- `inference.py`: load **newest .h5** from `models/` at app start; preprocess to 224×224; predict class + subtypes.
- `worker.py`: BackgroundTask that runs inference for a given case id.
- Apply atypia/borderline thresholds; write probabilities & flags to `cases`.

**Verify**
- After upload, background logs show inference; case updates to `ready` with populated fields.

**Step 2.6 — Grad‑CAM & Data Quality**
- Generate Grad‑CAM (transparent red overlay) and store PNG BLOB in `artifacts` (type=`gradcam`).
- Compute data‑quality metrics (blur, brightness/contrast, color cast proxy, JPEG artefact proxy, texture density) and save as JSON in `cases.data_quality_json`.

**Verify**
- `GET /cases/{id}/gradcam` returns PNG; JSON contains quality metrics.

**Step 2.7 — Case Retrieval APIs**
- `GET /cases?status=any|ready|processing&limit&offset`  
- `GET /cases/{id}`  
Return only **own + shared** cases; admin sees all.

**Verify**
- Postman responses match schema; 404 for unauthorized case id.

**Step 2.8 — Sharing & Comments**
- Implement `GET/POST/DELETE /cases/{id}/share` and `POST/PUT/DELETE /comments` per access rules: shared users read + comment; comment authors can edit/delete; owner cannot modify others.

**Verify**
- Sharing works; comments CRUD with permissions enforced.

**Step 2.9 — PDF Reports**
- `pdf.py` generates formal clinical PDF `Slide<ID>_Report.pdf` (WeasyPrint). Endpoint `GET /cases/{id}/report` returns BLOB (generated or stored).

**Verify**
- PDF downloads; sections present; filename correct.

---

### Phase 3 — Frontend (Auth → Upload → Results → Detail → Admin)

**Step 3.1 — Core Assets & Lint/Test Setup**
- Place local `bootstrap.min.css` and local `chart.umd.min.js` in `program3-frontend`.
- Configure `.eslintrc.json` with `eslint:recommended`; add `vitest.config.js` for DOM.

**Verify**
- `npx eslint program3-frontend` passes; `npx vitest` runs (0 tests).

**Step 3.2 — Auth Flow**
- `auth.js` implements `login/logout/getSession`; `login.html` form posts to `/auth/login`, redirects to `upload.html` on success.

**Verify**
- Manual: valid/invalid login behaviors and toasts.

**Step 3.3 — Upload UI**
- `upload.html` + `upload.js`: file chooser (JPEG ≤10 MB), optional metadata (age group, menstrual phase, specimen id, magnification, stain), notes. POST `/cases?gradcam=auto`; toast with `Slide<ID>` and link to results.

**Verify**
- Upload works; validations enforced; toast shows Slide# link.

**Step 3.4 — Results Grid & Filters**
- `results.html` + `results.js`: summary panel; collapsible filter sidebar; card grid (thumbnail, status badge, Slide#, top class, confidence bar via Chart.js, timestamp). Poll list endpoint.

**Verify**
- Cards render, filter, and update with polling.

**Step 3.5 — Detail View (Overlay + Charts)**
- On card click: load `/cases/{id}`; show original + Grad‑CAM with **opacity slider**; per‑class bar; subtype mini‑charts; EH↔EA delta/ratio; flags; metadata; notes; view/download PDF.

**Verify**
- Slider and charts function; alt text and ARIA labels present.

**Step 3.6 — Comments & Sharing UI**
- Threaded comments with author edit/delete; `@mentions` highlighted. Owner panel to add/remove usernames for sharing.

**Verify**
- End‑to‑end comments and sharing behaviors work; backend permissions respected.

**Step 3.7 — Final Polish & Accessibility**
- Add toasts for all actions; ensure responsive layout; ensure keyboard navigation and ARIA roles. Summary panel includes totals, by status/type, avg processing time, last model version.

**Verify**
- ESLint and Vitest pass; manual a11y check passes.

---

## Part B — Iterative Chunking (Programs 2 & 3)

**Round 1 — Medium Chunks**
1. Backend scaffolding + static hosting + health.
2. DB schema + init users + auth endpoints.
3. Upload endpoint + BLOB storage.
4. Async inference + thresholds.
5. Grad‑CAM + data quality + case detail API.
6. Results list + polling + charts.
7. Sharing + comments.
8. PDF reports + buttons.
9. Final UX, a11y, tests.

**Round 2 — Small Steps**
1. `main.py` + `config.py` + static mount + `/health`.
2. `db.py` schema + `init_db.py` seeding.
3. `auth.py` login/logout + session renewal dependency.
4. `POST /cases` with size/type validation + return id.
5. `inference.py` singleton model loader + predict + save probs.
6. Threshold logic (atypia/borderline) + set `ready`.
7. Grad‑CAM generation + `GET /cases/{id}/gradcam` + `GET /cases/{id}/image`.
8. Data quality metrics + include in `GET /cases/{id}`.
9. `GET /cases` list own+shared; admin sees all.
10. Frontend base assets + ESLint/Vitest configs + `utils.js`.
11. `login.html` + `auth.js` happy/error paths.
12. `upload.html` + `upload.js` multipart + toasts.
13. `results.html` + `results.js` grid + filters + polling + charts.
14. Detail view modal: overlay slider + charts + flags + metadata.
15. Comments & sharing UI + permission flows.
16. PDF endpoint + UI buttons + summary panel.
17. Final accessibility + responsive polish; all tests green.

---

## Part C — Code‑Generation LLM Prompts (Programs 2 & 3 Only)

> Execute sequentially; verify after each; commit frequently.

### Prompt B2.1 — Backend Skeleton & Health
```text
Create program2-backend/app with main.py that mounts / (static from program3-frontend) and exposes GET /health -> {"status":"ok"} and GET /version. Add config.py to read .env (PORT, DB_PATH, MODEL_DIR, thresholds). Include requirements.txt listing FastAPI, uvicorn[standard], python-multipart, bcrypt, python-dotenv, pydantic, jinja2, weasyprint, Pillow, numpy, tensorflow==2.15.* (CPU).
```

### Prompt B2.2 — DB & Seed
```text
Implement db.py schema for users, sessions, cases, artifacts, shares, comments as per spec. Implement init_db.py to create tables and seed admin, NikhilPratul, RupaliArora with bcrypt(password=username).
```

### Prompt B2.3 — Auth
```text
Implement auth.py with POST /auth/login (httpOnly cookie; 60-min TTL; simple rate limit) and POST /auth/logout. Add a dependency that renews the session on each protected route. Wire into main.py.
```

### Prompt B2.4 — Upload
```text
Implement POST /cases to accept multipart JPEG ≤10 MB with optional metadata fields. Store image bytes as BLOB in cases.image_blob, set status=processing, and return {id}.
```

### Prompt B2.5 — Inference
```text
Implement inference.py to load the newest .h5 model from models/ at startup (singleton). Preprocess to 224x224, predict main/subtypes, compute EH<->EA delta/ratio, apply thresholds from .env, and write results to cases. Wire a BackgroundTask in main.py to process uploads asynchronously.
```

### Prompt B2.6 — Grad‑CAM & Data Quality
```text
Generate a transparent-red-overlay Grad-CAM PNG and store it as an artifact (type=gradcam). Compute blur, brightness/contrast, color-cast proxy, JPEG artefact proxy, and texture density; store JSON in cases.data_quality_json. Add GET /cases/{id}/image (JPEG) and GET /cases/{id}/gradcam (PNG).
```

### Prompt B2.7 — Retrieval APIs
```text
Implement GET /cases?status=any|ready|processing&limit&offset (own+shared; admin all) and GET /cases/{id} (detail JSON). Enforce access control (404 for unauthorized).
```

### Prompt B2.8 — Sharing & Comments
```text
Implement sharing endpoints (GET/POST/DELETE /cases/{id}/share) and comments endpoints (POST/PUT/DELETE). Shared users can read and comment; only comment authors can edit/delete; owner cannot modify others' comments; admin can view all.
```

### Prompt B2.9 — PDF Reports
```text
Create pdf.py to render a formal clinical PDF (WeasyPrint) named Slide<ID>_Report.pdf including all required sections. Add GET /cases/{id}/report.
```

### Prompt F3.1 — Frontend Base
```text
Add local bootstrap.min.css and chart.umd.min.js. Create utils.js (toasts, fetch wrapper), api.js (REST calls with credentials), and minimal styles.css. Configure .eslintrc.json and vitest.config.js.
```

### Prompt F3.2 — Login Page
```text
Implement login.html and auth.js with a Bootstrap form that posts to /auth/login and redirects to upload.html on success. Handle failures with toasts.
```

### Prompt F3.3 — Upload Page
```text
Implement upload.html and upload.js to perform multipart POST /cases?gradcam=auto with validations (JPEG ≤10 MB). On success, show toast with Slide<ID> and a link to results.html.
```

### Prompt F3.4 — Results Page
```text
Implement results.html and results.js: render a summary panel, a collapsible filter sidebar, and a grid of cards (thumbnail, status, Slide#, top class, confidence bar via Chart.js, timestamp). Poll GET /cases periodically.
```

### Prompt F3.5 — Detail & Interactions
```text
On card click, load GET /cases/{id} to show a detail modal/page with overlay slider (original + Grad-CAM), per-class bar chart, subtype mini-charts, EH<->EA delta/ratio, flags, metadata table, notes, and View/Download PDF buttons.
```

### Prompt F3.6 — Comments & Sharing UI
```text
Implement threaded comments with author-only edit/delete and @mention highlighting. Add owner-only share panel to add/remove usernames. Refresh view on changes; show toasts for actions.
```

### Prompt F3.7 — Final Polish & Tests
```text
Add responsive tweaks and ARIA labels/roles. Implement Vitest tests for API wrappers and key UI flows (login, upload, results rendering). Ensure ESLint passes and tests are green.
```

---

## Part D — Verification After Each Prompt (Programs 2 & 3)

- **Backend run:** `uvicorn program2-backend.app.main:app --host 127.0.0.1 --port 8080`
- **ESLint:** `npx eslint program3-frontend`
- **Vitest:** `npx vitest`
- **Manual checks:** visit `/login.html`, `/upload.html`, `/results.html`.

---

**This Programs 2 & 3 plan is self‑contained and ready for execution on the backend/frontend machine.**
