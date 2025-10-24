# TODO — Programs 2 & 3 (Backend + Frontend)

## Backend Setup
- [ ] Create `.env` with PORT, DB_PATH, MODEL_DIR, thresholds, SESSION_TTL_MINUTES
- [ ] `pip install -r program2-backend/requirements.txt`
- [ ] `python program2-backend/app/init_db.py` to seed users
- [ ] Run server and check `/health`

## Auth
- [ ] Implement `POST /auth/login` and `POST /auth/logout`
- [ ] Session renewal dependency (60-min TTL)
- [ ] Basic rate-limit for login

## Upload & Inference
- [ ] Implement `POST /cases` (JPEG ≤10 MB) storing BLOB + metadata
- [ ] Copy latest `.h5` into `program2-backend/models/`
- [ ] Background inference: preprocess, predict, thresholds, save results

## Artifacts & Quality
- [ ] Generate Grad‑CAM PNG and store as artifact
- [ ] Compute data-quality metrics JSON

## Retrieval & Sharing
- [ ] Implement `GET /cases` list; `GET /cases/{id}` detail
- [ ] Implement `GET /cases/{id}/image`, `GET /cases/{id}/gradcam`, `GET /cases/{id}/report`
- [ ] Implement share endpoints and comments endpoints
- [ ] Enforce permissions: owner vs shared vs admin

## Frontend Setup
- [ ] Place local Bootstrap and Chart.js
- [ ] Configure ESLint and Vitest
- [ ] Implement `utils.js` and `api.js`

## Frontend Screens
- [ ] `login.html` + `auth.js` (toasts, redirect)
- [ ] `upload.html` + `upload.js` (multipart, validations)
- [ ] `results.html` + `results.js` (summary, filters, grid, polling, charts)
- [ ] Detail view: overlay slider; subtype charts; flags; metadata; notes; PDF buttons
- [ ] Comments & Sharing UI

## Accessibility & QA
- [ ] ARIA labels and alt text present
- [ ] Keyboard navigation verified
- [ ] Responsive layout OK
- [ ] ESLint + Vitest pass

## Demo Readiness
- [ ] End-to-end manual walkthrough complete
- [ ] Prepare screenshots and sample PDFs
