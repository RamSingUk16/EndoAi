"""Upload and case management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from typing import Optional
import uuid
from datetime import datetime
from .db import get_conn
from .auth import require_auth
from .inference import process_case_background

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post('/cases')
async def upload_case(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    age: Optional[int] = Form(None),
    clinical_history: Optional[str] = Form(None),
    gradcam: str = Form('auto'),
    user: dict = Depends(require_auth)
):
    """
    Upload a case image (JPEG) with metadata.
    
    Args:
        file: JPEG image file (max 10 MB)
        patient_id: Patient identifier
        age: Patient age (optional)
        clinical_history: Clinical history notes (optional)
        gradcam: GradCAM setting ('auto', 'on', 'off')
    
    Returns:
        Case ID and slide ID
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/jpeg'):
        raise HTTPException(status_code=400, detail='Only JPEG images are supported')
    
    # Read file content
    contents = await file.read()
    
    # Validate file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f'File size exceeds {MAX_FILE_SIZE / 1024 / 1024} MB limit')
    
    # Validate gradcam parameter
    if gradcam not in ['auto', 'on', 'off']:
        raise HTTPException(status_code=400, detail='gradcam must be "auto", "on", or "off"')
    
    # Debug logging
    print(f"DEBUG: Uploading case - patient_id={patient_id}, age={age}, clinical_history={clinical_history}")
    
    # Generate case ID
    case_id = str(uuid.uuid4())
    slide_id = f"SLIDE-{uuid.uuid4().hex[:8].upper()}"
    
    # Store in database
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        cur.execute('''
            INSERT INTO cases (
                id, user_id, slide_id, patient_id, age, 
                clinical_history, image_data, filename,
                gradcam_requested, status, uploaded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            case_id,
            user['id'],
            slide_id,
            patient_id,
            age,
            clinical_history,
            contents,
            file.filename,
            gradcam,
            'pending',
            datetime.utcnow().isoformat()
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f'Failed to store case: {str(e)}')
    
    # Trigger background inference
    background_tasks.add_task(process_case_background, case_id)
    
    return {
        'id': case_id,
        'slide_id': slide_id,
        'status': 'pending',
        'message': 'Case uploaded successfully'
    }



@router.get('/cases/{case_id}')
def get_case(case_id: str, user: dict = Depends(require_auth)):
    """Get case details by ID."""
    conn = get_conn()
    cur = conn.cursor()
    
    # Get case info (excluding image_data)
    cur.execute('''
        SELECT id, user_id, slide_id, patient_id, age, clinical_history,
               filename, gradcam_requested, status, prediction,
               confidence, uploaded_at, processed_at
        FROM cases
        WHERE id = ?
    ''', (case_id,))
    
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail='Case not found')
    
    # Check if user owns this case or is admin
    cur.execute('SELECT user_id FROM cases WHERE id = ?', (case_id,))
    owner = cur.fetchone()
    if owner['user_id'] != user['id'] and not user['is_admin']:
        raise HTTPException(status_code=403, detail='Access denied')
    
    return dict(row)


@router.get('/cases')
def list_cases(limit: int = 50, offset: int = 0, user: dict = Depends(require_auth)):
    """List cases for the current user."""
    conn = get_conn()
    cur = conn.cursor()
    
    # Admins see all cases, regular users see only their own
    if user['is_admin']:
        cur.execute('''
            SELECT id, user_id, slide_id, patient_id, clinical_history, 
                   status, prediction, confidence, uploaded_at, processed_at
            FROM cases
            ORDER BY uploaded_at DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))
    else:
        cur.execute('''
            SELECT id, user_id, slide_id, patient_id, clinical_history,
                   status, prediction, confidence, uploaded_at, processed_at
            FROM cases
            WHERE user_id = ?
            ORDER BY uploaded_at DESC
            LIMIT ? OFFSET ?
        ''', (user['id'], limit, offset))
    
    rows = cur.fetchall()
    cases = [dict(row) for row in rows]
    
    # Debug logging
    if cases:
        print(f"DEBUG: Returning {len(cases)} cases")
        print(f"DEBUG: First case clinical_history: {cases[0].get('clinical_history', 'N/A')}")
    
    return {'cases': cases}


@router.get('/cases/{case_id}/gradcam')
def get_gradcam(case_id: str, user: dict = Depends(require_auth)):
    """Get GradCAM visualization for a case."""
    from fastapi.responses import Response
    
    conn = get_conn()
    cur = conn.cursor()
    
    # Get case with gradcam_data
    cur.execute('''
        SELECT user_id, gradcam_data, status
        FROM cases
        WHERE id = ?
    ''', (case_id,))
    
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail='Case not found')
    
    # Check access
    if row['user_id'] != user['id'] and not user['is_admin']:
        raise HTTPException(status_code=403, detail='Access denied')
    
    # Check if GradCAM is available
    if not row['gradcam_data']:
        if row['status'] == 'pending' or row['status'] == 'processing':
            raise HTTPException(status_code=202, detail='GradCAM is still being generated')
        else:
            raise HTTPException(status_code=404, detail='GradCAM not available for this case')
    
    # Return JPEG image
    return Response(content=row['gradcam_data'], media_type='image/jpeg')


@router.get('/cases/{case_id}/image')
def get_original_image(case_id: str, user: dict = Depends(require_auth)):
    """Get original image for a case."""
    from fastapi.responses import Response
    
    print(f"DEBUG: Fetching image for case_id={case_id}")
    
    conn = get_conn()
    cur = conn.cursor()
    
    # Get case with image_data
    cur.execute('''
        SELECT user_id, image_data
        FROM cases
        WHERE id = ?
    ''', (case_id,))
    
    row = cur.fetchone()
    if not row:
        print(f"DEBUG: Case {case_id} not found")
        raise HTTPException(status_code=404, detail='Case not found')
    
    # Check access
    if row['user_id'] != user['id'] and not user['is_admin']:
        print(f"DEBUG: Access denied for user {user['id']} to case owned by {row['user_id']}")
        raise HTTPException(status_code=403, detail='Access denied')
    
    # Check if image is available
    if not row['image_data']:
        print(f"DEBUG: No image_data for case {case_id}")
        raise HTTPException(status_code=404, detail='Image not found')
    
    print(f"DEBUG: Returning image for case {case_id}, size={len(row['image_data'])} bytes")
    # Return JPEG image
    return Response(content=row['image_data'], media_type='image/jpeg')
