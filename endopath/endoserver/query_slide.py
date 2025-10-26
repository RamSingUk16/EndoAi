#!/usr/bin/env python
"""Query slide status without importing anything from app (avoids TF)."""
import sqlite3

db_path = r'C:\dev\endoui\endopath\endoserver\app\endometrial.db'
conn = sqlite3.connect(db_path)
r = conn.execute('SELECT status, prediction, confidence FROM cases WHERE slide_id=?', ('SLIDE-49E1FC56',)).fetchone()
print(f'Slide: SLIDE-49E1FC56')
print(f'  Status: {r[0]}')
print(f'  Prediction: {r[1]}')
if r[2] and r[2] > 0:
    print(f'  Confidence: {r[2]:.1%}')
else:
    print(f'  Confidence: N/A')
