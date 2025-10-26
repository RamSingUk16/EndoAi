import sqlite3
from collections import Counter

db = 'app/endometrial.db'
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row
c = conn.cursor()

c.execute("SELECT COUNT(*) FROM cases")
print('Total cases:', c.fetchone()[0])

c.execute("SELECT status, COUNT(*) cnt FROM cases GROUP BY status")
counts = {row['status']: row['cnt'] for row in c.fetchall()}
print('By status:', counts)

print('\nSample (latest 5):')
c.execute("""
SELECT slide_id, status, prediction, confidence, uploaded_at, processed_at
FROM cases
ORDER BY processed_at DESC NULLS LAST, uploaded_at DESC
LIMIT 5
""")
for row in c.fetchall():
    d = dict(row)
    pred = d.get('prediction')
    if pred and len(pred) > 120:
        pred = pred[:120] + '...'
    d['prediction'] = pred
    print(d)
