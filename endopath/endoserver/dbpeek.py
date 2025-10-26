import sqlite3
import sys
slide_id = sys.argv[1] if len(sys.argv) > 1 else None
conn = sqlite3.connect('app/endometrial.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()
c.execute("SELECT slide_id,status,prediction,confidence FROM cases WHERE slide_id=?", (slide_id,))
row = c.fetchone()
if row:
    d = dict(row)
    print('status:', d.get('status'))
    print('confidence:', d.get('confidence'))
    pred = d.get('prediction')
    print('prediction_len:', len(pred) if pred is not None else None)
    print('prediction_preview:', (pred[:200] + '...') if isinstance(pred, str) and len(pred) > 200 else pred)
else:
    print('No such slide')
