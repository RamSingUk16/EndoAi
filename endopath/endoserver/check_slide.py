from app.db import get_conn
c = get_conn().cursor()
c.execute("SELECT slide_id, status, prediction, confidence FROM cases WHERE slide_id=?", ("SLIDE-49E1FC56",))
row = c.fetchone()
if row:
    print(dict(row))
else:
    print("No such slide")
