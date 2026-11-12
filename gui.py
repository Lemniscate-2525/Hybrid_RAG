
from src.db.database import SessionLocal
from src.db.models import User

db = SessionLocal()

try:
    users = db.query(User).all()

    for user in users:
        print(f"Email: {user.email}")
        print(f"User UUID: {user.id}")
        print("-" * 40)
finally:
    db.close()
