
from src.config import settings
from src.db.database import SessionLocal
from src.db.models import User, DocumentRecord
from qdrant_client import QdrantClient

print("\n=== CONFIGURATION ===")
print("Qdrant URL:", settings.QDRANT_URL or "LOCAL")
print("Collection:", settings.COLLECTION_NAME)
print("Upload directory:", settings.UPLOAD_DIR)

db = SessionLocal()
client = None

try:
    print("\n=== SQLITE DOCUMENTS ===")

    users = db.query(User).all()

    for user in users:
        docs = db.query(DocumentRecord).filter(
            DocumentRecord.user_id == user.id
        ).all()

        print(f"\nUser: {user.id} ({user.email})")
        for doc in docs:
            print({
                "doc_id": doc.id,
                "filename": doc.original_filename,
                "chunks_expected": doc.chunk_count,
            })

    print("\n=== QDRANT ===")

    if settings.QDRANT_URL:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    else:
        path = str(settings.UPLOAD_DIR.parent / "qdrant_local")
        print("Local storage path:", path)
        client = QdrantClient(path=path)

    collection = settings.COLLECTION_NAME

    if not client.collection_exists(collection):
        print("COLLECTION DOES NOT EXIST:", collection)
    else:
        info = client.get_collection(collection)
        print("Collection:", collection)
        print("Total points:", info.points_count)

        points, _ = client.scroll(
            collection_name=collection,
            limit=5,
            with_payload=True,
            with_vectors=False,
        )

        print("\nSample payloads:")
        for point in points:
            p = point.payload or {}
            print({
                "user_id": p.get("user_id"),
                "doc_id": p.get("doc_id"),
                "file_name": p.get("file_name"),
            })

finally:
    db.close()
    # Let the Qdrant client be cleaned up at process exit.
