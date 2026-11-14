
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import settings

client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)

collection = settings.COLLECTION_NAME

print("Collection:", collection)

# Scroll through indexed points without retrieving their text.
offset = None
identities = defaultdict(set)

while True:
    points, offset = client.scroll(
        collection_name=collection,
        limit=256,
        offset=offset,
        with_payload=["user_id", "doc_id", "file_name"],
        with_vectors=False,
    )

    for point in points:
        payload = point.payload or {}
        user_id = payload.get("user_id")
        file_name = payload.get("file_name")
        doc_id = payload.get("doc_id")

        if user_id:
            identities[user_id].add((file_name, doc_id))

    if offset is None:
        break

print("\nCloud identities found:")

for user_id, documents in identities.items():
    print("\nUSER UUID:", user_id)
    print("Documents:")
    for file_name, doc_id in sorted(documents):
        print(f"  {file_name} | {doc_id}")

client.close()
