
import json
import logging
from pathlib import Path
from typing import Optional

from src.services.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)


class CorpusInspector:
    """
    Extracts and exports actual indexed chunks from Qdrant
    for evaluation dataset generation.
    """

    def __init__(self, output_dir: str = "evaluation/corpus/snapshots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_corpus(
        self,
        user_id: str,
        doc_id: Optional[str] = None,
        page_size: int = 256
    ) -> list[dict]:

        all_chunks = []
        offset = None

        while True:
            chunks, next_offset = qdrant_service.scroll_chunks(
                user_id=user_id,
                doc_id=doc_id,
                limit=page_size,
                offset=offset
            )

            all_chunks.extend(chunks)

            logger.info(
                "Fetched %d chunks; total=%d",
                len(chunks),
                len(all_chunks)
            )

            if next_offset is None:
                break

            offset = next_offset

        return all_chunks

    def export_jsonl(
        self,
        chunks: list[dict],
        filename: str = "corpus_snapshot.jsonl"
    ) -> Path:

        output_path = self.output_dir / filename

        with output_path.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )

        logger.info("Corpus snapshot exported: %s", output_path)

        return output_path

    def inspect(
        self,
        user_id: str,
        doc_id: Optional[str] = None
    ) -> dict:

        chunks = self.fetch_corpus(
            user_id=user_id,
            doc_id=doc_id
        )

        document_names = {
            chunk["file_name"] for chunk in chunks
        }

        return {
            "total_chunks": len(chunks),
            "total_documents": len(document_names),
            "documents": sorted(document_names),
            "chunks": chunks
        }
