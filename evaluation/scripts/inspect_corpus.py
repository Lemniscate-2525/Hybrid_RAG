
import argparse
import logging
from pathlib import Path

from evaluation.corpus.inspector import CorpusInspector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and export indexed Qdrant corpus."
    )

    parser.add_argument(
        "--user-id",
        required=True,
        help="User UUID whose indexed corpus should be inspected."
    )

    parser.add_argument(
        "--doc-id",
        default=None,
        help="Optional document UUID to restrict inspection."
    )

    parser.add_argument(
        "--output",
        default="corpus_snapshot.jsonl",
        help="Output JSONL filename."
    )

    args = parser.parse_args()

    inspector = CorpusInspector()

    logger.info("Starting corpus inspection...")

    result = inspector.inspect(
        user_id=args.user_id,
        doc_id=args.doc_id
    )

    output_path = inspector.export_jsonl(
        result["chunks"],
        filename=args.output
    )

    chunks = result["chunks"]

    unique_chunk_ids = {
        chunk["chunk_id"] for chunk in chunks
    }

    missing_text = sum(
        not chunk.get("text", "").strip()
        for chunk in chunks
    )

    logger.info("Corpus inspection completed.")
    logger.info("Documents: %d", result["total_documents"])
    logger.info("Chunks: %d", result["total_chunks"])
    logger.info("Unique chunk IDs: %d", len(unique_chunk_ids))
    logger.info("Chunks with missing text: %d", missing_text)
    logger.info("Snapshot: %s", output_path)


if __name__ == "__main__":
    main()
