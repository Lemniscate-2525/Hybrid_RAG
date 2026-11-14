"""Propose qrels against the exported corpus using the local Ollama evaluator.

Proposals remain pending until a human reviews and verifies them.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path

from evaluation.generators.qrels_proposer import QrelsProposer
from evaluation.schemas.benchmark import read_jsonl, write_jsonl


def load_chunks(path: Path) -> list[dict]:
    chunks = []
    seen_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if (
                not isinstance(item, dict)
                or not item.get("chunk_id")
                or not str(item.get("text", "")).strip()
            ):
                raise ValueError(f"Invalid corpus chunk at line {line_no}: {path}")
            chunk_id = str(item["chunk_id"])
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id {chunk_id!r} at line {line_no}")
            seen_ids.add(chunk_id)
            chunks.append(item)
    if not chunks:
        raise ValueError("Corpus snapshot contains no usable chunks.")
    return chunks


async def run(args: argparse.Namespace) -> None:
    records = read_jsonl(args.benchmark)
    chunks = load_chunks(args.corpus)
    proposer = QrelsProposer()

    pending = [
        record for record in records
        if record.expected_behavior != "abstain"
        and record.qrels_status != "verified"
        and (args.force or not record.qrels)
    ]
    if args.limit > 0:
        pending = pending[:args.limit]

    print(f"Benchmark records: {len(records)}")
    print(f"Full-corpus chunks per question: {len(chunks)}")
    print(f"Qrels proposals to make: {len(pending)}")
    print("Evaluator: local Ollama (LOCAL_EVAL_BASE_URL / LOCAL_EVAL_MODEL)")
    print("Existing verified qrels will be preserved.")

    for index, record in enumerate(pending, 1):
        compact_chunks = [
            {
                "chunk_id": str(chunk["chunk_id"]),
                "text": str(chunk["text"])[:args.chunk_chars],
            }
            for chunk in chunks
        ]
        try:
            proposal = await proposer.propose(
                question=record.query,
                chunks=compact_chunks,
                reference_answer=record.reference_answer,
                source_chunk_ids=record.source_chunk_ids,
            )
        except Exception as exc:
            write_jsonl(records, args.output)
            logging.exception(
                "Qrels proposal failed for record %s (%s/%s)",
                record.id,
                index,
                len(pending),
            )
            raise RuntimeError(
                f"Qrels proposal failed for record {record.id} "
                f"({index}/{len(pending)}): {type(exc).__name__}: {exc}. "
                f"Completed proposals saved to {args.output}."
            ) from exc

        record.qrels = proposal["qrels"]
        record.qrels_status = "pending"
        write_jsonl(records, args.output)
        print(
            f"[{index}/{len(pending)}] {record.id}: "
            f"{len(record.qrels)} relevant chunks proposed (pending review)",
            flush=True,
        )

    write_jsonl(records, args.output)
    print(f"Saved: {args.output}")
    print("No qrels were marked verified. Human review is still required.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path,
        default=Path("evaluation/results/benchmark_combined_candidates.jsonl"),
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("evaluation/results/corpus_28_chunks.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("evaluation/results/benchmark_v1_with_qrels.jsonl"),
    )
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Optional maximum number of questions to process (0 means all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-propose qrels for records with existing pending labels",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(run(parse_args()))
