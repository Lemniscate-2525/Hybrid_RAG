"""Propose qrels against the full exported corpus using LOCAL_EVAL_API.

Proposals remain pending; this script never marks labels as verified.
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
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not item.get("chunk_id") or not str(item.get("text", "")).strip():
                raise ValueError(f"Invalid chunk record at line {line_no}")
            chunks.append(item)
    if not chunks:
        raise ValueError("Corpus snapshot contains no usable chunks.")
    return chunks


async def run(args: argparse.Namespace) -> None:
    records = read_jsonl(args.benchmark)
    chunks = load_chunks(args.corpus)
    proposer = QrelsProposer()
    pending = [r for r in records if r.expected_behavior != "abstain" and r.qrels_status != "verified"]
    if args.limit:
        pending = pending[:args.limit]

    print(f"Benchmark records: {len(records)}")
    print(f"Full-corpus chunks per question: {len(chunks)}")
    print(f"Qrels proposals to make: {len(pending)}")
    print("Using LOCAL_EVAL_API. Existing verified qrels will be preserved.")

    for index, record in enumerate(pending, 1):
        # Keep payload bounded while comparing against every chunk in the corpus.
        compact_chunks = [
            {"chunk_id": str(c["chunk_id"]), "text": str(c["text"])[:args.chunk_chars]}
            for c in chunks
        ]
        proposal = await proposer.propose(
            question=record.query,
            chunks=compact_chunks,
            reference_answer=record.reference_answer,
        )
        record.qrels = proposal["qrels"]
        record.qrels_status = "pending"
        write_jsonl(records, args.output)
        print(
            f"[{index}/{len(pending)}] {record.id}: "
            f"{len(record.qrels)} relevant chunks proposed (pending review)"
        )

    # Always write output, including when there were no records to process.
    write_jsonl(records, args.output)
    print(f"Saved: {args.output}")
    print("No qrels were marked verified. Human review is still required.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("evaluation/results/benchmark_candidates.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("evaluation/results/corpus_28_chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/benchmark_with_qrels.jsonl"))
    parser.add_argument("--chunk-chars", type=int, default=1200)
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of questions to process")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(run(parse_args()))
