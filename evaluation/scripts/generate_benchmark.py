"""Generate, clean, validate, and package a corpus-grounded RAG benchmark.

Input: JSONL containing one chunk per line with chunk_id, text, and optional
doc_id/file_name. Keep corpus snapshots and generated datasets out of Git.
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from evaluation.generators.question_generator import QuestionGenerator
from evaluation.processing.clusterer import QuestionClusterer
from evaluation.processing.deduplicator import QuestionDeduplicator
from evaluation.processing.validator import CandidateValidator
from evaluation.schemas.benchmark import BenchmarkRecord, write_jsonl


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_no} must contain a JSON object.")
            if item.get("chunk_id") and str(item.get("text", "")).strip():
                chunks.append(item)
    return chunks


async def run(args: argparse.Namespace) -> None:
    chunks = load_chunks(args.input)
    if not chunks:
        raise SystemExit("No usable chunks found. Each JSONL row needs chunk_id and text.")

    generator = QuestionGenerator()
    candidates = await generator.generate(
        chunks,
        questions_per_scenario=args.per_scenario,
        max_chunks=args.max_chunks,
        chunks_per_batch=args.chunks_per_batch,
        max_candidates_per_batch=args.max_candidates_per_batch,
        scenario_ids=args.scenario_ids,
    )

    deduplicator = QuestionDeduplicator(similarity_threshold=args.dedupe_threshold)
    unique, duplicates = deduplicator.deduplicate(candidates)

    validator = CandidateValidator()
    accepted, rejected = validator.validate(unique, chunks)

    records = [
        BenchmarkRecord(
            query=item["question"],
            scenario_id=item["scenario_id"],
            category=item.get("category", item["scenario_id"]),
            difficulty=item.get("difficulty", "medium"),
            expected_behavior=item["expected_behavior"],
            reference_answer=item.get("reference_answer", ""),
            source_chunk_ids=item.get("source_chunk_ids", []),
            qrels={},
            qrels_status="pending",
            notes=item.get("notes", ""),
            split=args.split,
        )
        for item in accepted
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, args.output)

    clusters = QuestionClusterer(similarity_threshold=args.cluster_threshold).cluster(accepted)
    cluster_path = args.output.with_name(args.output.stem + ".clusters.json")
    with cluster_path.open("w", encoding="utf-8") as handle:
        json.dump(clusters, handle, ensure_ascii=False, indent=2)

    rejected_path = args.output.with_name(args.output.stem + ".rejected.jsonl")
    with rejected_path.open("w", encoding="utf-8") as handle:
        for item in rejected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    duplicate_path = args.output.with_name(args.output.stem + ".duplicates.jsonl")
    with duplicate_path.open("w", encoding="utf-8") as handle:
        for item in duplicates:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\nBenchmark candidate generation complete")
    print(f"Chunks read:       {len(chunks)}")
    print(f"Candidates made:   {len(candidates)}")
    print(f"Duplicates:        {len(duplicates)}")
    print(f"Accepted:          {len(records)}")
    print(f"Rejected:          {len(rejected)}")
    print(f"Review clusters:   {len(clusters)}")
    print(f"Benchmark JSONL:   {args.output}")
    print(f"Clusters:          {cluster_path}")
    print(f"Rejected:          {rejected_path}")
    print(f"Duplicates:        {duplicate_path}")
    print("\nAll qrels remain pending. Review candidates and assign/verify qrels before scoring.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Corpus chunk snapshot JSONL")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/benchmark_candidates.jsonl"))
    parser.add_argument("--max-chunks", type=int, default=40)
    parser.add_argument("--chunks-per-batch", type=int, default=2)
    parser.add_argument("--max-candidates-per-batch", type=int, default=2)
    parser.add_argument("--per-scenario", type=int, default=1)
    parser.add_argument("--scenario-ids", type=lambda value: [item.strip() for item in value.split(",") if item.strip()], default=None, help="Optional comma-separated scenario IDs to generate")
    parser.add_argument("--dedupe-threshold", type=float, default=0.88)
    parser.add_argument("--cluster-threshold", type=float, default=0.22)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    asyncio.run(run(parse_args()))
