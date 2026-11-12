"""Merge Layer A/B/C candidates, then deduplicate, validate, and cluster globally.

This script does not generate production queries or verify reference answers/qrels.
Layer C must contain real exported production queries or explicitly labeled
production-style probes; do not present synthetic probes as observed traffic.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any

from evaluation.processing.clusterer import QuestionClusterer
from evaluation.processing.deduplicator import QuestionDeduplicator
from evaluation.processing.validator import CandidateValidator
from evaluation.schemas.benchmark import BenchmarkRecord, write_jsonl

logger = logging.getLogger(__name__)


def load_jsonl(path: Path, layer: str, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        logger.warning("Layer %s file not found; skipping: %s", layer, path)
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            # Accept either canonical BenchmarkRecord rows or generator candidates.
            item = dict(item)
            item["question"] = str(item.get("question") or item.get("query") or "").strip()
            item["query"] = item["question"]
            item.setdefault("scenario_id", "production_observed" if layer == "C" else "")
            item.setdefault("category", "production" if layer == "C" else item.get("scenario_id", ""))
            item.setdefault("difficulty", "medium")
            item.setdefault("expected_behavior", "answer")
            item.setdefault("reference_answer", "")
            item.setdefault("source_chunk_ids", [])
            item.setdefault("qrels", {})
            item["qrels"] = {}  # Never carry unreviewed labels into the merged candidate set.
            item["qrels_status"] = "pending"
            item.setdefault("notes", "")
            item["notes"] = f"[Layer {layer}] {item['notes']}".strip()
            items.append(item)
    logger.info("Loaded Layer %s: %d records from %s", layer, len(items), path)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-a", type=Path, default=Path("evaluation/results/benchmark_candidates.jsonl"))
    parser.add_argument("--layer-b", type=Path, default=Path("evaluation/results/benchmark_layer_b_candidates.jsonl"))
    parser.add_argument("--layer-c", type=Path, default=Path("evaluation/results/benchmark_layer_c_candidates.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("evaluation/results/corpus_28_chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/benchmark_combined_candidates.jsonl"))
    parser.add_argument("--dedupe-threshold", type=float, default=0.88)
    parser.add_argument("--cluster-threshold", type=float, default=0.22)
    parser.add_argument("--allow-missing-c", action="store_true",
                        help="Allow A+B merge when no Layer C export exists yet.")
    args = parser.parse_args()

    with args.corpus.open("r", encoding="utf-8") as handle:
        corpus = [json.loads(line) for line in handle if line.strip()]
    if not corpus:
        raise ValueError("Corpus snapshot is empty.")

    candidates = []
    candidates.extend(load_jsonl(args.layer_a, "A"))
    candidates.extend(load_jsonl(args.layer_b, "B"))
    layer_c = load_jsonl(args.layer_c, "C", required=not args.allow_missing_c)
    candidates.extend(layer_c)
    if not candidates:
        raise ValueError("No benchmark candidates were loaded.")

    deduplicator = QuestionDeduplicator(args.dedupe_threshold)
    unique, duplicates = deduplicator.deduplicate(candidates, question_key="question")
    accepted, rejected = CandidateValidator().validate(unique, corpus)

    records = []
    for item in accepted:
        record_data = {
            "query": item["question"],
            "scenario_id": item["scenario_id"],
            "category": item.get("category", item["scenario_id"]),
            "difficulty": item.get("difficulty", "medium"),
            "expected_behavior": item["expected_behavior"],
            "reference_answer": item.get("reference_answer", ""),
            "source_chunk_ids": item.get("source_chunk_ids", []),
            "qrels": {},
            "qrels_status": "pending",
            "id": item.get("id") or None,
            "corpus_snapshot": item.get("corpus_snapshot"),
            "notes": item.get("notes", ""),
            "split": item.get("split", "dev"),
        }
        if record_data["id"] is None:
            record_data.pop("id")
        if record_data["corpus_snapshot"] is None:
            record_data.pop("corpus_snapshot")
        records.append(BenchmarkRecord.from_dict(record_data))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, args.output)
    for suffix, rows in (
        (".duplicates.jsonl", duplicates),
        (".rejected.jsonl", rejected),
    ):
        with args.output.with_name(args.output.stem + suffix).open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\\n")

    clusters = QuestionClusterer(args.cluster_threshold).cluster(accepted)
    cluster_path = args.output.with_name(args.output.stem + ".clusters.json")
    with cluster_path.open("w", encoding="utf-8") as out:
        json.dump(clusters, out, ensure_ascii=False, indent=2)

    print("\\nCombined candidate curation complete")
    print(f"Layer A: {sum(1 for x in candidates if '[Layer A]' in x.get('notes', ''))}")
    print(f"Layer B: {sum(1 for x in candidates if '[Layer B]' in x.get('notes', ''))}")
    print(f"Layer C: {len(layer_c)}")
    print(f"Unique after dedup: {len(unique)}")
    print(f"Accepted for human review: {len(records)}")
    print(f"Rejected: {len(rejected)} | Duplicates: {len(duplicates)}")
    print(f"Output: {args.output}")
    print(f"Clusters: {cluster_path}")
    print("All qrels were reset to pending/empty. Human review is still required.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
