"""Marks reviewed qrels as verified and create the frozen benchmark."""

import argparse
from pathlib import Path

from evaluation.schemas.benchmark import read_jsonl, write_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type = Path,
        default = Path("evaluation/results/benchmark_v1_with_qrels.jsonl"),
    )
    parser.add_argument(
        "--output",
        type = Path,
        default=Path("evaluation/results/benchmark_v1.jsonl"),
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)

    verified = 0
    needs_review = []

    for record in records:
        if record.expected_behavior == "abstain":
            record.qrels_status = "verified"
            verified += 1
        elif record.qrels:
            record.qrels_status = "verified"
            verified += 1
        else:
            needs_review.append(record.id)

    write_jsonl(records, args.output)

    print(f"Total records: {len(records)}")
    print(f"Verified: {verified}")
    print(f"Still requiring review: {len(needs_review)}")
    print(f"Frozen benchmark candidate: {args.output}")

    if needs_review:
        print("\nAnswerable records with empty qrels:")
        for record_id in needs_review:
            print(f"  {record_id}")


if __name__ == "__main__":
    main()
