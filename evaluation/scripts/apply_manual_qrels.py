import json
import shutil
from pathlib import Path

path = Path("evaluation/results/benchmark_v1_with_qrels.jsonl")

assignments = {
    "8cc2820e-5c58-4c40-a493-00dca4d387e7": "281a3b2c-62ed-5c08-9675-a9ddbb887857",
    "c294bbd7-9b4c-4469-a4eb-3604b68c88bb": "281a3b2c-62ed-5c08-9675-a9ddbb887857",
    "a760659f-11c6-4a74-a37c-b8f64c02c89b": "281a3b2c-62ed-5c08-9675-a9ddbb887857",
    "dc5ba205-c6c8-4609-ba96-0858a76ca895": "281a3b2c-62ed-5c08-9675-a9ddbb887857",
    "31aa2e03-ec98-4465-9bc6-286eb1ecc862": "06c2ba2c-8cc6-53b6-8e7e-80317789a2a8",
    "8fc28482-a159-4a5b-b9d1-27ca11acaec9": "83e17598-b8ea-5d49-b501-577f15539377",
    "2c6b7f9a-7e2d-4f1b-9a31-8c5d2e6f1a02": "83e17598-b8ea-5d49-b501-577f15539377",
    "2c6b7f9a-7e2d-4f1b-9a31-8c5d2e6f1a19": "e7d573ad-e81b-539f-8e68-bc255dfdd025",
    "2c6b7f9a-7e2d-4f1b-9a31-8c5d2e6f1a20": "e7d573ad-e81b-539f-8e68-bc255dfdd025",
    "2c6b7f9a-7e2d-4f1b-9a31-8c5d2e6f1a22": "e7d573ad-e81b-539f-8e68-bc255dfdd025",
}

records = []
with path.open("r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

by_id = {r["id"]: r for r in records}

missing = set(assignments) - set(by_id)
if missing:
    raise ValueError(f"Missing benchmark IDs: {sorted(missing)}")

# Preserve the original file before modifying it : 
backup = path.with_suffix(".before_manual_review.bak")
shutil.copy2(path, backup)

for question_id, chunk_id in assignments.items():
    record = by_id[question_id]

    if chunk_id not in record.get("source_chunk_ids", []):
        raise ValueError(
            f"Chunk {chunk_id} is not a source hint for {question_id}"
        )

    record["qrels"] = {chunk_id: 2}
    record["qrels_status"] = "verified"

# Correcting unsupported calendar year and sharpening Phase 2 answer : 
by_id["dc5ba205-c6c8-4609-ba96-0858a76ca895"][
    "reference_answer"
] = (
    "The evaluation uses the Phase-1 training cutoff date, "
    "when the Phase-1 model is the freshest. The cited evidence "
    "does not specify a calendar year."
)

by_id["8cc2820e-5c58-4c40-a493-00dca4d387e7"][
    "reference_answer"
] = (
    "Phase 2 uses task-specific post-training, including "
    "domain-specific data, rewards, and verbalizations."
)

with path.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Updated: {len(assignments)} records")
print(f"Backup: {backup}")
print(f"Saved: {path}")
