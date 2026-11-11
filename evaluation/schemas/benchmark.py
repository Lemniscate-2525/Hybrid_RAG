"""Canonical benchmark record and JSONL serialization helpers."""
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ExpectedBehavior = Literal["answer", "abstain", "partial"]
QrelsStatus = Literal["pending", "verified"]

@dataclass
class BenchmarkRecord:
    query: str
    scenario_id: str
    category: str
    difficulty: str
    expected_behavior: ExpectedBehavior
    reference_answer: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)
    qrels: dict[str, int] = field(default_factory=dict)
    qrels_status: QrelsStatus = "pending"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    corpus_snapshot: str | None = None
    notes: str = ""
    split: Literal["train", "dev", "test"] = "dev"

    def validate(self) -> None:
        if not self.query.strip():
            raise ValueError("query cannot be empty")
        if self.expected_behavior not in {"answer", "abstain", "partial"}:
            raise ValueError("invalid expected_behavior")
        if self.qrels_status not in {"pending", "verified"}:
            raise ValueError("invalid qrels_status")
        if any(not isinstance(grade, int) or grade not in (1, 2) for grade in self.qrels.values()):
            raise ValueError("qrels grades must be integers 1 or 2")
        if self.expected_behavior == "abstain" and self.qrels:
            raise ValueError("abstain records must not contain relevant chunk labels")
        if self.qrels_status == "verified" and not self.qrels and self.expected_behavior != "abstain":
            raise ValueError("empty verified qrels are only valid for abstention cases")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRecord":
        allowed = cls.__dataclass_fields__.keys()
        record = cls(**{key: value for key, value in data.items() if key in allowed})
        record.validate()
        return record

def write_jsonl(records: list[BenchmarkRecord], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return destination

def read_jsonl(path: str | Path) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(BenchmarkRecord.from_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"Invalid benchmark record at line {line_number}: {exc}") from exc
    return records
