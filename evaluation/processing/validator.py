"""Structural and evidence-grounding checks for generated benchmark candidates."""
from typing import Any, Iterable

_REQUIRED = ("question", "scenario_id", "expected_behavior")

class CandidateValidator:
    """Validate candidate shape and evidence references; this is not a substitute for review."""

    def validate(
        self,
        candidates: Iterable[dict[str, Any]],
        corpus_chunks: Iterable[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        chunks = {str(c.get("chunk_id")): c for c in corpus_chunks if c.get("chunk_id")}
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for original in candidates:
            item = dict(original)
            errors: list[str] = []
            for field in _REQUIRED:
                if not str(item.get(field, "")).strip():
                    errors.append(f"missing_{field}")
            if item.get("expected_behavior") not in {"answer", "abstain", "partial"}:
                errors.append("invalid_expected_behavior")

            source_ids = item.get("source_chunk_ids", [])
            if not isinstance(source_ids, list):
                errors.append("source_chunk_ids_not_list")
                source_ids = []
            unknown_ids = [str(cid) for cid in source_ids if str(cid) not in chunks]
            if unknown_ids:
                errors.append("unknown_source_chunk_ids")
            item["source_chunk_ids"] = [str(cid) for cid in source_ids if str(cid) in chunks]

            behavior = item.get("expected_behavior")
            if behavior == "answer" and not item["source_chunk_ids"]:
                errors.append("answer_candidate_without_evidence")
            if behavior == "abstain" and item["source_chunk_ids"]:
                errors.append("abstain_candidate_has_evidence")
            if behavior != "abstain" and not str(item.get("reference_answer", "")).strip():
                errors.append("missing_reference_answer")

            item["validation_errors"] = errors
            item["validation_status"] = "rejected" if errors else "needs_human_review"
            (rejected if errors else accepted).append(item)
        return accepted, rejected
