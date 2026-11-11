"""Reusable RAG evaluation scenarios, independent of any specific corpus.

These are scenario specifications, not a universal question/answer dataset:
questions and ground-truth labels must still be grounded in each application's corpus.
"""
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    category: str
    description: str
    instruction: str
    expected_behavior: Literal["answer", "abstain", "partial"]
    difficulty: Literal["easy", "medium", "hard"]

SCENARIOS: tuple[Scenario, ...] = (
    Scenario("direct_fact", "factual", "Retrieve an explicitly stated fact.",
             "Ask for one specific fact directly stated in the evidence.", "answer", "easy"),
    Scenario("paraphrase", "semantic", "Ask using different wording from the source.",
             "Paraphrase a source fact without reusing its distinctive wording.", "answer", "medium"),
    Scenario("lexical_mismatch", "retrieval", "Test semantic retrieval despite vocabulary mismatch.",
             "Ask about a concept using a synonym or related phrase absent from the source wording.", "answer", "medium"),
    Scenario("multi_hop", "reasoning", "Require combining evidence from multiple chunks.",
             "Ask a question that requires combining at least two supplied evidence chunks. Do not use this scenario unless two chunks jointly support the answer.", "answer", "hard"),
    Scenario("comparison", "reasoning", "Compare two entities, methods, or concepts.",
             "Ask for a comparison where both sides are supported by the evidence.", "answer", "medium"),
    Scenario("numerical", "factual", "Retrieve a number, date, threshold, or quantity accurately.",
             "Ask for a numerical value with its relevant unit, condition, or context.", "answer", "medium"),
    Scenario("partial_evidence", "answerability", "Test whether the system communicates incomplete evidence.",
             "Ask a question for which the evidence supports only part of the requested answer.", "partial", "hard"),
    Scenario("unanswerable", "answerability", "Test abstention when evidence is absent.",
             "Create a plausible question about the document's topic whose answer is not present in the supplied evidence. Mark it abstain; never invent an answer.", "abstain", "medium"),
    Scenario("distractor", "retrieval", "Distinguish relevant evidence from similar but irrelevant content.",
             "Ask about a specific entity or concept while the evidence also contains a plausible distractor.", "answer", "hard"),
    Scenario("ambiguous", "robustness", "Test ambiguity handling.",
             "Ask a question that is genuinely ambiguous from the evidence and should request clarification or state the ambiguity.", "partial", "hard"),
)

SCENARIO_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}

def get_scenarios(scenario_ids: list[str] | None = None) -> list[Scenario]:
    """Return all scenarios or a validated subset in library order."""
    if scenario_ids is None:
        return list(SCENARIOS)
    unknown = set(scenario_ids) - SCENARIO_BY_ID.keys()
    if unknown:
        raise ValueError(f"Unknown scenario IDs: {sorted(unknown)}")
    selected = set(scenario_ids)
    return [scenario for scenario in SCENARIOS if scenario.scenario_id in selected]
