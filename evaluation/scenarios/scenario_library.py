"""Reusable RAG evaluation scenarios, independent of any specific corpus.

The library contains general RAG scenarios plus document/PDF-QA-specific scenarios.
These are scenario specifications, not fixed question/answer pairs: generated questions
and relevance labels must still be grounded in each application's corpus.
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


# General scenarios reusable across RAG applications.
GENERAL_SCENARIOS: tuple[Scenario, ...] = (
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


# Document/PDF-QA scenarios: reusable across document-grounded RAG products.
PDF_QA_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("table_cell_lookup", "document_structure",
             "Retrieve a value from a table while preserving its row/column meaning.",
             "Only use evidence that visibly contains a table or table-like row/column structure. Ask for a specific cell or the value associated with a named row and column. The answer must preserve the row label, column label, value, and unit where present; do not infer missing cells.",
             "answer", "medium"),
    Scenario("table_comparison", "document_structure",
             "Compare values across rows or columns in a table.",
             "Use a table with at least two comparable entries. Ask which entry is larger/smaller or how two entries differ. Include the compared labels and exact values in the answer; calculate only when the required values are explicit.",
             "answer", "medium"),
    Scenario("definition_and_acronym", "document_understanding",
             "Retrieve a formal definition, abbreviation expansion, or domain-specific term.",
             "Ask what a named term or acronym means according to the document. The answer should give the document's definition or exact expansion and retain any stated qualification; do not supply outside knowledge.",
             "answer", "easy"),
    Scenario("procedure_sequence", "document_structure",
             "Recover ordered steps from instructions or a process description.",
             "Only use evidence that describes a procedure, workflow, or ordered process. Ask for the steps or what must happen before/after a named step. Answer in the source order and do not invent omitted steps.",
             "answer", "medium"),
    Scenario("condition_exception", "policy_and_constraints",
             "Retrieve a rule together with its conditions, exclusions, or exceptions.",
             "Ask when a stated rule applies, does not apply, or what exception is specified. The answer must include the relevant condition and exception rather than stating the rule without its qualifiers.",
             "answer", "hard"),
    Scenario("section_scope", "document_navigation",
             "Identify which section, chapter, or scope a statement belongs to.",
             "Ask which section/topic/scope contains a stated requirement or which items fall under a named section. Answer only from explicit headings, labels, or contextual evidence supplied; do not invent page or section numbers.",
             "answer", "medium"),
    Scenario("cross_reference", "document_navigation",
             "Resolve a reference to another section, appendix, figure, or document part.",
             "Only use evidence containing an explicit cross-reference such as 'see section', 'refer to appendix', or a figure/table reference. Ask what the reference points to or what information it directs the reader to. Preserve the exact referenced identifier.",
             "answer", "hard"),
    Scenario("entity_attribute_bundle", "structured_extraction",
             "Extract several attributes belonging to one named entity or item.",
             "Ask for multiple explicitly stated attributes of one entity, product, role, component, or record. Return a compact structured list of attribute-value pairs and do not mix attributes from similarly named entities.",
             "answer", "medium"),
    Scenario("date_version_scope", "temporal_context",
             "Distinguish dates, effective periods, revisions, or version-specific statements.",
             "Ask which date/version/period applies to a stated requirement or fact. The answer must distinguish publication, revision, and effective dates when the evidence does; never assume the newest-looking date is the effective date.",
             "answer", "hard"),
    Scenario("citation_page_grounding", "evidence_grounding",
             "Test whether an answer can be tied to a precise source location.",
             "Ask for a fact that is explicitly supported by the supplied chunk and require the answer to identify the supporting heading, table, figure, or page label only when that locator is present in the evidence. Never fabricate page numbers or citations.",
             "answer", "medium"),
    Scenario("list_and_enumeration", "structured_extraction",
             "Retrieve a complete list of items within a stated scope.",
             "Ask for all explicitly listed items in a named category or scope. The answer must preserve the source's scope and avoid adding related items that are not in the evidence.",
             "answer", "medium"),
)


SCENARIOS: tuple[Scenario, ...] = GENERAL_SCENARIOS + PDF_QA_SCENARIOS
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
