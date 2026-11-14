# Layer B and Layer C benchmark workflow

## Layer B: PDF-Q&A scenarios

The scenario library already defines the 11 PDF-Q&A scenario IDs:
`table_cell_lookup,table_comparison,definition_and_acronym,procedure_sequence,condition_exception,section_scope,cross_reference,entity_attribute_bundle,date_version_scope,citation_page_grounding,list_and_enumeration`.

Generate up to four candidates per scenario from the same corpus snapshot used by Layer A:

```powershell
python -m evaluation.scripts.generate_benchmark `
  --input evaluation/results/corpus_28_chunks.jsonl `
  --output evaluation/results/benchmark_layer_b_candidates.jsonl `
  --chunks-per-batch 2 `
  --max-candidates-per-batch 11 `
  --per-scenario 4 `
  --scenario-ids table_cell_lookup,table_comparison,definition_and_acronym,procedure_sequence,condition_exception,section_scope,cross_reference,entity_attribute_bundle,date_version_scope,citation_page_grounding,list_and_enumeration
```

The generator already invokes deduplication, structural/evidence validation, and clustering for that individual run. The global merge below repeats deduplication, validation, and clustering across A+B+C, which is the important cross-layer pass.

**Important:** a scenario quota is a maximum, not a promise. If the 28-chunk snapshot has no actual table, explicit cross-reference, procedure, or page locator, the model should return no candidate for that scenario. Do not manufacture PDF structure that is absent from the extracted corpus. Inspect the generated scenario counts and rejected file.

## Layer C: production-derived queries

Layer C must come from actual user queries and observed failures, not questions invented by the benchmark generator.

The current application `QueryLog` model stores `query_hash`, duration, cache status, and timestamp; it does not persist raw query text. Therefore the database audit table cannot reconstruct historical user queries. Export real queries from an authorized source if available (for example, explicitly retained application traces or a manually curated incident log).

Create `evaluation/results/benchmark_layer_c_candidates.jsonl`, one JSON object per line, using this shape:

```json
{"query":"Actual user query","scenario_id":"production_observed","category":"production","difficulty":"medium","expected_behavior":"answer","reference_answer":"Human-verified expected answer from the corpus","source_chunk_ids":["exact-corpus-chunk-id"],"qrels":{},"qrels_status":"pending","notes":"Observed failure: retrieved the wrong section; source: trace/incident reference"}
```

Only include a query as an evaluation candidate once its expected answer and evidence can be established. If it is merely a raw unresolved query, retain it in a separate intake file until a reviewer grounds it. Never label a synthetic probe as an observed production query.

## Global merge and curation

After A, B, and C candidate files exist:

```powershell
python -m evaluation.scripts.combine_benchmark_layers `
  --layer-a evaluation/results/benchmark_candidates.jsonl `
  --layer-b evaluation/results/benchmark_layer_b_candidates.jsonl `
  --layer-c evaluation/results/benchmark_layer_c_candidates.jsonl `
  --corpus evaluation/results/corpus_28_chunks.jsonl `
  --output evaluation/results/benchmark_combined_candidates.jsonl
```

For an interim A+B-only pass while collecting real Layer C data, add `--allow-missing-c`. Do not call that output the complete benchmark.

Outputs: combined JSONL, `.duplicates.jsonl`, `.rejected.jsonl`, and `.clusters.json`. All qrels are deliberately reset to empty/pending. Human review and qrels verification happen after global curation; freezing happens last.
