# Single-call drafts and design knowledge v2

Initial drafts now choose the model, precision, training and evaluation setup in
one call that returns a brief plan and one complete script. Drafts do not call
the pipeline-decision agent, hardware feature selector, staged coder or merger.
Extraction retries, review, repair, precision checks, preflight and scheduler
contracts remain in place. `pipeline_decision_enabled` still controls decisions
for later actions. Stage labels remain useful for review and old traces.

## Concise records and storage

`knowledge.records` defines `design-knowledge-v2`. Each record carries its ID,
revision, category, semantic topics, complete summary, applicability, conditions,
restrictions, fallbacks, source/evidence references, confidence and verification
status. Topics are architecture, precision, training, evaluation and runtime.
An optimization-mode restriction is distinct from the precision format itself.

Hardware and lessons remain independently enabled domains:

- Neo4j retains hardware facts and relationships. Relationship property
  `design_records_v2_json` contains concise knowledge for that exact pairing.
- Code-knowledge Qdrant collections retain original evidence. Separate
  `<existing_collection>_design_v2` indexes contain concise records and filter
  metadata. Updating a source replaces its derived summaries, including an empty
  result when a replacement source requires summarization.
- Lesson SQLite retains observations, immutable revisions, conflicts and the
  publication outbox. `design_knowledge_v2` is a derived table keyed by profile,
  revision and record ID. A separate lesson Qdrant index is rebuilt from the same
  frozen publication; SQLite's active revision remains authoritative.
- Search-history records retain their original plans and metadata alongside
  concise `design_records_v2`. Legacy history is projected to complete first
  proposal sentences and recorded metrics when loaded by a v2 run.

All 37 curated hardware features have explicit `design_summary` fields. Source
descriptions and examples remain intact. Unprocessed documents, logs and code
blocks are excluded from ordinary knowledge retrieval until summarized.
Summarization happens during ingestion or through the existing lesson worker,
never through an extra LLM call on the draft path. The lesson worker validates
evidence references and numeric claims and does not cut summaries mid-sentence.

Retrieval filters hardware, runtime, backend, workload, role and precision before
using advice. Duplicate claims with identical applicability and restrictions
share one prompt entry, retaining all evidence references. Conflicting findings
remain distinct. Before family/shape selection, lessons are explicitly
conditional on their recorded family and workload shape. They are not numeric
defaults for an unselected model.

## Configuration and context accounting

```yaml
agent:
  design_knowledge_version: v2
  code:
    context_window_tokens: null
    completion_tokens: null
    tokenizer_path: null
```

There is no fixed knowledge character/token cap in v2. Legacy
`hardware_context_mode` and prompt-character limits remain accepted for v1.
Existing retrieval-count controls still select relevant records.

For exact context accounting, configure the actual deployment window and a
matching local tokenizer. `completion_tokens` is both the output reservation
and the maximum passed to generation. A vLLM deployment can instead use its
existing `vllm_client.default_completion_tokens` reservation. Tokenizers are
loaded locally without downloads. An embedded caller can supply
`design_prompt_token_counter(prompt)` for its provider's exact message format.

When sizing is available, only optional lower-priority records are removed to
fit the request. Required facts, restrictions and task requirements are never
silently truncated. An irreducibly oversized request fails before generation.
Unavailable sizing is reported in logs and node diagnostics; prompt safety is
not claimed from a characters-to-tokens estimate.

New workspaces pin the requested version in `design_knowledge_version.json`.
An existing journal without a pin uses v1. A pin wins over later configuration
changes. Preserve the pin when copying or resuming a workspace. Rollback for
new runs uses `design_knowledge_version: v1`; existing source stores and old
journals remain readable. The single-call draft route applies in either version.

Standalone hardware/code knowledge can configure
`hardware_knowledge.settings.code_knowledge` with the existing Qdrant settings;
it defaults disabled. When attached, the scheduler's existing code-knowledge
store is reused. No scheduler job/profile schema or admission policy changes.

## Migration

Inventory first; this reads source evidence and writes nothing:

```bash
.venv/bin/python -m knowledge.migration --domain hardware \
  --source schema/hardware_knowledge_graph.json \
  --output records/design-knowledge-migration/hardware
```

Add `--apply` to create the local migration ledger and validated export.
`--max-sources N` permits bounded backfills; repeating the command resumes from
committed source hashes. Every source is converted, deduplicated or marked
`awaiting_correction`. Original source versions remain in the ledger. Only a
complete pass replaces the exported manifest and records.

For lessons, use `--domain lesson --source /path/to/lesson_profiles.sqlite3`.
The source database must contain the lesson-profile tables; scheduler databases
are rejected. `--apply --publish` backfills only the derived lesson table.
Adding `--settings /path/to/lesson-settings.yaml` also publishes its vector
index. Frozen revisions and outbox payloads are not rewritten.

Hardware publication requires `--apply --publish --settings
/path/to/hardware-settings.yaml`, where the YAML is the standalone hardware
settings mapping, not the entire run config. Graph JSON updates the graph's
derived relationship records; code-document/recipe YAML uses code ingestion
and its separate concise index. No destructive recreate operation is used.

Review the complete inventory, source mappings, awaiting-correction entries and
export hashes before enabling v2 for new runs. The current graph inventory
finds five pre-existing relationships with missing endpoints; these remain
preserved and excluded from concise retrieval. No live database publication is
part of local implementation validation.

## Offline validation

`tests/test_design_knowledge.py` covers single-call drafting, conditions,
deduplication, incompatible runtimes, migration resume, context accounting,
Neo4j payloads and an in-memory Qdrant round trip. Lesson tests cover publication
immutability, idempotent backfill and conditional family retrieval.

```bash
PYTHONPATH="$PWD/nn-model-preflight-checker/src:$PWD" .venv/bin/python \
  benchmarks/audit_single_call_design.py \
  --pipeline-db runs/20260828_154842_petfinder_sonnet5090_50nodes/logs/pipeline.sqlite3 \
  --output-prefix records/2026-09-13_single_call_design_audit
```

This captures the rebuilt prompt and a combined call-sequence/prompt-size PNG.
It uses the unchanged archived task, a fixed preview fixture and current curated
hardware evidence. The archive includes its staged outputs. It measures prompt
overhead only, not model quality, training correctness or elapsed LLM time.
