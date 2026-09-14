# Single-call model design implementation

Initial drafts use the existing bounded plan-and-code generator. A successful
first extraction requires one generation call; there is no draft decision,
feature-selection, summarization, staged-coding or merge call. Joint knowledge
assembly keeps architecture, precision, training, evaluation and runtime facts
with their conditions, restrictions, fallbacks and evidence references.

Hardware/code and lesson storage remain independent. The shared v2 contract
adds concise records, derived indexes, exact-runtime filtering, explicit
conditional family alternatives and versioned retrieval. Old detailed evidence,
lesson revisions, publication queues and v1 readers remain available. New
workspaces pin v2; resumed unpinned journals use v1. Later decision, review,
repair, execution and precision/preflight contracts remain in place.

There is no fixed knowledge character/token cap. Full-request sizing uses a
configured matching tokenizer and deployment window with completion space
reserved. Without that information diagnostics report sizing as unavailable.
Task requirements and required restrictions are never silently truncated.

## Files

- `agents/draft_agent.py`, `agents/coder/base_coder.py`,
  `agents/design_knowledge.py`: one-call drafting and shared prompt assembly.
- `knowledge/records.py`, `knowledge/runtime.py`, `knowledge/lessons.py`,
  `knowledge/migration.py`: concise contract, context accounting, version pins
  and resumable evidence-preserving backfill.
- Hardware, code, CUDA and lesson clients/stores/builders: concise retrieval,
  ingestion, publication and separate derived indexes. Agent context/history
  adapters use concise records while retaining diagnostic evidence.
- `schema/hardware_knowledge_graph.json` and its Cypher mirror: 37 concise
  feature summaries; source descriptions and examples remain unchanged.
- Config, focused tests, `docs/single_call_design_knowledge.md` and
  `benchmarks/audit_single_call_design.py`: configuration, migration usage,
  regression coverage and repeatable offline prompt accounting.

## Local validation

With `PATH="$PWD/.venv/bin:$PATH"` and
`PYTHONPATH="$PWD/nn-model-preflight-checker/src:$PWD"`:

```bash
.venv/bin/python -m pytest tests \
  localml_scheduler/tests/test_graph_db_validation.py \
  localml_scheduler/tests/test_backend_guidance.py \
  localml_scheduler/tests/test_cuda_docs_schema_roundtrip.py \
  localml_scheduler/tests/test_cuda_docs_gateway.py \
  localml_scheduler/tests/test_hardware_features.py \
  localml_scheduler/tests/test_feature_filter.py \
  --ignore=tests/integration -q
```

Result: **607 passed**. After the final software-filter, history deduplication and
single-call assertion changes, **40 focused checks passed** across design knowledge, backend guidance,
CUDA schema round trips and hardware features. Syntax checks and
`git diff --check` passed. Shell-based tests require the virtual environment on
PATH; the base Python lacks `humanize`.

Coverage includes exact generation count, restrictions/fallbacks, deduplication,
incompatible hardware/software, conditional and conflicting lessons, unsupported
numeric lesson claims, unavailable stores, oversized documents, migration
resume/idempotency, source revisions, immutable lesson publication, version pins,
legacy trace loading, precision, preflight, scheduler hooks and component gates.

## Migration rehearsal

[Source mapping and validation report](2026-09-13_design_knowledge_migration.json):
1,316 curated sources across graph, backend guidance and hardware feature seeds;
1,311 converted into 1,344 concise records; five pre-existing relationships
remain awaiting correction because an endpoint is missing. Every source is
accounted for. A backfill interrupted after 31 sources, its completed resume and
an idempotent repeat produced identical export hashes. The local ledger retains
original source versions; no live database was published or wiped.

Lesson migration/publication was exercised against temporary SQLite databases
and fake/in-memory vector stores, including frozen revision preservation and
idempotent derived-record backfill.

## Prompt overhead evidence

[Audit JSON](2026-09-13_single_call_design_audit.json),
[rebuilt prompt](2026-09-13_single_call_design_audit.prompt.json), and
[combined PNG](2026-09-13_single_call_design_audit.png).

Four captured archived coding requests total **192,210 characters**. The rebuilt
single request contains **29,523 characters**, a **84.64%** reduction across those
captured requests. The rebuilt request includes 33 selected records. The audit
uses the unchanged archived task, a fixed preview and current curated hardware
knowledge. It excludes any separately logged decision/selector calls. It does
not measure tokens, LLM latency, training correctness or model quality.

Concurrent integration edits were preserved. No live GPU experiment, scheduler
job, branch profile, admission policy or experiment configuration was changed by
this refactor. Live database deployment and GPU comparisons remain separate.
