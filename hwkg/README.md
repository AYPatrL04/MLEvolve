# Isolated HWKG treatment

This branch starts at InternScience/MLEvolve commit
`9c5c8a3b23f0361708b59a401452dddc00f97189`. Its application change is
limited to this pinned knowledge graph and optional prompt injection in the
original draft and improve agents. It adds no optimizer implementation,
precision enforcement, review policy, scheduler, evaluator, or stopping rule.

The graph is a frozen snapshot with SHA-256
`0d422cfe81e2531980204e71b49fa9e5ffed9c2606a9a5569e7a43ebfc0078b4`.
Set `MLEVOLVE_HWKG_GRAPH_PATH`, `MLEVOLVE_HWKG_GRAPH_SHA256`, and
`MLEVOLVE_HWKG_HARDWARE_ID` to enable treatment. Optionally set
`MLEVOLVE_HWKG_AUDIT_DIR` to record selected graph nodes, character count,
and context hash. Without `MLEVOLVE_HWKG_GRAPH_PATH`, injection is a no-op.

The A10 selection intentionally includes only verified supported feature
edges. Advice is optional; generated candidates remain subject to the
upstream evaluator. The baseline is the unmodified upstream commit, run with
identical task, seed, model, hardware, limits, and external scoring.
