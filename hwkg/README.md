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
context hash, and the observed GPU identity. Without `MLEVOLVE_HWKG_GRAPH_PATH`,
injection is a no-op. With treatment enabled, the container must expose exactly
one CUDA GPU whose name, compute capability, and approximate memory match the
pinned graph node uniquely; otherwise injection fails rather than using the
wrong card. Cards that share indistinguishable runtime properties in the graph
are intentionally rejected.

Verified supported AMP edges form the primary advice. Other edges linked to
the exact GPU remain visible at lower confidence: recommended but unverified
supported features are optional trials only when task and candidate evidence
support them; the rest are background IDs only. Draft/improve agents pass task
description and, for improve, parent code. Image decode needs image-task
evidence; Muon additionally needs explicit hidden-matrix layers and an
installed helper or native PyTorch Muon. Repeated AMP/FP16 text is merged.
No graph example implies that its software helper is installed. Advice is
optional; generated candidates remain subject to the upstream evaluator. The
baseline is the unmodified upstream commit, run with identical task, seed,
model, hardware, limits, and external scoring.
This runtime check verifies device identity, not the correctness or task
relevance of every pattern stored in the graph.
