# Full A10 Seed-42 Task

User-authorized full MLEvolve search, isolated from the existing HWDB ablation.

- Cluster/context: `nautilus`; namespace: `ecepxie`.
- CPU gate: `mlevolve-full-a10-s42-20260914-prepare`.
- GPU job: `mlevolve-full-a10-s42-20260914-run` (submitted only after the CPU gate passes).
- Hardware: exactly one `NVIDIA-A10`; no other GPU product is eligible.
- Local source: `329b042c3ac66380ea661a4ebf3312a917041e77` plus the deployment scripts and their tests; all tracked working-tree files and both subrepositories were bundled locally, without fetching a new branch.
- Snapshot SHA256: `3d40142712415916dea1527a569f7644136cb901df3923feb066b9b57e626846`.
- Dataset: prepared public Disaster Tweets (`nlp-getting-started`), no held-out labels mounted.
- Agent: existing DeepSeek Flash API profile; credentials are referenced by Kubernetes Secret, never embedded in source or records.
- Seed: 42; 10 search nodes; 3 initial drafts; ordinary subsequent search/optimization enabled. No first-valid-node early stop or experiment-level three-hour cutoff. Each candidate retains the existing one-hour execution timeout; Kubernetes has a 78-hour safety deadline.
- Precision: normal, IEEE FP32 with optional selective FP16 AMP; no TF32/BF16.
- Preflight: enabled, internal errors fail closed, generated adapters required.
- Scheduler: enabled, incremental admission with live telemetry; CUDA process backend; no fixed concurrency cap; VRAM ceiling based on actual GPU memory, never above 31 GiB.
- Predictor: real A10 PerfSeer CPU student selected from the local model registry. Startup fails if the A10 model cannot load. Unsupported candidate conversions may explicitly fall back to Branch-Profile; inspect actual prediction sources before claiming predictor coverage.
- HWDB: enabled, using the current local schema for stage-specific prompts and attached scheduler profile evidence. The optional Neo4j search backend is disabled; no external HWDB server is required by this path.
- Persistent artifacts: PVC `yuze-li-vol`, subdirectory `aypatrl04-hwdb-deepseek-20260911/mlevolve-full-a10-s42-20260914`, mounted at `/experiment/mlevolve-full-a10-s42-20260914`.
- Hourly heartbeat: `mlevolve-a10-seed42-hourly-checks`; preserves the separate paused ablation monitor.

## Submission

The launcher refuses to restart existing Jobs or overwrite the source snapshot:

```bash
.venv/bin/python -m deployments.launch_full_a10 stage
.venv/bin/python -m deployments.launch_full_a10 prepare
.venv/bin/python -m deployments.launch_full_a10 run
```

The archive is retained locally but excluded from Git. `source.json` records the exact source identity. The same SHA256-checked archive and pinned container image are used in CPU preparation and GPU execution. CPU checks cover configuration, prompt filtering, Preflight, pipeline logging, training contracts and the real A10 predictor, followed by an agent API smoke test.

## Monitoring

Check `prepare.log`, `regression-tests.log`, `agent-smoke.log`, `predictor-health.json`, `runtime.log`, `status.json`, per-attempt Preflight/HWDB evidence, `runs/*/logs/pipeline.sqlite3` and the scheduler directory. A Running Pod is not proof of successful end-to-end execution. `gantt_metrics.png` is generated after execution, using recorded scheduler start/end times above node validation metrics. Preserve and report predictor fallback and Preflight rejection evidence rather than counting only successful nodes.
