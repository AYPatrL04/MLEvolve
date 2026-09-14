# Seed-42 Prompt Evidence Set

Source run: `20260914_071234_full_a10_seed42`

These are immutable copies of rendered `*.draft.prompt.md` files from the full
seed-42 A10 experiment. Filenames identify the selected outcome. Full prompt
contents are retained because section ordering, repeated instructions, stale
candidate history, and HWDB filtering must be analyzed together.

| File | Node | Outcome | Why selected |
| --- | --- | --- | --- |
| `01_success_best_f1_077105_node_7ac9abc3.prompt.md` | `7ac9abc389d14140bb890843ea478383` | Verified valid, F1 `0.7710526315789473` | Best successful node |
| `02_success_first_valid_f1_073413_node_d3846957.prompt.md` | `d3846957d72446c4879ae453c96cbcec` | Verified valid, F1 `0.7341269841269841` | First successful GPU result |
| `03_success_second_valid_f1_075726_node_c300fdde.prompt.md` | `c300fdde6f7444a6814c7d67685cecea` | Verified valid, F1 `0.7572559366754618` | Intermediate successful node |
| `04_runtime_error_device_mismatch_node_0bb72b17.prompt.md` | `0bb72b1775224031aa7d5bcdd3a309b0` | Buggy GPU execution | `pos_weight` stayed on CPU while logits were on CUDA |
| `05_review_rejected_precision_plus_scheduler_node_a8a7de79.prompt.md` | `a8a7de79269b4da1a353c8998cae9023` | Pre-execution review rejection | Unresolved autocast and scheduler-contract failures |
| `06_review_rejected_precision_only_node_0fecae71.prompt.md` | `0fecae711ab748118f7cc9d5f5dc3efc` | Pre-execution review rejection | Precision-policy issues without scheduler failure |
| `07_preflight_failed_after_repair_node_8a83b28e.prompt.md` | `8a83b28ee2394aa196618ec2043dce31` | Preflight admission rejection | CPU Preflight remained failed after repairs |
| `08_review_rejected_truncated_incomplete_script_node_5c4ce497.prompt.md` | `5c4ce49795e64d9eaf9de98853ee472c` | Pre-execution review rejection | Truncated/incomplete generated script |
| `09_review_rejected_scheduler_only_node_953b5a77.prompt.md` | `953b5a77686e4497a6e299c8365e9eaf` | Pre-execution review rejection | Scheduler safe-point contract failures only |
| `10_review_rejected_multistage_contract_node_816ea2ef.prompt.md` | `816ea2efe2ca48bd8e72e974acee826d` | Pre-execution review rejection | Metric, precision, training, and progress-reporting failures |

## Analysis Notes

- Prompt sizes range from approximately 264 KB to 332 KB, before accounting for
  the model's response or subsequent repair prompts.
- The set is intended to compare successful and unsuccessful prompt assembly,
  not just final candidate code.
- `04` is the only selected candidate that passed review and Preflight, was
  admitted, and then failed during GPU execution.
- `05` through `10` never reached GPU execution.
- Review rejection is not equivalent to a confirmed software bug. The retained
  failure evidence must be adjudicated separately.
