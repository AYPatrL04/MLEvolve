# Muon integration

Use `utils.muon_optimizer.MuonAdamW` as one PyTorch optimizer in both the
CandidateAdapter and the actual training loop. Explicitly select hidden matrix
names; keep embeddings, output heads, biases and normalization parameters on
AdamW. A disabled-Muon branch must construct AdamW over all trainable parameters.
TrainingDiagnostics checks complete, nonduplicated coverage before training.

```python
from utils.muon_optimizer import MuonAdamW

optimizer = MuonAdamW(
    model, hidden_names=["encoder.hidden.weight"],
    lr=1e-3, weight_decay=1e-4, momentum=0.95, ns_steps=5,
)
```

The named weight is illustrative: select actual hidden matrices in the candidate.
Starting LR/decay above reproduce our small-model comparison and are not universal
recommended optima. Tune Muon LR with `muon_lr` independently. The helper uses
Nesterov momentum, eps=1e-7 and match_rms_adamw scaling. A normal PyTorch scheduler
updates both groups; its state_dict includes both optimizer states for resume.

The equations and numerical defaults follow
[PyTorch 2.12 Muon](https://docs.pytorch.org/docs/2.12/generated/torch.optim.Muon.html).
Native Muon casts orthogonalization to BF16. Our helper instead keeps that step
FP32 and disables ambient autocast within it, without patching global PyTorch
functions. Keep TF32 disabled under conservative/normal policy. FP32 Muon can cost
more than the native implementation; published large-model FLOP savings are not
small-model runtime guarantees. A regression test compares updates against native
Muon with only the orthogonalization dtype changed.

Do not bypass training-contract checks: preserve scheduler safe points, complete
resume state, batch-quality declarations, early stopping and epoch metric output.
Malformed repair responses remain rejected; retries now receive the failure
reason and must patch the unchanged original source.

The graph's design_summary deliberately contains parameter values and the source
URL, so compression preserves them. Detailed notes/examples alone are insufficient
because they may not reach the rendered prompt. Tests cover the precision filter
and 3500-character budget.
