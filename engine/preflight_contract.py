"""Shared generated-code contract for the pinned preflight checker."""

PREFLIGHT_BATCH_CONTRACT = (
    "Batch builders must return (inputs, target) or a mapping with recognized "
    "`inputs` and `target` keys. For multiple positional model inputs, use "
    "{'inputs': (word_ids, char_ids, keyword_ids), 'target': targets}; "
    "training_step and validation_step must unpack those same inputs into the "
    "real model. Custom top-level keys alone (such as word_ids) are not recognized. "
    "Preserve any authoritative named fixture keys, adding an inputs tuple that "
    "references those same tensors when needed. Token/embedding indices remain "
    "torch.long and masks keep their required dtype; FP32 precision rules apply "
    "to floating-point computation, not integer indices. The checker owns "
    "zero_grad, backward and optimizer.step: training_step only computes and "
    "returns the connected scalar loss, without performing those updates itself."
)
