"""Canonical scheduler API wording shared by generation and repair."""

SCHEDULER_SAFE_POINT_INSTRUCTION = (
    "Use the installed API exactly: context.control_hook.safe_point("
    "SafePointType.STEP, epoch=epoch, global_step=global_step, "
    "steps_per_epoch=steps_per_epoch, state_factory=state_factory). "
    "Use the same keyword arguments for BEFORE_TRAIN and EPOCH. "
    "There is no payload= parameter and no second positional argument. Import script_scheduler_context from "
    "localml_scheduler.execution.script_context and SafePointType from "
    "localml_scheduler.domain. Guard hooks with `if context is not None`; "
    "do not catch and suppress hook/checkpoint exceptions. Define state_factory "
    "to capture actual model, optimizer, scaler, LR scheduler, RNG, global_step "
    "and exact step_in_epoch/sampler_state/data_state. Restore those same state "
    "objects and loader position before continuing. Use direct hook calls and "
    "TrainingDiagnostics construction so the static admission gate can verify "
    "the contract; do not add unreachable token-matching calls."
)
