"""Architecture-aware training precision policy shared across MLEvolve.

The policy intentionally distinguishes native, supported training formats from
capability-only integer formats and from storage/inference-only formats.  A
format is eligible for recommendation only when the repository has an
end-to-end training policy for it; low-level CUDA type availability alone is
not sufficient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping


PRECISION_MODE_NORMAL = "normal"
PRECISION_MODE_CONSERVATIVE = "conservative"
PRECISION_MODE_AGGRESSIVE = "aggressive"
PRECISION_OPTIMIZATION_MODES = frozenset(
    {PRECISION_MODE_CONSERVATIVE, PRECISION_MODE_NORMAL, PRECISION_MODE_AGGRESSIVE}
)

CONSERVATIVE_PRECISION_INSTRUCTION = (
    "Conservative precision is mandatory for candidate models: use float32 parameters, floating inputs, "
    "optimizer state, training, validation, and inference. Keep integer indices/labels unchanged. "
    "Disable AMP/autocast and GradScaler; do not use FP16, BF16, FP8, FP4, FP64, quantized models, "
    "or lower-precision adapters. Disable TF32 for matmul and cuDNN convolution using the installed "
    "PyTorch version's supported controls; use IEEE FP32, including float32 matmul precision 'highest'. "
    "This rule overrides inherited code and hardware recommendations."
)

NORMAL_PRECISION_INSTRUCTION = (
    "Normal mode permits FP32 and selective FP16 AMP only, not BF16, TF32, quantization or TE recipes. "
    "FP16 is an option, not a requirement or a whole-pipeline dtype. Keep model parameters and floating "
    "inputs and floating regression targets in FP32; preserve integer indices and labels. Use CUDA autocast only around eligible forward "
    "operations and compatible loss computation. Run backward and optimizer updates outside autocast, "
    "with GradScaler for FP16 and unscale before gradient clipping. For fragile losses/reductions, disable "
    "autocast locally and explicitly cast inputs to FP32. Keep metric accumulation and prediction export "
    "in FP32; validation/test forwards may share a validated autocast path. Do not call model.half() or "
    "cast the entire batch to half. CPU preflight uses FP32 with AMP/scaling disabled. Disable TF32 for "
    "matmul and convolution. Compare finite loss/gradients, task quality and elapsed time against FP32; "
    "retain FP32 for unsupported, unstable or slower operations. Record the actual regions and fallback, "
    "not an unmeasured speedup."
)


def precision_mode_instruction(mode: str) -> str:
    if mode == PRECISION_MODE_CONSERVATIVE:
        return CONSERVATIVE_PRECISION_INSTRUCTION
    if mode == PRECISION_MODE_NORMAL:
        return NORMAL_PRECISION_INSTRUCTION + " " + MIXED_PRECISION_INSTRUCTION
    return "Select only hardware/mode-allowed precision policies and preserve an FP32 fallback."


def precision_advice_allowed(text: str, policy: PrecisionPolicy) -> bool:
    """Filter positive advice across every stage; this is not a code validator."""
    text = str(text or "").lower().replace("-", "_")
    if policy.mode == PRECISION_MODE_AGGRESSIVE:
        if "fp6" in text or ("fp4" in text and "nvfp4" not in text):
            return False
        return all(token not in text or policy.allows(required) for token, required in {
            "fp8": "fp8_te", "mxfp8": "mxfp8_te", "nvfp4": "nvfp4_te"
        }.items())
    requirements = {
        r"\b(?:fp16|float16|half|amp|autocast|gradscaler)\b": "fp16_amp",
        r"\b(?:bf16|bfloat16)\b": "bf16_amp",
        r"\btf32\b|set_float32_matmul_precision\(['\"](?:high|medium)['\"]\)": "tf32",
        r"\b(?:fp8\w*|float8\w*)\b": "fp8_te",
        r"\bmxfp8\b": "mxfp8_te",
        r"\bnvfp4\b": "nvfp4_te",
    }
    if re.search(r"\b(?:fp64|float64|fp6|fp4|mxfp4)\b", text):
        return False
    if policy.mode != PRECISION_MODE_AGGRESSIVE and re.search(r"\b(?:quantiz\w*|qlora|int[48])\b", text):
        return False
    return all(not re.search(pattern, text) or policy.allows(required) for pattern, required in requirements.items())



MIXED_PRECISION_INSTRUCTION = (
    "Hardware precision recommendations are conditional execution choices; task quality takes priority over epoch speed. "
    "Follow the configured mode and GPU allowlist. For mode-allowed AMP, keep model parameters and optimizer state "
    "in FP32; do not use model.half(), model.bfloat16(), lower-precision model loading, or whole-model dtype casts. "
    "Use autocast for eligible forward/loss operations, GradScaler for FP16, and FP32 for sensitive reductions "
    "and exported predictions/metrics. Preserve model family, loss, features, input resolution, effective batch, "
    "and training/evaluation budget when optimizing hardware alone. Keep an explicit FP32 execution path. "
    "On non-finite loss/gradients, restore the last finite checkpoint including optimizer, scaler, RNG and data "
    "position before an FP32 retry; never step on invalid gradients or reset the consumed training budget. "
    "A faster epoch or finite smoke test does not establish accuracy preservation. Compare against a matching "
    "FP32 reference with the same architecture, data split, initialization, preprocessing and training budget; "
    "use only an explicitly supplied quality tolerance, and report missing evidence as unverified. "
    "Before enabling lower precision, call utils.precision_quality.select_validated_precision(requested, "
    "protocol=actual_protocol, comparison=supplied_comparison). Without supplied matching measurements, "
    "call it with no comparison and use the returned FP32 choice; do not invent measurements or launch extra "
    "comparison training outside the task budget. Derive autocast/scaler enablement from the returned 'precision', "
    "and store the returned decision as settings['precision_quality'] in TrainingDiagnostics."
)
_BASE_POLICIES = ("fp32", "disabled")
_POLICY_TO_FEATURES: dict[str, tuple[str, ...]] = {
    "fp16_amp": ("amp", "fp16"),
    "bf16_amp": ("amp", "bf16"),
    "tf32": ("tf32",),
    "fp8_te": ("fp8", "fp8_e4m3", "fp8_e5m2"),
    "mxfp8_te": ("mxfp8",),
    "nvfp4_te": ("nvfp4",),
}
_INTEGER_PREFIXES = ("int", "uint", "sint", "u4", "s4", "b1")
_KNOWN_HIDDEN_PRECISION_FEATURES = frozenset(
    {"fp4", "fp64", "fp6", "mxfp4"}
)


def normalize_precision_optimization_mode(value: Any) -> str:
    mode = str(value or PRECISION_MODE_NORMAL).strip().lower().replace("-", "_")
    if mode not in PRECISION_OPTIMIZATION_MODES:
        expected = ", ".join(sorted(PRECISION_OPTIMIZATION_MODES))
        raise ValueError(
            f"Unsupported agent.precision_optimization_mode: {value}. "
            f"Expected one of: {expected}"
        )
    return mode


def normalize_precision_policy_name(value: Any) -> str | None:
    name = str(value or "").strip().lower().replace("-", "_").replace("torch.", "")
    aliases = {
        "fp16": "fp16_amp",
        "float16": "fp16_amp",
        "half": "fp16_amp",
        "fp16_amp": "fp16_amp",
        "bf16": "bf16_amp",
        "bfloat16": "bf16_amp",
        "bf16_amp": "bf16_amp",
        "fp32": "fp32",
        "float32": "fp32",
        "tf32": "tf32",
        "fp8": "fp8_te",
        "float8": "fp8_te",
        "te_fp8": "fp8_te",
        "fp8_te": "fp8_te",
        "mxfp8": "mxfp8_te",
        "mx_fp8": "mxfp8_te",
        "te_mxfp8": "mxfp8_te",
        "mxfp8_te": "mxfp8_te",
        "nvfp4": "nvfp4_te",
        "te_nvfp4": "nvfp4_te",
        "nvfp4_te": "nvfp4_te",
        "disabled": "disabled",
        "none": "disabled",
    }
    return aliases.get(name)


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    mode: str
    architecture: str
    compute_capability: str | None
    allowed_policies: tuple[str, ...]
    permitted_features: tuple[str, ...]
    recommended_features: tuple[str, ...]
    integer_capability_indicators: tuple[str, ...]
    hidden_features: tuple[str, ...]
    requires_transformer_engine: tuple[str, ...]
    fallback_policy: str = "fp32"
    preferred_policy: str | None = None

    def allows(self, value: Any) -> bool:
        normalized = normalize_precision_policy_name(value)
        return bool(normalized and normalized in self.allowed_policies)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_precision_policy(
    hardware: Mapping[str, Any] | None = None,
    *,
    mode: Any = PRECISION_MODE_NORMAL,
    architecture: Any = None,
    compute_capability: Any = None,
    datatypes: Any = None,
) -> PrecisionPolicy:
    """Return the training precision allowlist for one hardware target."""
    normalized_mode = normalize_precision_optimization_mode(mode)
    source = dict(hardware or {})
    nested = source.get("hardware")
    if isinstance(nested, Mapping):
        source = {**source, **dict(nested)}

    architecture_value = architecture
    if architecture_value is None:
        architecture_value = source.get("architecture") or source.get("architectures")
    capability_value = compute_capability
    if capability_value is None:
        capability_value = source.get("compute_capability") or source.get("compute_capabilities")
    datatype_values = datatypes if datatypes is not None else source.get("datatypes")

    capability = _first_text(capability_value)
    normalized_architecture = _normalize_architecture(architecture_value, capability)
    allowed = list(_BASE_POLICIES)
    if normalized_mode == PRECISION_MODE_NORMAL:
        if normalized_architecture in {"volta", "turing", "ampere", "ada_lovelace", "hopper", "blackwell"}:
            allowed.append("fp16_amp")
    elif normalized_mode == PRECISION_MODE_AGGRESSIVE and normalized_architecture in {"volta", "turing"}:
        allowed.append("fp16_amp")
    elif normalized_mode == PRECISION_MODE_AGGRESSIVE and normalized_architecture in {"ampere", "ada_lovelace", "hopper", "blackwell"}:
        allowed.extend(("tf32", "bf16_amp", "fp16_amp"))

    if normalized_mode == PRECISION_MODE_AGGRESSIVE:
        if normalized_architecture in {"ada_lovelace", "hopper", "blackwell"}:
            allowed.append("fp8_te")
        if normalized_architecture == "blackwell":
            allowed.extend(("mxfp8_te", "nvfp4_te"))

    permitted_features: list[str] = []
    for policy in allowed:
        for feature in _POLICY_TO_FEATURES.get(policy, ()):
            if feature not in permitted_features:
                permitted_features.append(feature)

    # Aggressive mode expands the deterministic allowlist, but these lower
    # precision recipes remain opt-in experiments rather than default advice.
    recommended_policies = [
        policy
        for policy in allowed
        if policy not in {"fp8_te", "mxfp8_te", "nvfp4_te"}
    ]
    recommended_features: list[str] = []
    for policy in recommended_policies:
        for feature in _POLICY_TO_FEATURES.get(policy, ()):
            if feature not in recommended_features:
                recommended_features.append(feature)

    integer_indicators = tuple(
        value
        for value in _normalize_string_list(datatype_values)
        if value.startswith(_INTEGER_PREFIXES)
    )
    visible = set(permitted_features) | set(integer_indicators)
    known_datatypes = set(_normalize_string_list(datatype_values))
    hidden = sorted((known_datatypes | set(_KNOWN_HIDDEN_PRECISION_FEATURES)) - visible)

    return PrecisionPolicy(
        mode=normalized_mode,
        architecture=normalized_architecture,
        compute_capability=capability,
        allowed_policies=tuple(dict.fromkeys(allowed)),
        permitted_features=tuple(permitted_features),
        recommended_features=tuple(recommended_features),
        integer_capability_indicators=integer_indicators,
        hidden_features=tuple(hidden),
        requires_transformer_engine=tuple(
            policy for policy in allowed if policy in {"fp8_te", "mxfp8_te", "nvfp4_te"}
        ),
        preferred_policy=(
            "fp32" if normalized_mode == PRECISION_MODE_CONSERVATIVE
            else "bf16_amp" if normalized_mode == PRECISION_MODE_AGGRESSIVE and normalized_architecture == "ampere" else None
        ),
    )


def precision_feature_visibility(feature_id: Any, policy: PrecisionPolicy) -> str:
    """Classify a precision feature for datatype-optimization prompts."""
    feature = str(feature_id or "").strip().lower().replace("-", "_")
    if policy.mode == PRECISION_MODE_CONSERVATIVE and feature not in policy.permitted_features:
        return "hidden"
    if feature in policy.recommended_features:
        return "recommendation"
    if feature in policy.permitted_features:
        return "permitted"
    if feature.startswith("fp8_") and "fp8_te" in policy.allowed_policies:
        return "permitted"
    if feature in policy.integer_capability_indicators or feature.startswith(_INTEGER_PREFIXES):
        return "integer_indicator"
    return "hidden"


def _normalize_architecture(value: Any, capability: str | None) -> str:
    values = _normalize_string_list(value)
    joined = " ".join(values).lower().replace("-", "_").replace(" ", "_")
    for token, normalized in (
        ("blackwell", "blackwell"),
        ("hopper", "hopper"),
        ("ada", "ada_lovelace"),
        ("ampere", "ampere"),
        ("turing", "turing"),
        ("volta", "volta"),
    ):
        if token in joined:
            return normalized

    parsed = _parse_compute_capability(capability)
    if parsed is None:
        return "unknown"
    major, minor = parsed
    if major >= 10:
        return "blackwell"
    if major == 9:
        return "hopper"
    if major == 8 and minor == 9:
        return "ada_lovelace"
    if major == 8:
        return "ampere"
    if major == 7 and minor >= 5:
        return "turing"
    if major == 7:
        return "volta"
    return "unknown"


def _parse_compute_capability(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    text = str(value).strip().lower().removeprefix("sm_")
    try:
        if "." in text:
            major, minor = text.split(".", 1)
            return int(major), int("".join(ch for ch in minor if ch.isdigit()) or 0)
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 2:
            return int(digits[:-1]), int(digits[-1])
    except ValueError:
        return None
    return None


def _first_text(value: Any) -> str | None:
    values = _normalize_string_list(value)
    return values[0] if values else None


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item).strip().lower().replace("-", "_") for item in values if str(item).strip()]
