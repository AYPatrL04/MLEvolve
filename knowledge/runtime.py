"""Run-local knowledge version pinning and optional exact context accounting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from knowledge.records import render_records


def version_for(agent: Any) -> str:
    return str(getattr(agent, "design_knowledge_version", getattr(getattr(agent, "acfg", None), "design_knowledge_version", "v1")))


def pin_version(workspace: Path, requested: str, *, resuming: bool) -> str:
    path = Path(workspace) / "design_knowledge_version.json"
    if path.exists():
        version = str(json.loads(path.read_text())["version"])
    else:
        version = "v1" if resuming else requested
        if version not in {"v1", "v2"}:
            raise ValueError("design_knowledge_version must be v1 or v2")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x") as stream:
                json.dump({"version": version}, stream)
        except FileExistsError:
            return pin_version(workspace, requested, resuming=resuming)
    if version not in {"v1", "v2"}:
        raise ValueError("Unsupported pinned design knowledge version")
    return version


def fit_prompt(agent: Any, build_prompt: Callable[[str], Any], records: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Fit only to known deployment capacity; never invent a token estimate.

    A caller may supply a tokenizer adapter for a hosted model. Otherwise a
    configured local tokenizer is loaded without network access and cached.
    """
    config = agent.acfg.code
    window = getattr(config, "context_window_tokens", None)
    completion = getattr(config, "completion_tokens", None)
    if completion is None and getattr(config, "provider", "") == "vllm":
        completion = getattr(getattr(getattr(agent, "cfg", None), "vllm_client", None), "default_completion_tokens", None)
    counter = getattr(agent, "design_prompt_token_counter", None)
    tokenizer_path = getattr(config, "tokenizer_path", None)
    diagnostic: dict[str, Any] = {"sizing": "unavailable", "knowledge_version": version_for(agent), "dropped_record_ids": []}
    if window and completion and counter is None and tokenizer_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = getattr(agent, "_design_tokenizer", None)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                agent._design_tokenizer = tokenizer

            def counter(prompt):
                from llm.openai import _prompt_to_messages

                messages = _prompt_to_messages(prompt, model=config.model)
                return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        except (OSError, ValueError, ImportError) as exc:
            diagnostic["reason"] = type(exc).__name__
    selected = list(records)
    while True:
        prompt = build_prompt(render_records(selected))
        if not window or not completion or counter is None:
            return prompt, selected, diagnostic
        try:
            count = int(counter(prompt))
        except (ValueError, TypeError, RuntimeError) as exc:
            diagnostic["reason"] = type(exc).__name__
            return prompt, selected, diagnostic
        diagnostic.update(sizing="exact", prompt_tokens=count, context_window_tokens=int(window), completion_tokens=int(completion))
        if count + int(completion) <= int(window):
            return prompt, selected, diagnostic
        removable = next((i for i in range(len(selected) - 1, -1, -1) if selected[i].get("strength") != "hard" and not selected[i].get("restrictions")), None)
        if removable is None:
            raise ValueError("Draft task and required knowledge exceed the configured model context window; task requirements were not truncated")
        diagnostic["dropped_record_ids"].append(selected.pop(removable)["record_id"])
