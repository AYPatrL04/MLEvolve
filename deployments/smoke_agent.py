"""Check generation and the real review schema before allocating a run budget."""

import argparse
import json
import os
from pathlib import Path

import jsonschema
from omegaconf import OmegaConf

from deployments.run_hwdb_precision_matrix import AGENT_PROFILES, agent_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-profile", choices=AGENT_PROFILES, required=True)
    args = parser.parse_args()
    stage = agent_settings(args.agent_profile)
    if stage["provider"] == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured; mount an authorized Secret, never put keys in result configs")
    if stage["provider"] == "deepseek" and not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not configured; mount an authorized Secret")
    from llm import generate, query
    from agents.code_review_agent import CODE_REVIEW_SPEC

    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "config.example.yaml")
    for role in ("code", "feedback"):
        cfg.agent[role].update(stage)
    cfg.context_cache.enabled = False
    cfg.vllm_client.structured_output_mode = "json_schema"
    cfg.vllm_client.default_completion_tokens = 2048
    cfg.exp_name = "lightweight-agent-smoke"
    text = generate("Reply with the single word READY.", cfg, max_tokens=1024, max_retries=1)
    if not text.strip():
        raise RuntimeError("Generation returned empty content")
    output = query(
        "Review this harmless Python snippet. Report no issues unless one is actually present. Return the requested structured review.",
        "def add(a, b):\n    return a + b\n", model=stage["model"], cfg=cfg,
        stage_name="feedback", func_spec=CODE_REVIEW_SPEC, max_tokens=2048,
    )
    jsonschema.Draft7Validator(CODE_REVIEW_SPEC.json_schema).validate(output)
    print(json.dumps({"model": stage["model"], "generation": "passed", "review_schema": "passed"}), flush=True)


if __name__ == "__main__":
    main()
