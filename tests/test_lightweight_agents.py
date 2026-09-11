from types import SimpleNamespace

import jsonschema
import pytest

from context_cache.config import ContextCacheSettings
from deployments.run_hwdb_precision_matrix import agent_settings
from llm.common import FunctionSpec
from llm.openai import _normalize_gpt5_params, _stage_api_key, query
from llm.vllm import _VLLMHttpClient


def test_openai_profile_keeps_credentials_out_of_config():
    settings = agent_settings("openai-mini")
    assert settings["model"] == "gpt-5.4-mini"
    assert settings["provider"] == "openai"
    assert settings["api_key"] == ""


def test_small_qwen_requires_verified_endpoint(monkeypatch):
    monkeypatch.delenv("MLEVOLVE_QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("MLEVOLVE_QWEN_MODEL", raising=False)
    with pytest.raises(ValueError, match="verified"):
        agent_settings("qwen-small")
    monkeypatch.setenv("MLEVOLVE_QWEN_BASE_URL", "http://small:8000/v1")
    monkeypatch.setenv("MLEVOLVE_QWEN_MODEL", "qwen-small-test")
    assert agent_settings("qwen-small")["model"] == "qwen-small-test"


@pytest.mark.parametrize("content,valid", [('{"ok":true}', True), ('{"ok":"wrong"}', False)])
def test_schema_query_needs_no_tool_parser(content, valid):
    captured = {}
    def create(**params):
        captured.update(params)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    stage = SimpleNamespace(model="qwen-small-test", provider="vllm", base_url="http://local/v1", api_key="")
    cfg = SimpleNamespace(agent=SimpleNamespace(code=stage, feedback=stage),
                          context_cache=ContextCacheSettings(enabled=False),
                          vllm_client=SimpleNamespace(default_completion_tokens=128), exp_name="schema-test")
    spec = FunctionSpec("report", {"type":"object", "properties":{"ok":{"type":"boolean"}}, "required":["ok"]}, "Report")
    kwargs = dict(cfg=cfg, model=stage.model, func_spec=spec, _structured_output_mode="json_schema",
                  _client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    if valid:
        output, *_ = query("system", "user", **kwargs)
        assert output == {"ok": True}
    else:
        with pytest.raises(jsonschema.ValidationError):
            query("system", "user", **kwargs)
    assert "tools" not in captured and "tool_choice" not in captured
    assert captured["response_format"]["json_schema"]["schema"] == spec.json_schema


def test_vllm_thinking_is_a_template_option():
    client = _VLLMHttpClient(api_key="EMPTY", base_url="http://local/v1", timeout=1)
    try:
        _, body = client._request_parts({"extra_body":{"enable_thinking":False,"cache_salt":"test"}})
    finally:
        client.close()
    assert body["chat_template_kwargs"] == {"enable_thinking":False}
    assert "enable_thinking" not in body
    assert body["cache_salt"] == "test"


def test_gpt5_uses_completion_budget_and_server_sampling_defaults():
    params = dict(max_tokens=128, temperature=0.7, top_p=0.8, presence_penalty=0.1)
    _normalize_gpt5_params(params, "gpt-5.4-mini", SimpleNamespace(provider="openai"))
    assert params == {"max_completion_tokens":128}


def test_other_providers_are_unchanged():
    params = dict(max_tokens=128, temperature=0.7)
    _normalize_gpt5_params(params, "gpt-5.4-mini", SimpleNamespace(provider="openrouter"))
    assert params == dict(max_tokens=128, temperature=0.7)


def test_unauthed_local_endpoint_does_not_inherit_openai_key():
    assert _stage_api_key(SimpleNamespace(provider="vllm", api_key="")) == "EMPTY"
    assert _stage_api_key(SimpleNamespace(provider="openai", api_key="")) is None
    assert _stage_api_key(SimpleNamespace(provider="openai", api_key="explicit")) == "explicit"
