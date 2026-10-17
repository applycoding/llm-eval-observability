"""OpenTelemetry attributes and the Langfuse OTLP destination builder."""

import base64

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_eval_obs.tracing import (
    add_langfuse_exporter,
    add_span_exporter,
    build_provider,
    langfuse_basic_headers,
    langfuse_settings_from_env,
    langfuse_traces_endpoint,
    record_llm_span,
    span_rows,
)


def test_span_records_tokens_cost_and_latency() -> None:
    provider = build_provider()
    exporter = InMemorySpanExporter()
    add_span_exporter(provider, exporter)
    record_llm_span(
        provider.get_tracer("test"),
        model="example-model",
        prompt_text="prompt",
        completion="answer",
        input_tokens=1000,
        output_tokens=500,
        latency_ms=12.5,
        prompt_id="support",
        prompt_version="2",
        case_id="refund-window",
        role="task",
        cost_usd=0.0105,
        cost_known=True,
    )
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = spans[0].attributes or {}
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.request.model"] == "example-model"
    assert attributes["gen_ai.usage.input_tokens"] == 1000
    assert attributes["gen_ai.usage.output_tokens"] == 500
    assert attributes["llm_eval_obs.latency_ms"] == 12.5
    assert attributes["llm_eval_obs.cost.usd"] == 0.0105
    assert attributes["llm_eval_obs.cost.known"] is True
    assert attributes["llm_eval_obs.role"] == "task"
    rows = span_rows(spans)
    assert rows[0]["case"] == "refund-window"
    assert rows[0]["status"] == "UNSET"


def test_error_span_sets_status() -> None:
    provider = build_provider()
    exporter = InMemorySpanExporter()
    add_span_exporter(provider, exporter)
    record_llm_span(
        provider.get_tracer("test"),
        model="fake-llm",
        prompt_text="prompt",
        completion=None,
        input_tokens=None,
        output_tokens=None,
        latency_ms=None,
        prompt_id="support",
        prompt_version="1",
        case_id="c",
        role="judge",
        cost_usd=0.0,
        cost_known=False,
        error="down",
    )
    spans = exporter.get_finished_spans()
    assert span_rows(spans)[0]["status"] == "ERROR"
    assert (spans[0].attributes or {})["gen_ai.system"] == "fake"


def test_unknown_role_is_rejected() -> None:
    provider = build_provider()
    with pytest.raises(ValueError, match="role"):
        record_llm_span(
            provider.get_tracer("test"),
            model="fake-llm",
            prompt_text="prompt",
            completion=None,
            input_tokens=None,
            output_tokens=None,
            latency_ms=None,
            prompt_id="support",
            prompt_version="1",
            case_id="c",
            role="other",
            cost_usd=0.0,
            cost_known=True,
        )


def test_langfuse_endpoint_and_basic_auth() -> None:
    assert (
        langfuse_traces_endpoint("https://cloud.langfuse.com/")
        == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    )
    headers = langfuse_basic_headers("pk-test", "sk-test")
    encoded = headers["Authorization"].removeprefix("Basic ")
    assert base64.b64decode(encoded).decode() == "pk-test:sk-test"
    assert headers["x-langfuse-ingestion-version"] == "4"


def test_langfuse_settings_need_both_keys() -> None:
    assert langfuse_settings_from_env({}) is None
    assert langfuse_settings_from_env({"LANGFUSE_PUBLIC_KEY": "pk"}) is None
    settings = langfuse_settings_from_env(
        {
            "LANGFUSE_PUBLIC_KEY": "pk",
            "LANGFUSE_SECRET_KEY": "sk",
            "LANGFUSE_HOST": "https://us.cloud.langfuse.com",
        }
    )
    assert settings is not None
    assert settings.host == "https://us.cloud.langfuse.com"


def test_langfuse_exporter_attaches_without_sending() -> None:
    provider = build_provider()
    add_langfuse_exporter(
        provider,
        host="https://cloud.langfuse.com",
        public_key="pk",
        secret_key="sk",
    )
