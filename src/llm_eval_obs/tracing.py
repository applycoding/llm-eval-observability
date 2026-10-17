"""OpenTelemetry spans for LLM calls, with an optional Langfuse OTLP destination.

Langfuse accepts OTLP/HTTP at `/api/public/otel/v1/traces`. Project keys are
sent as Basic auth plus `x-langfuse-ingestion-version: 4`, which Langfuse
documents for real-time ingestion. The exporter is only constructed when a
caller asks for it. Nothing in this module opens a connection by itself.
"""

import base64
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Status, StatusCode, Tracer

SERVICE_NAME = "llm-eval-obs"
LANGFUSE_INGESTION_HEADER = "x-langfuse-ingestion-version"
LANGFUSE_INGESTION_VERSION = "4"
DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"
_CLIP = 2000


@dataclass(frozen=True)
class LangfuseSettings:
    host: str
    public_key: str
    secret_key: str


def build_provider() -> TracerProvider:
    return TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))


def add_span_exporter(provider: TracerProvider, exporter: SpanExporter) -> None:
    provider.add_span_processor(SimpleSpanProcessor(exporter))


def langfuse_traces_endpoint(host: str) -> str:
    """Full OTLP/HTTP traces URL. The Python exporter does not append this path."""

    return f"{host.rstrip('/')}/api/public/otel/v1/traces"


def langfuse_basic_headers(public_key: str, secret_key: str) -> dict[str, str]:
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return {
        "Authorization": f"Basic {token}",
        LANGFUSE_INGESTION_HEADER: LANGFUSE_INGESTION_VERSION,
    }


def langfuse_settings_from_env(env: Mapping[str, str] | None = None) -> LangfuseSettings | None:
    source = os.environ if env is None else env
    public_key = source.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = source.get("LANGFUSE_SECRET_KEY", "").strip()
    if not public_key or not secret_key:
        return None
    host = source.get("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST).strip() or DEFAULT_LANGFUSE_HOST
    return LangfuseSettings(host=host, public_key=public_key, secret_key=secret_key)


def add_langfuse_exporter(
    provider: TracerProvider,
    *,
    host: str,
    public_key: str,
    secret_key: str,
) -> None:
    exporter = OTLPSpanExporter(
        endpoint=langfuse_traces_endpoint(host),
        headers=langfuse_basic_headers(public_key, secret_key),
    )
    add_span_exporter(provider, exporter)


def record_llm_span(
    tracer: Tracer,
    *,
    model: str,
    prompt_text: str,
    completion: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: float | None,
    prompt_id: str,
    prompt_version: str,
    case_id: str,
    role: str,
    cost_usd: float,
    cost_known: bool,
    error: str | None = None,
) -> None:
    """Emit one `gen_ai.chat` span. Attributes follow GenAI semantic conventions."""

    if role not in {"task", "judge"}:
        raise ValueError(f"unknown span role: {role}")
    with tracer.start_as_current_span("gen_ai.chat") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.system", _gen_ai_system(model))
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("llm_eval_obs.prompt.id", prompt_id)
        span.set_attribute("llm_eval_obs.prompt.version", prompt_version)
        span.set_attribute("llm_eval_obs.case_id", case_id)
        span.set_attribute("llm_eval_obs.role", role)
        span.set_attribute("llm_eval_obs.cost.usd", cost_usd)
        span.set_attribute("llm_eval_obs.cost.known", cost_known)
        span.set_attribute("llm_eval_obs.prompt", _clip(prompt_text))
        if completion is not None:
            span.set_attribute("gen_ai.response.model", model)
            span.set_attribute("llm_eval_obs.completion", _clip(completion))
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        if latency_ms is not None:
            span.set_attribute("llm_eval_obs.latency_ms", latency_ms)
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, error))


def span_rows(spans: Sequence[ReadableSpan]) -> list[dict[str, str]]:
    """Flatten finished spans into rows the dashboard can show without extra deps."""

    rows: list[dict[str, str]] = []
    for span in spans:
        attributes = span.attributes or {}
        rows.append(
            {
                "name": span.name,
                "case": _as_text(attributes.get("llm_eval_obs.case_id")),
                "role": _as_text(attributes.get("llm_eval_obs.role")),
                "model": _as_text(attributes.get("gen_ai.request.model")),
                "input_tokens": _as_text(attributes.get("gen_ai.usage.input_tokens")),
                "output_tokens": _as_text(attributes.get("gen_ai.usage.output_tokens")),
                "latency_ms": _as_text(attributes.get("llm_eval_obs.latency_ms")),
                "cost_usd": _as_text(attributes.get("llm_eval_obs.cost.usd")),
                "cost_known": _as_text(attributes.get("llm_eval_obs.cost.known")),
                "status": span.status.status_code.name,
            }
        )
    return rows


def _gen_ai_system(model: str) -> str:
    if model == "fake-llm":
        return "fake"
    if model.startswith("gpt-"):
        return "openai"
    return "unknown"


def _clip(text: str) -> str:
    if len(text) <= _CLIP:
        return text
    return text[:_CLIP]


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)
