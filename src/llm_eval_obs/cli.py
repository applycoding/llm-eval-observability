"""Command line for eval, report, and the score gate."""

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, assert_never

from opentelemetry.sdk.trace import TracerProvider

from llm_eval_obs.artifacts import BASELINE_NOTE
from llm_eval_obs.errors import EvalObsError
from llm_eval_obs.gate import evaluate_gate
from llm_eval_obs.report import render_html, render_markdown
from llm_eval_obs.serialize import (
    baseline_from_run,
    baseline_to_dict,
    dump_json,
    load_baseline,
    load_run,
    run_to_dict,
)
from llm_eval_obs.service import run_prompt_version
from llm_eval_obs.tracing import (
    add_langfuse_exporter,
    build_provider,
    langfuse_settings_from_env,
    langfuse_traces_endpoint,
)

Command = Literal["eval", "report", "gate", "freeze"]
ReportFormat = Literal["md", "html"]


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args.command, args)
    except EvalObsError as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _dispatch(command: str, args: argparse.Namespace) -> int:
    kind = _command(command)
    match kind:
        case "eval":
            return _eval(args)
        case "report":
            return _report(args)
        case "gate":
            return _gate(args)
        case "freeze":
            return _freeze(args)
        case _ as unreachable:
            assert_never(unreachable)


def _eval(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    provider = build_provider()
    _maybe_langfuse(provider, enabled=not args.no_langfuse)
    tracer = provider.get_tracer("llm_eval_obs")
    try:
        run = run_prompt_version(
            root=root,
            version=args.version,
            suite_path=_optional_path(args.suite),
            prompt_dir=_optional_path(args.prompt_dir),
            tracer=tracer,
        )
    finally:
        provider.force_flush()
    payload = run_to_dict(run)
    if args.out:
        dump_json(Path(args.out), payload)
        print(f"wrote {args.out}")
        return 0
    print(json.dumps(payload, indent=2) + "\n", end="")
    return 0


def _report(args: argparse.Namespace) -> int:
    before = load_run(Path(args.before))
    after = load_run(Path(args.after))
    kind = _format(args.format)
    match kind:
        case "md":
            body = render_markdown(before, after)
        case "html":
            body = render_html(before, after)
        case _ as unreachable:
            assert_never(unreachable)
    if not body.endswith("\n"):
        body += "\n"
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(body, encoding="utf-8")
        print(f"wrote {args.out}")
        return 0
    print(body, end="")
    return 0


def _gate(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    baseline_path = Path(args.baseline) if args.baseline else root / "baselines" / "support_v2.json"
    baseline = load_baseline(baseline_path)
    if args.run:
        run = load_run(Path(args.run))
    else:
        provider = build_provider()
        _maybe_langfuse(provider, enabled=not args.no_langfuse)
        try:
            run = run_prompt_version(
                root=root,
                version=baseline.prompt_version,
                suite_path=_optional_path(args.suite),
                prompt_dir=_optional_path(args.prompt_dir),
                tracer=provider.get_tracer("llm_eval_obs"),
            )
        finally:
            provider.force_flush()
    result = evaluate_gate(run, baseline, tolerance=args.tolerance)
    if result.passed:
        print(f"gate passed ({len(result.comparisons)} scorers, tolerance {args.tolerance})")
        return 0
    for failure in result.failures:
        print(failure, file=sys.stderr)
    return 1


def _freeze(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    run = run_prompt_version(
        root=root,
        version=args.version,
        suite_path=_optional_path(args.suite),
        prompt_dir=_optional_path(args.prompt_dir),
    )
    note = args.note or BASELINE_NOTE
    dump_json(Path(args.out), baseline_to_dict(baseline_from_run(run, note)))
    print(f"wrote {args.out}")
    return 0


def _maybe_langfuse(provider: TracerProvider, *, enabled: bool) -> None:
    if not enabled:
        return
    settings = langfuse_settings_from_env()
    if settings is None:
        return
    add_langfuse_exporter(
        provider,
        host=settings.host,
        public_key=settings.public_key,
        secret_key=settings.secret_key,
    )
    print(
        f"exporting spans to {langfuse_traces_endpoint(settings.host)}",
        file=sys.stderr,
    )


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=".")
    common.add_argument("--suite")
    common.add_argument("--prompt-dir")

    parser = argparse.ArgumentParser(prog="llm-eval-obs")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_cmd = sub.add_parser("eval", parents=[common], help="score one prompt version")
    eval_cmd.add_argument("--version", required=True)
    eval_cmd.add_argument("--out")
    eval_cmd.add_argument(
        "--no-langfuse",
        action="store_true",
        help="do not export spans even if Langfuse keys are set",
    )

    report = sub.add_parser("report", help="compare two saved eval runs")
    report.add_argument("--before", required=True)
    report.add_argument("--after", required=True)
    report.add_argument("--format", choices=("md", "html"), default="md")
    report.add_argument("--out")

    gate = sub.add_parser("gate", parents=[common], help="fail if scores drop below the baseline")
    gate.add_argument("--baseline")
    gate.add_argument("--run", help="compare this saved run instead of executing the suite")
    gate.add_argument("--tolerance", type=float, default=0.0)
    gate.add_argument("--no-langfuse", action="store_true")

    freeze = sub.add_parser("freeze", parents=[common], help="write a baseline from a fresh run")
    freeze.add_argument("--version", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--note")
    return parser


def _optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _command(value: str) -> Command:
    match value:
        case "eval" | "report" | "gate" | "freeze":
            return value
        case _:
            raise EvalObsError(f"unknown command {value}")


def _format(value: str) -> ReportFormat:
    match value:
        case "md" | "html":
            return value
        case _:
            raise EvalObsError(f"unknown report format {value}")
