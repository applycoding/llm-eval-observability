"""Fail a run when a scorer mean drops below a frozen baseline.

The comparison allows a caller-supplied tolerance plus a tiny slack so binary
floating point does not fail a gate that did not actually move. The slack is
not a quality margin.
"""

from dataclasses import dataclass

from llm_eval_obs.aggregate import scorer_means
from llm_eval_obs.errors import GateError
from llm_eval_obs.models import Baseline, EvalRun

COMPARISON_SLACK = 1e-6


@dataclass(frozen=True)
class ScorerComparison:
    name: str
    baseline: float
    actual: float
    dropped: bool


@dataclass(frozen=True)
class GateResult:
    passed: bool
    tolerance: float
    comparisons: tuple[ScorerComparison, ...]
    failures: tuple[str, ...]


def evaluate_gate(run: EvalRun, baseline: Baseline, *, tolerance: float) -> GateResult:
    if tolerance < 0:
        raise GateError("tolerance must be >= 0")
    failures: list[str] = []
    failures.extend(_metadata_failures(run, baseline))
    actual = scorer_means(run)
    comparisons: list[ScorerComparison] = []
    for name, expected in sorted(baseline.scorer_means.items()):
        if name not in actual:
            failures.append(f"{name} is missing from the run")
            continue
        seen = actual[name]
        dropped = expected - seen > tolerance + COMPARISON_SLACK
        if dropped:
            failures.append(
                f"{name} dropped from {expected:.6f} to {seen:.6f} (tolerance {tolerance})"
            )
        comparisons.append(
            ScorerComparison(name=name, baseline=expected, actual=seen, dropped=dropped)
        )
    for name in sorted(set(actual) - set(baseline.scorer_means)):
        failures.append(f"{name} is not in the baseline")
    return GateResult(
        passed=not failures,
        tolerance=tolerance,
        comparisons=tuple(comparisons),
        failures=tuple(failures),
    )


def _metadata_failures(run: EvalRun, baseline: Baseline) -> list[str]:
    checks = (
        ("prompt_id", run.prompt_id, baseline.prompt_id),
        ("prompt_version", run.prompt_version, baseline.prompt_version),
        ("dataset", run.dataset, baseline.dataset),
        ("model", run.model, baseline.model),
        ("suite_id", run.suite_id, baseline.suite_id),
    )
    return [
        f"{label} is {actual!r}; baseline has {expected!r}"
        for label, actual, expected in checks
        if actual != expected
    ]
