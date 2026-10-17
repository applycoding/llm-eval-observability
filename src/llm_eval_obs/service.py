"""Load a suite from disk and evaluate one prompt version."""

from pathlib import Path

from opentelemetry.trace import Tracer

from llm_eval_obs.cost import load_catalog
from llm_eval_obs.datasets import load_dataset
from llm_eval_obs.llm import FakeLLM, LLMClient
from llm_eval_obs.models import EvalRun
from llm_eval_obs.prompts import PromptRegistry
from llm_eval_obs.runner import evaluate
from llm_eval_obs.scoring import build_scorer
from llm_eval_obs.suite import load_suite
from llm_eval_obs.tracing import build_provider


def run_prompt_version(
    *,
    root: Path,
    version: str,
    suite_path: Path | None = None,
    prompt_dir: Path | None = None,
    llm: LLMClient | None = None,
    tracer: Tracer | None = None,
) -> EvalRun:
    base = root.resolve()
    suite = load_suite(suite_path or base / "suites" / "support.yaml", root=base)
    prompts = PromptRegistry.load(prompt_dir or base / "prompts" / "support")
    prompt = prompts.get(suite.prompt_id, version)
    cases = load_dataset(suite.dataset)
    catalog = load_catalog(suite.price_catalog)
    scorers = [build_scorer(spec) for spec in suite.scorers]
    active = tracer if tracer is not None else build_provider().get_tracer("llm_eval_obs")
    return evaluate(
        prompt=prompt,
        cases=cases,
        scorers=scorers,
        llm=llm or FakeLLM(),
        model=suite.model,
        catalog=catalog,
        tracer=active,
        dataset_name=suite.dataset_name,
        suite_id=suite.id,
    )
