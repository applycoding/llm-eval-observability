"""Exact, regex, hashing similarity, and judge parsing."""

import pytest

from llm_eval_obs.errors import ScorerError
from llm_eval_obs.llm import FakeLLM
from llm_eval_obs.models import TestCase
from llm_eval_obs.rubric import load_rubric
from llm_eval_obs.scoring import (
    EmbeddingScorer,
    ExactScorer,
    HashingEmbedder,
    JudgeScorer,
    RegexScorer,
    ScoreContext,
    build_scorer,
    cosine,
    parse_judge_payload,
)
from llm_eval_obs.suite import EmbeddingSpec, ExactSpec, JudgeSpec, RegexSpec
from llm_eval_obs.tracing import build_provider
from tests.factories import ROOT


def _ctx(model: str = "fake-llm") -> ScoreContext:
    return ScoreContext(
        llm=FakeLLM(),
        model=model,
        tracer=build_provider().get_tracer("test"),
        prompt_id="support",
        prompt_version="2",
        case_id="case",
        catalog={},
    )


def test_exact_match_casefolds_whitespace() -> None:
    scorer = ExactScorer(ExactSpec(type="exact", name="exact_reply"))
    case = TestCase("c", {"question": "q"}, expected="Hello   there")
    outcome = scorer.score(case, " hello there ", _ctx())
    assert outcome.score.value == 1.0
    strict = ExactScorer(ExactSpec(type="exact", name="exact_reply", casefold=False))
    assert strict.score(case, "hello there", _ctx()).score.value == 0.0


def test_exact_scorer_skips_when_expected_is_absent() -> None:
    scorer = ExactScorer(ExactSpec(type="exact", name="exact_reply"))
    assert scorer.applies(TestCase("c", {"question": "q"})) is False


def test_regex_is_case_insensitive_by_default() -> None:
    scorer = RegexScorer(RegexSpec(type="regex", name="policy_regex"))
    case = TestCase("c", {"question": "q"}, expected_regex="escalate")
    assert scorer.score(case, "Please Escalate this.", _ctx()).score.value == 1.0
    assert scorer.score(case, "no match here", _ctx()).score.value == 0.0


def test_invalid_regex_raises() -> None:
    scorer = RegexScorer(RegexSpec(type="regex", name="policy_regex"))
    case = TestCase("c", {"question": "q"}, expected_regex="(")
    with pytest.raises(ScorerError, match="invalid regex"):
        scorer.score(case, "text", _ctx())


def test_hashing_embedder_is_deterministic_and_local() -> None:
    embedder = HashingEmbedder()
    same = cosine(embedder.embed("refund window"), embedder.embed("refund window"))
    different = cosine(embedder.embed("alpha"), embedder.embed("beta"))
    empty = cosine(embedder.embed(""), embedder.embed("alpha"))
    assert same == 1.0
    assert different < 1.0
    assert empty == 0.0


def test_embedding_scorer_uses_the_reference() -> None:
    spec = EmbeddingSpec(type="embedding", name="reference_similarity", threshold=0.99)
    scorer = EmbeddingScorer(spec)
    case = TestCase("c", {"question": "q"}, reference="same words")
    matched = scorer.score(case, "same words", _ctx())
    assert matched.score.value == 1.0
    assert matched.score.passed is True
    missed = scorer.score(case, "completely other", _ctx())
    assert missed.score.value is not None
    assert missed.score.value < 1.0


def test_judge_parser_accepts_fenced_json_and_rejects_bad_scores() -> None:
    rubric = load_rubric(ROOT / "rubrics" / "support_quality.yaml")
    parsed = parse_judge_payload(
        '```json\n{"scores": {"faithfulness": 5, "helpfulness": 3}, "rationale": "ok"}\n```',
        rubric,
    )
    assert parsed == {"faithfulness": 5, "helpfulness": 3}
    with pytest.raises(ScorerError, match="outside"):
        parse_judge_payload('{"scores": {"faithfulness": 9, "helpfulness": 3}}', rubric)
    with pytest.raises(ScorerError, match="JSON"):
        parse_judge_payload("not json", rubric)


def test_judge_weights_are_normalized() -> None:
    rubric_path = ROOT / "rubrics" / "support_quality.yaml"
    scorer = build_scorer(JudgeSpec(type="llm_judge", name="rubric_judge", rubric=rubric_path))
    assert isinstance(scorer, JudgeScorer)
    case = TestCase("c", {"question": "q"}, reference="policy text")

    def handler(prompt: str) -> str:
        assert "RESPOND WITH JSON ONLY" in prompt
        assert "faithfulness" in prompt
        return '{"scores": {"faithfulness": 5, "helpfulness": 1}, "rationale": "split"}'

    ctx = _ctx()
    ctx = type(ctx)(
        llm=FakeLLM(handler),
        model=ctx.model,
        tracer=ctx.tracer,
        prompt_id=ctx.prompt_id,
        prompt_version=ctx.prompt_version,
        case_id=ctx.case_id,
        catalog=ctx.catalog,
    )
    outcome = scorer.score(case, "candidate", ctx)
    # scale 1..5, weights 0.6 and 0.4: (1.0 * 0.6) + (0.0 * 0.4)
    assert outcome.score.value == 0.6
    assert outcome.usage is not None
