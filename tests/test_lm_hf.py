"""Tests for the language model adapter.

Nothing here installs torch. The arithmetic in aic.lm_hf is plain Python over
plain lists precisely so that it can be checked in a dependency free job, and
the only thing the class itself is asked to do is fail with a useful message
when the optional extra is missing.
"""

from __future__ import annotations

import math
import random

import pytest

from aic.ai_detect import likelihood_features
from aic.lm_hf import (
    HuggingFaceLM,
    log_softmax,
    mask_spans,
    rank_of,
    refill,
    triples_from_rows,
)
from aic.normalize import normalize


def test_log_softmax_is_a_distribution():
    logps = log_softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(math.exp(x) for x in logps), 1.0, rel_tol=1e-12)
    assert logps[2] > logps[1] > logps[0]


def test_log_softmax_survives_large_logits():
    logps = log_softmax([1000.0, 1001.0])
    assert math.isclose(sum(math.exp(x) for x in logps), 1.0, rel_tol=1e-12)


def test_log_softmax_of_nothing_is_nothing():
    assert log_softmax([]) == []


def test_a_uniform_row_splits_probability_evenly():
    logps = log_softmax([2.0, 2.0, 2.0, 2.0])
    assert all(math.isclose(math.exp(x), 0.25, rel_tol=1e-12) for x in logps)


def test_rank_is_one_for_the_top_token():
    assert rank_of([0.1, 5.0, 2.0], 1) == 1


def test_rank_counts_ties_against_the_target():
    assert rank_of([5.0, 5.0, 1.0], 0) == 1
    assert rank_of([9.0, 5.0, 5.0], 1) == 2


def test_triples_follow_the_causal_off_by_one_contract():
    rows = [[0.0, 4.0], [3.0, 0.0]]
    targets = [1, 0]
    triples = triples_from_rows(rows, targets, lambda i: "t" + str(i))
    assert [t[0] for t in triples] == ["t1", "t0"]
    assert all(t[1] < 0.0 for t in triples)
    assert all(t[2] == 0.0 for t in triples)


def test_triples_skip_targets_outside_the_vocabulary():
    triples = triples_from_rows([[1.0, 2.0]], [7], lambda i: "x")
    assert triples == []


def test_triples_refuse_mismatched_lengths():
    with pytest.raises(ValueError):
        triples_from_rows([[1.0, 2.0]], [0, 1], lambda i: "x")


def test_masking_inserts_sentinels_and_reports_how_many():
    text = " ".join("word" + str(i) for i in range(40))
    masked, count = mask_spans(text, rate=0.2, rng=random.Random(7))
    assert count > 0
    assert masked.count("<extra_id_") == count


def test_masking_leaves_a_very_short_passage_alone():
    masked, count = mask_spans("two words", rng=random.Random(1))
    assert (masked, count) == ("two words", 0)


def test_masking_is_reproducible_for_a_given_seed():
    text = " ".join("word" + str(i) for i in range(60))
    first, _ = mask_spans(text, rng=random.Random(3))
    second, _ = mask_spans(text, rng=random.Random(3))
    assert first == second


def test_refill_puts_each_answer_back_in_its_own_slot():
    masked = "the <extra_id_0> sat on the <extra_id_1> today"
    generated = "<extra_id_0> small cat <extra_id_1> warm mat <extra_id_2>"
    assert refill(masked, generated) == "the small cat sat on the warm mat today"


def test_refill_drops_a_sentinel_the_model_never_answered():
    masked = "alpha <extra_id_0> gamma"
    assert refill(masked, "") == "alpha gamma"


def test_the_adapter_explains_itself_when_the_extra_is_missing():
    pytest.importorskip
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="optional extra"):
            HuggingFaceLM().token_logprobs("some prose to score")
    else:  # pragma: no cover - only when the extra is installed
        pytest.skip("transformers is installed, so the error path cannot fire")


def test_blank_text_needs_no_model_at_all():
    assert HuggingFaceLM().token_logprobs("   ") == []


class ToyLM:
    """A two token vocabulary standing in for a real causal model.

    Every second token is predicted confidently and the rest are not, which
    is enough to give the likelihood features something non degenerate to
    summarise without downloading half a gigabyte of weights.
    """

    def token_logprobs(self, text: str) -> list[tuple[str, float, float]]:
        words = text.split()
        rows = [[4.0, 0.0] if i % 2 else [0.5, 0.4] for i in range(len(words))]
        targets = [0] * len(words)
        return triples_from_rows(rows, targets, lambda i: words[min(i, 1)])

    def perturb(self, text: str, n: int = 8) -> list[str]:
        return [text.replace("carbon", "nitrogen") for _ in range(n)]


def test_the_likelihood_features_light_up_behind_a_toy_model():
    prose = (
        "Saltmarsh sediments accumulate organic carbon quickly. "
        "Porewater exchange moves alkalinity offshore. "
        "The stock persists for centuries in stable marshes."
    )
    doc = normalize(prose)
    fv = likelihood_features(doc.text, ToyLM(), doc.sentences)
    assert fv.has_likelihood is True
    assert fv.mean_log_prob < 0.0
    assert fv.log_prob_std > 0.0
    assert fv.mean_log_rank >= 0.0
