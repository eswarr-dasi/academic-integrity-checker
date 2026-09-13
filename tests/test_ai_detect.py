"""Tests for the AI-writing detector.

The point of these tests is not accuracy, which cannot be established without
a labelled corpus. It is that the plumbing is honest: features go in the
direction the literature says they do, an uncalibrated model announces itself,
the interval always contains the point estimate, and no code path can produce
a bare verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from aic.ai_detect import (
    AIDetector,
    FeatureVector,
    IsotonicCalibrator,
    band_for,
    likelihood_features,
    sigmoid,
    stylometry,
)
from aic.normalize import normalize

PROSE = (
    "Urban tree canopy reduces peak summer surface temperature mainly through "
    "transpiration and not through shade. A drought stressed plane tree closes "
    "its stomata by midday and behaves thermally like a lamp post. Irrigation "
    "planning therefore belongs in the canopy strategy rather than in the "
    "maintenance budget, where it is usually filed and then forgotten."
)

# Twenty distinct sentences rather than one template repeated thirty times. A
# repetitive document has almost no between-sentence variation, which makes
# burstiness untestable and every segment score identical by construction.
SENTENCES = [
    "The recharge budget of a semi arid basin is set by storm intensity rather than by annual rainfall totals.",
    "Channel infiltration dominates, so the wetted width of the wadi matters more than the peak depth of any single flood.",
    "Marker horizon surveys are cheap to install and tedious to read, which is why so few schemes have more than one year of data.",
    "A drought stressed street tree closes its stomata by midday and cools almost nothing for the rest of the afternoon.",
    "Discount rates flatten any benefit that accrues slowly, and sediment carbon accrues about as slowly as anything in the ledger.",
    "The appraisal question is not whether the sink exists but whether it can be evidenced before the funding decision closes.",
    "Two neighbouring estuaries can differ by an order of magnitude in accretion rate for reasons that are mostly geometric.",
    "Vegetation density is easy to map from the air and turns out to be a weak predictor of long term storage.",
    "Reviewers ask for uncertainty bounds, and the honest bounds are wide enough that the central estimate stops being persuasive.",
    "Monitoring budgets are approved annually while the processes being monitored operate on decadal timescales.",
    "Nobody disputes the mechanism; the dispute is entirely about the constant of proportionality.",
    "A single convective storm can deliver more runoff than the preceding month of steady drizzle.",
    "Antecedent moisture explained far less of the variance than the depth of the sandy alluvium did.",
    "Relocating runoff behind a small dam mostly relocates the losses to an evaporating reservoir surface.",
    "Managed realignment is unpopular locally for reasons that have nothing to do with the carbon arithmetic.",
    "The literature reports means without reporting the spread, which makes meta analysis close to impossible.",
    "Field crews measure what is measurable in a week, and that constraint quietly shapes the published record.",
    "Sensor drift over a five year deployment is larger than the effect most of these studies are trying to detect.",
    "Any model calibrated on one estuary should be treated as a hypothesis when applied to the next one.",
    "The policy conclusion survives the uncertainty even though none of the individual numbers do.",
]

LONG = " ".join(SENTENCES * 2)


@dataclass
class FakeLM:
    """Deterministic stand-in for a reference language model.

    Returning fixed numbers instead of loading gpt2 keeps this suite offline
    and fast. It tests the feature extraction code, not the model.
    """

    base: float = -3.0
    wobble: float = 0.0

    def token_logprobs(self, text: str) -> list[tuple[str, float, float]]:
        out = []
        for word in text.split():
            # Word derived, not position derived. A wobble that depends on the
            # index within the chunk cancels out across sentences of similar
            # length, which would hold burstiness constant and make the
            # feature look broken when it is the stub that is broken.
            seed = sum(ord(c) for c in word)
            jitter = (seed % 23) / 23.0 - 0.5
            lp = self.base + self.wobble * jitter - 0.02 * len(word)
            out.append((word, lp, math.log(2 + (seed % 5))))
        return out

    def perturb(self, text: str, n: int = 8) -> list[str]:
        words = text.split()
        if len(words) < 20:
            return []
        return [" ".join(words[offset::2]) for offset in range(min(n, 2))]


# --------------------------------------------------------------------------
# stylometry
# --------------------------------------------------------------------------

def test_stylometry_produces_bounded_ratios():
    fv = stylometry(normalize(PROSE))
    assert 0.0 < fv.type_token_ratio <= 1.0
    assert 0.0 <= fv.hapax_ratio <= 1.0
    assert 0.0 < fv.function_word_ratio < 1.0
    assert fv.sent_len_mean > 0
    assert fv.punct_entropy >= 0.0
    assert fv.has_likelihood is False


def test_stylometry_on_empty_input_does_not_explode():
    fv = stylometry(normalize(""))
    assert fv.sent_len_mean == 0.0
    assert fv.type_token_ratio == 0.0


# --------------------------------------------------------------------------
# likelihood features
# --------------------------------------------------------------------------

def test_likelihood_features_are_populated_when_a_model_is_supplied():
    doc = normalize(LONG)
    fv = likelihood_features(doc.text, FakeLM(wobble=0.4), doc.sentences)
    assert fv.has_likelihood is True
    assert fv.mean_log_prob < 0
    assert fv.log_prob_std > 0
    assert fv.mean_log_rank > 0


def test_burstiness_rises_with_sentence_level_variation():
    doc = normalize(LONG)
    smooth = likelihood_features(doc.text, FakeLM(wobble=0.0), doc.sentences)
    varied = likelihood_features(doc.text, FakeLM(wobble=1.5), doc.sentences)
    assert varied.burstiness > smooth.burstiness


# --------------------------------------------------------------------------
# model direction
# --------------------------------------------------------------------------

def _baseline() -> FeatureVector:
    return FeatureVector(
        mean_log_prob=-3.10,
        log_prob_std=2.40,
        burstiness=0.55,
        mean_log_rank=2.90,
        curvature=0.30,
        sent_len_mean=22.0,
        sent_len_cv=0.52,
        type_token_ratio=0.46,
        hapax_ratio=0.33,
        function_word_ratio=0.42,
        punct_entropy=2.10,
        discourse_marker_rate=0.012,
        has_likelihood=True,
    )


def test_a_baseline_feature_vector_sits_near_the_decision_middle():
    # By construction the baseline equals the standardisation means, so the
    # score should be sigmoid(bias) and nothing else.
    assert abs(AIDetector().raw_score(_baseline()) - sigmoid(-0.35)) < 1e-9


def test_burstiness_pushes_the_score_toward_human():
    det = AIDetector()
    base = _baseline()
    assert det.raw_score(replace(base, burstiness=1.20)) < det.raw_score(base)
    assert det.raw_score(replace(base, burstiness=0.05)) > det.raw_score(base)


def test_curvature_pushes_the_score_toward_machine():
    det = AIDetector()
    base = _baseline()
    assert det.raw_score(replace(base, curvature=1.50)) > det.raw_score(base)


def test_higher_token_probability_pushes_the_score_toward_machine():
    det = AIDetector()
    base = _baseline()
    assert det.raw_score(replace(base, mean_log_prob=-1.20)) > det.raw_score(base)


def test_likelihood_features_are_ignored_without_a_language_model():
    det = AIDetector()
    style_only = replace(_baseline(), has_likelihood=False)
    shifted = replace(style_only, mean_log_prob=-0.1, curvature=9.0)
    assert det.raw_score(style_only) == det.raw_score(shifted)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def test_isotonic_fit_is_monotone():
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [0, 0, 1, 0, 0, 1, 1, 0, 1, 1]
    cal = IsotonicCalibrator.fit(scores, labels)
    assert cal.fitted
    mapped = [cal(s) for s in scores]
    assert mapped == sorted(mapped)
    assert cal(0.01) <= cal(0.99)
    assert all(0.0 <= m <= 1.0 for m in mapped)


def test_unfitted_calibrator_is_the_identity():
    cal = IsotonicCalibrator()
    assert cal.fitted is False
    assert cal(0.42) == 0.42


def test_bands_are_ordered_and_have_a_wide_unclear_region():
    assert band_for(0.05) == "human"
    assert band_for(0.19) == "human"
    assert band_for(0.50) == "unclear"
    assert band_for(0.74) == "unclear"
    assert band_for(0.80) == "likely-ai"
    assert band_for(0.99) == "very-likely-ai"


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

def test_analysis_is_self_consistent():
    analysis = AIDetector(lm=FakeLM(wobble=0.6)).analyze(normalize(LONG))
    lo, hi = analysis.confidence_interval
    assert 0.0 <= lo <= analysis.index <= hi <= 1.0
    assert analysis.band == band_for(analysis.index)
    assert analysis.used_language_model is True
    assert len(analysis.segments) >= 2
    assert all(s.band == band_for(s.score) for s in analysis.segments)
    assert all(0 <= s.span[0] < s.span[1] <= len(LONG) for s in analysis.segments)


def test_an_uncalibrated_detector_says_so():
    analysis = AIDetector().analyze(normalize(LONG))
    assert analysis.calibrated is False
    assert any("calibration" in c for c in analysis.caveats)
    assert any("not evidence of misconduct" in c for c in analysis.caveats)


def test_running_without_a_language_model_is_disclosed():
    analysis = AIDetector().analyze(normalize(LONG))
    assert analysis.used_language_model is False
    assert any("stylometric features only" in c for c in analysis.caveats)


def test_short_documents_get_a_length_warning():
    analysis = AIDetector().analyze(normalize(PROSE))
    assert any("short" in c.lower() for c in analysis.caveats)


def test_calibration_narrows_the_reported_interval():
    doc = normalize(LONG)
    loose = AIDetector().analyze(doc)
    tight_detector = AIDetector()
    tight_detector.calibrator = IsotonicCalibrator.fit(
        [0.1, 0.3, 0.5, 0.7, 0.9], [0, 0, 1, 1, 1]
    )
    tight = tight_detector.analyze(doc)
    loose_width = loose.confidence_interval[1] - loose.confidence_interval[0]
    tight_width = tight.confidence_interval[1] - tight.confidence_interval[0]
    assert tight_width < loose_width


def test_report_dict_never_contains_a_verdict_field():
    analysis = AIDetector().analyze(normalize(LONG))
    payload = analysis.to_dict()
    assert "verdict" not in payload
    assert "cheated" not in payload
    assert payload["band"] in ("human", "unclear", "likely-ai", "very-likely-ai")
    assert payload["caveats"]
