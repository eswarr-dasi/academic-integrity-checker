"""AI-writing detection: features first, one small calibrated model on top.

Design constraints, in priority order:

1. Auditable. Every score decomposes into named features, and the model on top
   of them is logistic regression with eleven weights. If a student disputes a
   score, the exact reason can be printed.
2. Calibrated. A raw classifier margin is not a probability. Isotonic
   regression maps the margin onto the observed positive rate in a held out
   set, so 0.9 means 0.9 and not "high".
3. Honest about uncertainty. Output is a band plus an interval, never a
   verdict. The detector reports "unclear" for a wide middle region on
   purpose, because the alternative is accusing people on a coin flip.

Feature families
----------------
Likelihood, from a reference language model:
    mean_log_prob     - average token log probability, negated perplexity in
                        log space. Sampled text tends to sit closer to the
                        model mode than human text does.
    log_prob_std      - spread of token log probabilities.
    burstiness        - standard deviation of per-sentence mean log
                        probability. Human writing lurches between easy and
                        hard sentences, decoded text is smoother.
    mean_log_rank     - average log rank of the chosen token. Robust to
                        temperature in a way raw probability is not.
    curvature         - DetectGPT style probe. Perturb the passage and measure
                        how far log likelihood falls. Machine text sits near a
                        local maximum of the model log probability, human text
                        does not.

Stylometric, no model required:
    sent_len_mean, sent_len_cv, type_token_ratio, hapax_ratio,
    function_word_ratio, punct_entropy, discourse_marker_rate

Known failure modes are documented in docs/ETHICS.md. The important one:
likelihood features are systematically higher for second language writers and
for formulaic technical prose, which makes false positives unevenly
distributed across a student population. Treat any threshold as a policy
decision with a cost, not as a property of the text.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .normalize import NormalizedDoc, Span, normalize

HUMAN = "human"
UNCLEAR = "unclear"
LIKELY_AI = "likely-ai"
VERY_LIKELY_AI = "very-likely-ai"

# Wide on purpose. The gap between 0.20 and 0.75 is where the detector says it
# does not know, which for a typical essay corpus is a large fraction of cases.
BANDS = ((0.20, HUMAN), (0.75, UNCLEAR), (0.90, LIKELY_AI), (1.01, VERY_LIKELY_AI))

FUNCTION_WORDS = frozenset(
    """a an the and or but if while of to in on at by for with from into about
    as is are was were be been being have has had do does did will would can
    could should may might must this that these those it its they them their
    he she his her we us our you your i me my not no nor so than then there
    here when where which who whom whose what how why all any both each few
    more most other some such only own same very just also however therefore
    moreover furthermore thus hence although though because since unless
    whereas""".split()
)

DISCOURSE_MARKERS = frozenset(
    """however therefore moreover furthermore additionally consequently thus
    hence nevertheless nonetheless overall importantly notably specifically
    ultimately significantly comprehensive delve leverage underscore pivotal
    realm intricate multifaceted""".split()
)

SEGMENT_WORDS = 150
MIN_SEGMENT_WORDS = 40


class LanguageModel(Protocol):
    """Minimal interface the likelihood features need.

    token_logprobs returns one (token, log_prob, log_rank) triple per token.
    Any causal LM can implement this in about twenty lines; see
    HuggingFaceLM below for the reference adapter.
    """

    def token_logprobs(self, text: str) -> list[tuple[str, float, float]]:
        ...

    def perturb(self, text: str, n: int = 8) -> list[str]:
        ...


@dataclass
class FeatureVector:
    mean_log_prob: float = 0.0
    log_prob_std: float = 0.0
    burstiness: float = 0.0
    mean_log_rank: float = 0.0
    curvature: float = 0.0
    sent_len_mean: float = 0.0
    sent_len_cv: float = 0.0
    type_token_ratio: float = 0.0
    hapax_ratio: float = 0.0
    function_word_ratio: float = 0.0
    punct_entropy: float = 0.0
    discourse_marker_rate: float = 0.0
    has_likelihood: bool = False

    def as_dict(self) -> dict[str, float]:
        return {
            k: round(float(v), 4)
            for k, v in self.__dict__.items()
            if k != "has_likelihood"
        }


# --------------------------------------------------------------------------
# stylometry (always available)
# --------------------------------------------------------------------------

def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p, 2)
    return h


def stylometry(doc: NormalizedDoc) -> FeatureVector:
    words = doc.words
    fv = FeatureVector()
    if not words:
        return fv

    lengths = []
    for sent in doc.sentences:
        n = sum(1 for t in doc.tokens if sent.start <= t.start < sent.end)
        if n:
            lengths.append(n)
    if lengths:
        mean = sum(lengths) / len(lengths)
        var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        fv.sent_len_mean = mean
        fv.sent_len_cv = (math.sqrt(var) / mean) if mean else 0.0

    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    fv.type_token_ratio = len(counts) / len(words)
    fv.hapax_ratio = sum(1 for c in counts.values() if c == 1) / len(words)
    fv.function_word_ratio = sum(1 for w in words if w in FUNCTION_WORDS) / len(words)
    fv.discourse_marker_rate = sum(
        1 for w in words if w in DISCOURSE_MARKERS
    ) / len(words)

    punct: dict[str, int] = {}
    for ch in doc.text:
        if not ch.isalnum() and not ch.isspace():
            punct[ch] = punct.get(ch, 0) + 1
    fv.punct_entropy = _entropy(list(punct.values()))
    return fv


# --------------------------------------------------------------------------
# likelihood features (require a reference LM)
# --------------------------------------------------------------------------

def likelihood_features(
    text: str, lm: LanguageModel, sentences: Sequence[Span] | None = None
) -> FeatureVector:
    fv = FeatureVector()
    triples = lm.token_logprobs(text)
    if not triples:
        return fv

    logps = [lp for _, lp, _ in triples]
    ranks = [lr for _, _, lr in triples]
    n = len(logps)
    fv.mean_log_prob = sum(logps) / n
    fv.log_prob_std = math.sqrt(
        sum((x - fv.mean_log_prob) ** 2 for x in logps) / n
    )
    fv.mean_log_rank = sum(ranks) / n

    # Burstiness: variation of difficulty between sentences, not within them.
    per_sentence: list[float] = []
    if sentences:
        for sent in sentences:
            chunk = text[sent.start:sent.end]
            if len(chunk.split()) >= 4:
                sub = lm.token_logprobs(chunk)
                if sub:
                    per_sentence.append(sum(lp for _, lp, _ in sub) / len(sub))
    if len(per_sentence) >= 2:
        mean = sum(per_sentence) / len(per_sentence)
        fv.burstiness = math.sqrt(
            sum((x - mean) ** 2 for x in per_sentence) / len(per_sentence)
        )

    # Curvature: how much does log likelihood drop under small rewrites.
    perturbed = lm.perturb(text)
    if perturbed:
        drops = []
        for variant in perturbed:
            sub = lm.token_logprobs(variant)
            if sub:
                drops.append(fv.mean_log_prob - sum(lp for _, lp, _ in sub) / len(sub))
        if drops:
            mean_drop = sum(drops) / len(drops)
            spread = math.sqrt(
                sum((d - mean_drop) ** 2 for d in drops) / len(drops)
            ) or 1.0
            fv.curvature = mean_drop / spread

    fv.has_likelihood = True
    return fv


def merge_features(style: FeatureVector, like: FeatureVector) -> FeatureVector:
    out = FeatureVector(**style.__dict__)
    if like.has_likelihood:
        out.mean_log_prob = like.mean_log_prob
        out.log_prob_std = like.log_prob_std
        out.burstiness = like.burstiness
        out.mean_log_rank = like.mean_log_rank
        out.curvature = like.curvature
        out.has_likelihood = True
    return out


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

# PLACEHOLDER WEIGHTS. These encode the direction of each effect reported in
# the literature so the pipeline is runnable end to end, and nothing more.
# They are not fitted. Any deployment must run tools/fit_detector.py on its
# own labelled data and replace this block, or the reported probability is
# decoration. See docs/EVALUATION.md.
DEFAULT_WEIGHTS: dict[str, float] = {
    "mean_log_prob": 1.10,
    "log_prob_std": -0.55,
    "burstiness": -1.40,
    "mean_log_rank": -0.90,
    "curvature": 1.25,
    "sent_len_cv": -0.80,
    "type_token_ratio": -0.45,
    "hapax_ratio": -0.50,
    "function_word_ratio": 0.20,
    "punct_entropy": -0.35,
    "discourse_marker_rate": 0.60,
}
DEFAULT_BIAS = -0.35

# Feature standardisation, also placeholder, also from the fitting script.
DEFAULT_MEANS: dict[str, float] = {
    "mean_log_prob": -3.10, "log_prob_std": 2.40, "burstiness": 0.55,
    "mean_log_rank": 2.90, "curvature": 0.30, "sent_len_cv": 0.52,
    "type_token_ratio": 0.46, "hapax_ratio": 0.33,
    "function_word_ratio": 0.42, "punct_entropy": 2.10,
    "discourse_marker_rate": 0.012,
}
DEFAULT_STDS: dict[str, float] = {
    "mean_log_prob": 0.80, "log_prob_std": 0.60, "burstiness": 0.22,
    "mean_log_rank": 0.70, "curvature": 0.45, "sent_len_cv": 0.18,
    "type_token_ratio": 0.09, "hapax_ratio": 0.08,
    "function_word_ratio": 0.06, "punct_entropy": 0.40,
    "discourse_marker_rate": 0.010,
}


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass
class IsotonicCalibrator:
    """Piecewise constant, monotone map from raw score to probability.

    Fitted with pool adjacent violators on held out data. The identity
    calibrator below is a deliberate no-op so that an uncalibrated model is
    obvious in the report rather than silently trusted.
    """

    breakpoints: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    @property
    def fitted(self) -> bool:
        return bool(self.breakpoints)

    def __call__(self, raw: float) -> float:
        if not self.fitted:
            return raw
        for bp, val in zip(self.breakpoints, self.values, strict=True):
            if raw <= bp:
                return val
        return self.values[-1]

    @classmethod
    def fit(cls, scores: Sequence[float], labels: Sequence[int]) -> IsotonicCalibrator:
        pairs = sorted(zip(scores, labels, strict=True))
        blocks = [[s, float(y), 1.0] for s, y in pairs]
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][1] > blocks[i + 1][1]:
                s_hi = max(blocks[i][0], blocks[i + 1][0])
                w = blocks[i][2] + blocks[i + 1][2]
                mean = (
                    blocks[i][1] * blocks[i][2] + blocks[i + 1][1] * blocks[i + 1][2]
                ) / w
                blocks[i:i + 2] = [[s_hi, mean, w]]
                i = max(0, i - 1)
            else:
                i += 1
        return cls([b[0] for b in blocks], [b[1] for b in blocks])


@dataclass
class SegmentScore:
    span: tuple[int, int]
    score: float
    band: str
    words: int


@dataclass
class AIAnalysis:
    index: float
    band: str
    confidence_interval: tuple[float, float]
    segments: list[SegmentScore]
    features: FeatureVector
    calibrated: bool
    used_language_model: bool
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": round(self.index, 4),
            "band": self.band,
            "confidence_interval": [
                round(self.confidence_interval[0], 4),
                round(self.confidence_interval[1], 4),
            ],
            "calibrated": self.calibrated,
            "used_language_model": self.used_language_model,
            "segments": [
                {
                    "span": list(s.span),
                    "words": s.words,
                    "score": round(s.score, 4),
                    "band": s.band,
                }
                for s in self.segments
            ],
            "features": self.features.as_dict(),
            "caveats": self.caveats,
        }


def band_for(score: float) -> str:
    for upper, name in BANDS:
        if score < upper:
            return name
    return VERY_LIKELY_AI


@dataclass
class AIDetector:
    lm: LanguageModel | None = None
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    bias: float = DEFAULT_BIAS
    means: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MEANS))
    stds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_STDS))
    calibrator: IsotonicCalibrator = field(default_factory=IsotonicCalibrator)

    def raw_score(self, fv: FeatureVector) -> float:
        z = self.bias
        for name, weight in self.weights.items():
            if not fv.has_likelihood and name in (
                "mean_log_prob", "log_prob_std", "burstiness",
                "mean_log_rank", "curvature",
            ):
                continue
            value = getattr(fv, name, 0.0)
            mean = self.means.get(name, 0.0)
            std = self.stds.get(name, 1.0) or 1.0
            z += weight * (value - mean) / std
        return sigmoid(z)

    def features(self, doc: NormalizedDoc) -> FeatureVector:
        style = stylometry(doc)
        if self.lm is None:
            return style
        return merge_features(
            style, likelihood_features(doc.text, self.lm, doc.sentences)
        )

    def _segments(self, doc: NormalizedDoc) -> list[SegmentScore]:
        out: list[SegmentScore] = []
        tokens = doc.tokens
        step = SEGMENT_WORDS
        for start in range(0, len(tokens), step):
            chunk = tokens[start:start + step]
            if len(chunk) < MIN_SEGMENT_WORDS:
                if out:
                    break
                if not chunk:
                    break
            raw_text = doc.raw[chunk[0].start:chunk[-1].end]
            sub = normalize(raw_text)
            score = self.calibrator(self.raw_score(self.features(sub)))
            out.append(
                SegmentScore(
                    span=(chunk[0].start, chunk[-1].end),
                    score=score,
                    band=band_for(score),
                    words=len(chunk),
                )
            )
        return out

    def analyze(self, doc: NormalizedDoc) -> AIAnalysis:
        fv = self.features(doc)
        raw = self.raw_score(fv)
        index = self.calibrator(raw)
        segments = self._segments(doc)

        # Interval from the spread of segment scores, widened when the model
        # is uncalibrated or running without likelihood features. This is a
        # dispersion estimate, not a frequentist confidence interval, and it
        # is labelled as such in the report.
        if len(segments) >= 2:
            mean = sum(s.score for s in segments) / len(segments)
            sd = math.sqrt(
                sum((s.score - mean) ** 2 for s in segments) / (len(segments) - 1)
            )
            half = 1.96 * sd / math.sqrt(len(segments))
        else:
            half = 0.20
        if not self.calibrator.fitted:
            half += 0.15
        if not fv.has_likelihood:
            half += 0.10
        lo = max(0.0, index - half)
        hi = min(1.0, index + half)

        caveats = [
            "The AI-writing index is probabilistic and is not evidence of "
            "misconduct on its own.",
        ]
        if not self.calibrator.fitted:
            caveats.append(
                "Detector is running with placeholder weights and no isotonic "
                "calibration. Treat the number as ordinal, not as a probability."
            )
        if not fv.has_likelihood:
            caveats.append(
                "No reference language model was supplied, so the score uses "
                "stylometric features only and is substantially weaker."
            )
        if sum(s.words for s in segments) < 300:
            caveats.append(
                "Document is short. Detection accuracy degrades sharply below "
                "roughly 300 words."
            )

        return AIAnalysis(
            index=index,
            band=band_for(index),
            confidence_interval=(lo, hi),
            segments=segments,
            features=fv,
            calibrated=self.calibrator.fitted,
            used_language_model=fv.has_likelihood,
            caveats=caveats,
        )


# --------------------------------------------------------------------------
# reference adapter for a real language model
# --------------------------------------------------------------------------

@dataclass
class HuggingFaceLM:
    """Adapter for any causal LM from transformers.

    Kept out of the import path on purpose: the base engine must run without
    torch installed, so this class imports lazily and the pipeline falls back
    to stylometry when it is absent.
    """

    model_name: str = "gpt2-medium"
    device: str = "cpu"
    mask_ratio: float = 0.15

    def __post_init__(self) -> None:
        self._tok = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

    def token_logprobs(self, text: str) -> list[tuple[str, float, float]]:
        if not text.strip():
            return []
        self._load()
        import torch

        ids = self._tok(text, return_tensors="pt", truncation=True, max_length=1024)
        ids = {k: v.to(self.device) for k, v in ids.items()}
        with torch.no_grad():
            logits = self._model(**ids).logits[0]
        input_ids = ids["input_ids"][0]
        logprobs = torch.log_softmax(logits[:-1], dim=-1)
        targets = input_ids[1:]
        chosen = logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)
        ranks = (logprobs > chosen.unsqueeze(1)).sum(dim=1) + 1
        return [
            (
                self._tok.decode([int(t)]),
                float(lp),
                float(math.log(int(r))),
            )
            for t, lp, r in zip(targets, chosen, ranks, strict=True)
        ]

    def perturb(self, text: str, n: int = 8) -> list[str]:
        """Cheap word-dropout perturbation.

        The DetectGPT paper masks spans and infills them with T5. Dropping
        words is a weaker probe but needs no second model, and the curvature
        signal survives it well enough to be useful.
        """
        import random

        words = text.split()
        if len(words) < 20:
            return []
        rng = random.Random(7)
        out = []
        for _ in range(n):
            keep = [w for w in words if rng.random() > self.mask_ratio]
            if len(keep) > 10:
                out.append(" ".join(keep))
        return out
