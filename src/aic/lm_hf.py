"""Reference LanguageModel adapter built on freely available weights.

The likelihood features (mean_log_prob, log_prob_std, burstiness,
mean_log_rank, curvature) are the ones that actually carry signal, and they
need a causal language model. This is the adapter the ai_detect docstring
promises: any freely downloadable causal model works, gpt2 by default because
it is small enough to run on a laptop CPU.

The arithmetic is deliberately separated from the tensor plumbing. Everything
in the first half of this module is plain Python over plain lists, so it can
be tested without installing a deep learning stack, and the class at the
bottom is a thin shim that does a forward pass and hands rows of logits over.

Install the optional extra before using the class:

    pip install "academic-integrity-checker[detect]"

Two honest warnings. First, the shipped calibration profile was fitted with
no language model attached, so switching one on changes the feature vector
that profile was built for; refit before quoting probabilities. Second, a
detector is only ever as good as the model behind it, and scoring modern
generated text with gpt2 measures how gpt2-like the text is, which is not
quite the same question.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

DEFAULT_MODEL = "gpt2"
DEFAULT_MASK_MODEL = "t5-small"
MAX_TOKENS = 1024
MISSING = (
    "HuggingFaceLM needs torch and transformers. Install the optional extra "
    "with: pip install \"academic-integrity-checker[detect]\""
)


def log_softmax(row: Sequence[float]) -> list[float]:
    """Numerically stable log softmax over one row of logits."""
    if not row:
        return []
    top = max(row)
    exps = [math.exp(value - top) for value in row]
    total = sum(exps)
    return [value - top - math.log(total) for value in row]


def rank_of(row: Sequence[float], target: int) -> int:
    """One based rank of target among the logits, highest logit first.

    Ties count against the target, so the rank is the pessimistic one. Rank is
    used rather than probability because it barely moves when a sampler runs
    at an unusual temperature.
    """
    chosen = row[target]
    return 1 + sum(1 for value in row if value > chosen)


def triples_from_rows(
    rows: Sequence[Sequence[float]],
    targets: Sequence[int],
    decode: Callable[[int], str],
) -> list[tuple[str, float, float]]:
    """Build (token, log_prob, log_rank) triples from next token logits.

    rows[i] are the logits predicting targets[i]. The caller is responsible
    for the off by one: row 0 of a causal model predicts token 1, so the
    first token of the text never gets a triple.
    """
    out: list[tuple[str, float, float]] = []
    for row, target in zip(rows, targets, strict=True):
        if not row or not 0 <= target < len(row):
            continue
        logps = log_softmax(row)
        out.append(
            (decode(target), logps[target], math.log(rank_of(row, target)))
        )
    return out


_WORD = re.compile(r"\S+")


def mask_spans(
    text: str, rate: float = 0.15, span_words: int = 2, rng: random.Random | None = None
) -> tuple[str, int]:
    """Replace a share of word spans with T5 sentinel tokens.

    Returns the masked text and how many sentinels were inserted. This is the
    DetectGPT perturbation recipe: machine text sits near a local maximum of
    the model log likelihood, so masking and refilling costs it more
    likelihood than it costs human text.
    """
    rng = rng or random.Random(0)
    words = _WORD.findall(text)
    if len(words) <= span_words:
        return text, 0
    slots = list(range(0, len(words) - span_words + 1, span_words))
    wanted = max(1, int(len(words) * rate / span_words))
    picked = sorted(rng.sample(slots, min(wanted, len(slots))))
    out: list[str] = []
    cursor = 0
    for index, start in enumerate(picked):
        if start < cursor:
            continue
        out.extend(words[cursor:start])
        out.append("<extra_id_" + str(index) + ">")
        cursor = start + span_words
    out.extend(words[cursor:])
    return " ".join(out), sum(1 for token in out if token.startswith("<extra_id_"))


@dataclass
class HuggingFaceLM:
    """Causal language model adapter satisfying the LanguageModel protocol.

    model and tokenizer can be injected, which is what the tests do. Left
    unset, they are loaded lazily on first use so that importing this module
    never pulls a model down.
    """

    model_name: str = DEFAULT_MODEL
    mask_model_name: str = DEFAULT_MASK_MODEL
    device: str = "cpu"
    max_tokens: int = MAX_TOKENS
    seed: int = 0
    model: object | None = None
    tokenizer: object | None = None
    mask_model: object | None = None
    mask_tokenizer: object | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - needs the extra
            raise RuntimeError(MISSING) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def token_logprobs(self, text: str) -> list[tuple[str, float, float]]:
        if not text.strip():
            return []
        self._load()
        import torch

        ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.max_tokens)["input_ids"]
        if ids.shape[-1] < 2:
            return []
        with torch.no_grad():
            logits = self.model(ids.to(self.device)).logits[0]
        rows = logits[:-1].float().tolist()
        targets = ids[0][1:].tolist()
        return triples_from_rows(rows, targets, self._decode)

    def _decode(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def perturb(self, text: str, n: int = 8) -> list[str]:
        """Mask and refill the passage n times with the sentinel model.

        Returns an empty list if the mask model is unavailable, which leaves
        curvature at zero rather than inventing a number for it.
        """
        try:
            self._load_mask()
        except RuntimeError:
            return []
        import torch

        out: list[str] = []
        for _ in range(max(0, n)):
            masked, sentinels = mask_spans(text, rng=self._rng)
            if not sentinels:
                continue
            batch = self.mask_tokenizer(masked, return_tensors="pt",
                                        truncation=True, max_length=self.max_tokens)
            with torch.no_grad():
                generated = self.mask_model.generate(
                    **batch, max_new_tokens=64, do_sample=True, top_p=0.95
                )
            fills = self.mask_tokenizer.decode(
                generated[0], skip_special_tokens=False
            )
            out.append(refill(masked, fills))
        return out

    def _load_mask(self) -> None:
        if self.mask_model is not None and self.mask_tokenizer is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - needs the extra
            raise RuntimeError(MISSING) from exc
        self.mask_tokenizer = AutoTokenizer.from_pretrained(self.mask_model_name)
        self.mask_model = AutoModelForSeq2SeqLM.from_pretrained(self.mask_model_name)
        self.mask_model.to(self.device)
        self.mask_model.eval()


_SENTINEL = re.compile(r"<extra_id_(\d+)>")


def refill(masked: str, generated: str) -> str:
    """Put the sentinel model output back into the masked passage.

    The model emits its answers as a sentinel delimited sequence, so the
    fragment after sentinel k is the replacement for sentinel k in the input.
    Missing answers collapse the sentinel away rather than leaving it in the
    text, where it would poison the likelihood of the perturbed sample.
    """
    pieces = _SENTINEL.split(generated)
    answers: dict[str, str] = {}
    for index in range(1, len(pieces) - 1, 2):
        answers[pieces[index]] = pieces[index + 1].strip()

    def swap(match: re.Match[str]) -> str:
        return answers.get(match.group(1), "")

    return re.sub(r"\s+", " ", _SENTINEL.sub(swap, masked)).strip()
