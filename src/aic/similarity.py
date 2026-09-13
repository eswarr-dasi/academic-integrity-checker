"""Match extraction: seed, chain, align, merge.

Retrieval gives candidate documents. This module turns candidates into spans a
human can read, because "17% similar to 43 sources" is not actionable and
"these 281 words appear in this paper at this offset" is.

Pipeline per candidate:

1. seed      - fingerprint hashes shared by query and source give
               (query_pos, source_pos) anchor pairs for free from the index.
2. chain     - anchors on the same diagonal (query_pos - source_pos roughly
               constant) belong to one copied passage. Chain them with a gap
               tolerance so a swapped word does not split a match in two.
3. align     - Smith-Waterman local alignment inside the chained region gives
               exact boundaries and a real similarity ratio, which n-gram
               overlap alone cannot.
4. merge     - overlapping query spans are merged before scoring, otherwise a
               sentence found in five sources would be counted five times and
               the index could exceed 100%.

A separate paraphrase pass covers text that shares no n-grams at all, using
sentence embeddings. It is deliberately run last and reported with its own
match type, because semantic similarity is much weaker evidence than a shared
eight word phrase and should never be presented as if it were the same thing.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .fingerprint import DocumentFingerprint
from .index import Candidate, CorpusIndex
from .normalize import NormalizedDoc, Span

VERBATIM = "verbatim"
NEAR_DUPLICATE = "near-duplicate"
PARAPHRASE = "paraphrase"

MIN_MATCH_WORDS = 8
MAX_SEED_GAP = 12
SW_MATCH = 2.0
SW_MISMATCH = -1.0
SW_GAP = -1.5


class TokenSource(Protocol):
    """Anything that can return the token stream of an indexed document."""

    def tokens(self, doc_id: str) -> list[str]:
        ...


@dataclass
class DictTokenSource:
    """Reference TokenSource. Replace with object storage for a real corpus."""

    data: Mapping[str, list[str]] = field(default_factory=dict)

    def tokens(self, doc_id: str) -> list[str]:
        return list(self.data.get(doc_id, []))


@dataclass
class Match:
    """One attributable overlap between the submission and one source."""

    source_id: str
    source_label: str
    match_type: str
    similarity: float
    query_tokens: tuple[int, int]
    source_tokens: tuple[int, int]
    query_chars: tuple[int, int]
    excluded: bool = False
    exclusion_reason: str | None = None

    @property
    def n_words(self) -> int:
        return self.query_tokens[1] - self.query_tokens[0] + 1

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_label": self.source_label,
            "match_type": self.match_type,
            "similarity": round(self.similarity, 4),
            "words": self.n_words,
            "submission_tokens": list(self.query_tokens),
            "submission_span": list(self.query_chars),
            "source_tokens": list(self.source_tokens),
            "excluded": self.excluded,
            "exclusion_reason": self.exclusion_reason,
        }


# --------------------------------------------------------------------------
# 2. chaining
# --------------------------------------------------------------------------

@dataclass
class Chain:
    q_start: int
    q_end: int
    s_start: int
    s_end: int
    anchors: int = 1

    def widen(self, q: int, s: int) -> None:
        self.q_start = min(self.q_start, q)
        self.q_end = max(self.q_end, q)
        self.s_start = min(self.s_start, s)
        self.s_end = max(self.s_end, s)
        self.anchors += 1


def chain_seeds(seeds: Sequence[tuple[int, int]], max_gap: int = MAX_SEED_GAP) -> list[Chain]:
    """Group anchor pairs into candidate passages.

    Two anchors join the same chain when their diagonals agree within the gap
    tolerance and they are close in the query. The tolerance is what lets a
    single substituted synonym inside an otherwise copied paragraph stay one
    match instead of becoming two short ones that both fall under the minimum
    length and get discarded.
    """
    chains: list[Chain] = []
    for q, s in sorted(seeds):
        placed = False
        for ch in chains:
            same_diagonal = abs((q - s) - (ch.q_start - ch.s_start)) <= max_gap
            near_in_query = q - ch.q_end <= max_gap
            if same_diagonal and near_in_query:
                ch.widen(q, s)
                placed = True
                break
        if not placed:
            chains.append(Chain(q, q, s, s))
    return chains


# --------------------------------------------------------------------------
# 3. local alignment
# --------------------------------------------------------------------------

def smith_waterman(
    a: Sequence[str],
    b: Sequence[str],
    match: float = SW_MATCH,
    mismatch: float = SW_MISMATCH,
    gap: float = SW_GAP,
) -> tuple[int, int, int, int, float]:
    """Best local alignment of two token sequences.

    Returns (a_start, a_end, b_start, b_end, identity) with inclusive ends and
    identity as the fraction of aligned positions that matched exactly.
    Quadratic, which is fine because it only ever runs on a chained region of
    a few hundred tokens, never on whole documents.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return (0, -1, 0, -1, 0.0)

    prev = [0.0] * (m + 1)
    best = 0.0
    best_cell = (0, 0)
    # traceback pointers, 0 none, 1 diagonal, 2 up, 3 left
    ptr = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        cur = [0.0] * (m + 1)
        for j in range(1, m + 1):
            diag = prev[j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            up = prev[j] + gap
            left = cur[j - 1] + gap
            score = max(0.0, diag, up, left)
            cur[j] = score
            if score == 0.0:
                ptr[i][j] = 0
            elif score == diag:
                ptr[i][j] = 1
            elif score == up:
                ptr[i][j] = 2
            else:
                ptr[i][j] = 3
            if score > best:
                best = score
                best_cell = (i, j)
        prev = cur

    if best == 0.0:
        return (0, -1, 0, -1, 0.0)

    i, j = best_cell
    aligned = 0
    matched = 0
    while i > 0 and j > 0 and ptr[i][j] != 0:
        move = ptr[i][j]
        if move == 1:
            aligned += 1
            if a[i - 1] == b[j - 1]:
                matched += 1
            i -= 1
            j -= 1
        elif move == 2:
            aligned += 1
            i -= 1
        else:
            aligned += 1
            j -= 1
    identity = matched / aligned if aligned else 0.0
    return (i, best_cell[0] - 1, j, best_cell[1] - 1, identity)


def classify(identity: float) -> str:
    if identity >= 0.98:
        return VERBATIM
    return NEAR_DUPLICATE


# --------------------------------------------------------------------------
# 4. merging
# --------------------------------------------------------------------------

def merge_matches(matches: Iterable[Match]) -> list[Match]:
    """Collapse overlapping chains within a source, keep every source.

    Two rules pull in opposite directions here, so both are stated.

    Within one source, two overlapping chains are the same copied passage
    found twice, so only the stronger survives.

    Across sources, overlap is preserved. A sentence that appears in five
    papers is one copied sentence with five possible origins, and a reader
    deciding whether this is plagiarism or a shared textbook definition needs
    to see all five. Double counting is prevented in aic.scoring, which unions
    token positions instead of summing match lengths, and not by discarding
    evidence at this stage.
    """
    kept: list[Match] = []
    claimed: dict[str, list[tuple[int, int]]] = {}
    for m in sorted(matches, key=lambda m: (-m.n_words, -m.similarity)):
        q0, q1 = m.query_tokens
        seen = claimed.setdefault(m.source_id, [])
        overlap = sum(max(0, min(q1, c1) - max(q0, c0) + 1) for c0, c1 in seen)
        if overlap > 0.5 * m.n_words:
            continue
        kept.append(m)
        seen.append((q0, q1))
    kept.sort(key=lambda m: (m.query_tokens[0], -m.n_words))
    return kept


def covered_tokens(matches: Iterable[Match]) -> set[int]:
    out: set[int] = set()
    for m in matches:
        if not m.excluded:
            out.update(range(m.query_tokens[0], m.query_tokens[1] + 1))
    return out


# --------------------------------------------------------------------------
# paraphrase pass
# --------------------------------------------------------------------------

class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        ...


@dataclass
class LexicalEmbedder:
    """Dependency free fallback: hashed bag of words with sublinear tf.

    Real paraphrase detection wants a sentence transformer. This exists so the
    pipeline runs, and so the tests can assert behaviour without downloading
    500 MB of weights. It catches word-order changes and synonym-free
    rewrites, and nothing more, so do not report its output as semantic
    similarity in production.
    """

    dim: int = 512

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in text.split():
                vec[hash_word(word) % self.dim] += 1.0
            vec = [1.0 + math.log(v) if v > 0 else 0.0 for v in vec]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def hash_word(word: str) -> int:
    h = 2166136261
    for ch in word:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def find_matches(
    doc: NormalizedDoc,
    fp: DocumentFingerprint,
    index: CorpusIndex,
    store: TokenSource,
    candidates: Sequence[Candidate] | None = None,
    min_words: int = MIN_MATCH_WORDS,
    pad: int = 24,
) -> list[Match]:
    """Seed, chain, align and merge every candidate into a list of matches."""
    query_tokens = doc.words
    cands = list(candidates if candidates is not None else index.candidates(fp))
    raw: list[Match] = []

    for cand in cands:
        source_tokens = store.tokens(cand.doc_id)
        if not source_tokens:
            continue
        for ch in chain_seeds(index.seeds(fp, cand.doc_id)):
            q0 = max(0, ch.q_start - pad)
            q1 = min(len(query_tokens), ch.q_end + fp.k + pad)
            s0 = max(0, ch.s_start - pad)
            s1 = min(len(source_tokens), ch.s_end + fp.k + pad)

            qa, qb, sa, sb, identity = smith_waterman(
                query_tokens[q0:q1], source_tokens[s0:s1]
            )
            if qb < qa:
                continue
            first, last = q0 + qa, q0 + qb
            if last - first + 1 < min_words:
                continue
            raw.append(
                Match(
                    source_id=cand.doc_id,
                    source_label=cand.meta.label(),
                    match_type=classify(identity),
                    similarity=identity,
                    query_tokens=(first, last),
                    source_tokens=(s0 + sa, s0 + sb),
                    query_chars=(
                        doc.tokens[first].start,
                        doc.tokens[last].end,
                    ),
                )
            )

    return merge_matches(raw)


def apply_exclusions(
    doc: NormalizedDoc,
    matches: Sequence[Match],
    kinds: set[str],
    overlap_ratio: float = 0.8,
) -> list[Match]:
    """Mark, do not delete, matches that fall inside an excluded region.

    Keeping them in the report with a reason is the difference between a tool
    an instructor trusts and one that silently hides evidence. A match is only
    excluded when most of it sits inside the excluded span, so a paragraph
    that merely contains a citation is still reported.
    """
    active = [s for s in doc.exclusions if s.kind in kinds]
    out: list[Match] = []
    for m in matches:
        span = Span(*m.query_chars)
        width = max(1, span.end - span.start)
        best_kind = None
        best_cover = 0.0
        for ex in active:
            cover = max(0, min(span.end, ex.end) - max(span.start, ex.start)) / width
            if cover > best_cover:
                best_cover, best_kind = cover, ex.kind
        if best_cover >= overlap_ratio and best_kind:
            out.append(
                Match(
                    **{
                        **m.__dict__,
                        "excluded": True,
                        "exclusion_reason": best_kind,
                    }
                )
            )
        else:
            out.append(m)
    return out
