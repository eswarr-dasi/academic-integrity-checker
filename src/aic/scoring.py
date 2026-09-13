"""Turning matches into indices, with the exclusion policy made explicit.

The similarity index is defined here and nowhere else, because an index whose
definition is scattered across the code is an index nobody can defend in an
academic misconduct hearing.

    similarity_index = |union of matched scorable tokens| / |scorable tokens|

Three properties follow from that definition and are enforced by tests:

* It cannot exceed 1.0. Matched token positions are unioned, never summed, so
  a sentence found in twelve sources contributes its own length once.
* Excluded regions leave both numerator and denominator. A paper that is half
  correctly attributed block quotation is scored on the half that is the
  student's own work, not penalised for the quoting.
* Short matches are dropped before scoring, not after. Filtering after the
  fact would let sub-threshold noise inflate the denominator.

There is no "acceptable" index. The number is a pointer to spans a reader
should look at. Any institution that sets a numeric threshold for misconduct
is making a policy choice and should say so out loud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .normalize import NormalizedDoc
from .similarity import Match, apply_exclusions, covered_tokens

QUOTE = "quote"
CITATION = "citation"
BIBLIOGRAPHY = "bibliography"


@dataclass
class ScoringSettings:
    """Everything an instructor is allowed to change, in one place."""

    exclude_quotes: bool = True
    exclude_citations: bool = True
    exclude_bibliography: bool = True
    min_match_words: int = 8
    # Sources that are legitimately shared: the assignment brief, a lab
    # handout, a template, the student's own earlier draft.
    allowlist: set[str] = field(default_factory=set)

    def exclusion_kinds(self) -> set[str]:
        kinds: set[str] = set()
        if self.exclude_quotes:
            kinds.add(QUOTE)
        if self.exclude_citations:
            kinds.add(CITATION)
        if self.exclude_bibliography:
            kinds.add(BIBLIOGRAPHY)
        return kinds

    def to_dict(self) -> dict[str, object]:
        return {
            "exclude_quotes": self.exclude_quotes,
            "exclude_citations": self.exclude_citations,
            "exclude_bibliography": self.exclude_bibliography,
            "min_match_words": self.min_match_words,
            "allowlist": sorted(self.allowlist),
        }


@dataclass
class SourceBreakdown:
    source_id: str
    source_label: str
    words: int
    share: float
    best_similarity: float
    match_types: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_label": self.source_label,
            "words": self.words,
            "share": round(self.share, 4),
            "best_similarity": round(self.best_similarity, 4),
            "match_types": self.match_types,
        }


@dataclass
class SimilarityResult:
    index: float
    matched_words: int
    scorable_words: int
    total_words: int
    excluded_words: int
    matches: list[Match]
    by_source: list[SourceBreakdown]
    settings: ScoringSettings

    def to_dict(self) -> dict[str, object]:
        return {
            "index": round(self.index, 4),
            "matched_words": self.matched_words,
            "scorable_words": self.scorable_words,
            "total_words": self.total_words,
            "excluded_words": self.excluded_words,
            "sources": [s.to_dict() for s in self.by_source],
            "matches": [m.to_dict() for m in self.matches],
        }


def score_similarity(
    doc: NormalizedDoc,
    matches: Sequence[Match],
    settings: ScoringSettings | None = None,
) -> SimilarityResult:
    settings = settings or ScoringSettings()
    kinds = settings.exclusion_kinds()

    # 1. drop allowlisted sources and sub-threshold matches
    kept = [
        m
        for m in matches
        if m.source_id not in settings.allowlist
        and m.n_words >= settings.min_match_words
    ]

    # 2. mark matches that sit inside an excluded region
    kept = apply_exclusions(doc, kept, kinds)

    # 3. scorable denominator
    mask = doc.scorable_mask(kinds) if kinds else [True] * len(doc.tokens)
    scorable_positions = {i for i, ok in enumerate(mask) if ok}
    scorable = len(scorable_positions)
    total = len(doc.tokens)

    # 4. union the numerator, intersected with the scorable set
    matched_positions = covered_tokens(kept) & scorable_positions
    matched = len(matched_positions)
    index = (matched / scorable) if scorable else 0.0

    return SimilarityResult(
        index=index,
        matched_words=matched,
        scorable_words=scorable,
        total_words=total,
        excluded_words=total - scorable,
        matches=list(kept),
        by_source=_breakdown(kept, matched_positions, scorable),
        settings=settings,
    )


def _breakdown(
    matches: Iterable[Match], matched_positions: set[int], scorable: int
) -> list[SourceBreakdown]:
    """Per source contribution.

    Shares are computed against the same union used for the index, so they sum
    to the index and not to something larger. Overlap between two sources is
    attributed to whichever match claimed the tokens during merging.
    """
    grouped: dict[str, list[Match]] = {}
    for m in matches:
        if m.excluded:
            continue
        grouped.setdefault(m.source_id, []).append(m)

    claimed: set[int] = set()
    out: list[SourceBreakdown] = []
    for source_id, group in grouped.items():
        positions: set[int] = set()
        for m in group:
            positions.update(range(m.query_tokens[0], m.query_tokens[1] + 1))
        positions &= matched_positions
        positions -= claimed
        claimed |= positions
        if not positions:
            continue
        out.append(
            SourceBreakdown(
                source_id=source_id,
                source_label=group[0].source_label,
                words=len(positions),
                share=(len(positions) / scorable) if scorable else 0.0,
                best_similarity=max(m.similarity for m in group),
                match_types=sorted({m.match_type for m in group}),
            )
        )
    out.sort(key=lambda s: -s.words)
    return out


def review_priority(similarity_index: float, ai_index: float, ai_band: str) -> str:
    """A triage hint for a marker with eighty papers and one afternoon.

    This orders a queue. It does not decide anything, and the wording avoids
    any claim about the student, because the tool has no way to distinguish
    plagiarism from sloppy citation from a false positive.
    """
    if similarity_index >= 0.30:
        return "read-first"
    if ai_band in ("likely-ai", "very-likely-ai") and similarity_index >= 0.10:
        return "read-first"
    if similarity_index >= 0.15 or ai_index >= 0.75:
        return "worth-a-look"
    return "no-flags"
