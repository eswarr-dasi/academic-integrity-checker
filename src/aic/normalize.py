"""Normalization stage: fold the text, keep the offsets, tag the exclusions.

Every transformation in this module is offset preserving. The normalized token
stream carries (start, end) character offsets into the ORIGINAL document text,
so a match discovered on folded text can still be highlighted in the source
PDF without a second search.

Three taggers run after folding. Their spans are not deleted, they are marked,
because whether a quotation counts toward the similarity index is a policy
decision that belongs to the instructor and not to the tokenizer.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# character level folding tables
# --------------------------------------------------------------------------

# Zero width and invisible characters. Inserting these between words is the
# cheapest way to break n-gram matching, so they are stripped first.
ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff\u00ad"

# Cyrillic and Greek lookalikes for Latin letters. Substituting these is the
# second cheapest evasion and is invisible to a human reader.
HOMOGLYPHS = {
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x", "\u0456": "i",
    "\u0458": "j", "\u0455": "s",
    "\u0410": "A", "\u0415": "E", "\u041e": "O", "\u0420": "P",
    "\u0421": "C", "\u0423": "Y", "\u0425": "X", "\u0406": "I",
    "\u03bf": "o", "\u03b1": "a", "\u03b5": "e", "\u03c1": "p",
    "\u03bd": "v", "\u0392": "B", "\u039f": "O", "\u0399": "I",
}

SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2039": "'", "\u203a": "'",
}

DASHES = {ch: "-" for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"}

LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\u00e6": "ae", "\u0153": "oe",
}

_FOLD: dict[str, str] = {}
_FOLD.update(HOMOGLYPHS)
_FOLD.update(SMART_QUOTES)
_FOLD.update(DASHES)
_FOLD.update(LIGATURES)


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    """One word of the folded text, with a pointer back into the raw text."""

    text: str
    start: int
    end: int
    index: int


@dataclass(frozen=True)
class Span:
    """A half open character range in the raw text, with a label."""

    start: int
    end: int
    kind: str = ""

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class NormalizedDoc:
    raw: str
    text: str
    offsets: list[int] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)
    sentences: list[Span] = field(default_factory=list)
    exclusions: list[Span] = field(default_factory=list)

    @property
    def words(self) -> list[str]:
        return [t.text for t in self.tokens]

    def raw_span(self, first_token: int, last_token: int) -> Span:
        """Raw character span covering tokens [first_token, last_token]."""
        return Span(self.tokens[first_token].start, self.tokens[last_token].end)

    def scorable_mask(self, kinds: set[str] | None = None) -> list[bool]:
        """Per token flag: does this word count toward the similarity index.

        Tokens inside an excluded span (quotation, citation, bibliography) are
        dropped from the denominator as well as the numerator, which is what
        keeps a correctly quoted paper from scoring like a copied one.
        """
        active = [s for s in self.exclusions if kinds is None or s.kind in kinds]
        mask = [True] * len(self.tokens)
        for i, tok in enumerate(self.tokens):
            tok_span = Span(tok.start, tok.end)
            if any(s.overlaps(tok_span) for s in active):
                mask[i] = False
        return mask


# --------------------------------------------------------------------------
# folding
# --------------------------------------------------------------------------

def fold(raw: str) -> tuple[str, list[int]]:
    """Return (folded_text, offsets) where offsets[i] indexes into raw.

    Folding runs character by character rather than through str.translate so
    that one-to-many expansions (ligatures) and one-to-zero deletions (zero
    width characters) both keep the offset table honest.
    """
    out: list[str] = []
    offsets: list[int] = []
    for i, ch in enumerate(raw):
        if ch in ZERO_WIDTH:
            continue
        mapped = _FOLD.get(ch)
        if mapped is None:
            decomposed = unicodedata.normalize("NFKD", ch)
            mapped = "".join(c for c in decomposed if not unicodedata.combining(c))
            if not mapped:
                continue
        for out_ch in mapped:
            out.append(out_ch)
            offsets.append(i)
    return "".join(out), offsets


_HYPHEN_BREAK = re.compile(r"-[ \t]*\r?\n[ \t]*(?=[a-z])")


def dehyphenate(text: str, offsets: list[int]) -> tuple[str, list[int]]:
    """Rejoin words that a typesetter split across a line break.

    PDF extraction routinely yields know- / ledge on two lines. Without this
    step the shingles around every line break in a two column paper are
    garbage and recall drops for no good reason.
    """
    drop: set[int] = set()
    for m in _HYPHEN_BREAK.finditer(text):
        drop.update(range(m.start(), m.end()))
    if not drop:
        return text, offsets
    kept = [(c, o) for i, (c, o) in enumerate(zip(text, offsets)) if i not in drop]
    return "".join(c for c, _ in kept), [o for _, o in kept]


# --------------------------------------------------------------------------
# tokenization and sentence segmentation
# --------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "al",
    "e.g", "i.e", "cf", "fig", "eq", "no", "vol", "pp", "ed", "eds",
}

_SENT_END = re.compile(r"""[.!?]["')\]]*(?=\s|$)""")


def tokenize(text: str, offsets: list[int]) -> list[Token]:
    tokens: list[Token] = []
    for m in _WORD.finditer(text):
        start = offsets[m.start()]
        end = offsets[m.end() - 1] + 1
        tokens.append(Token(m.group(0).casefold(), start, end, len(tokens)))
    return tokens


def split_sentences(text: str, offsets: list[int]) -> list[Span]:
    """Cheap but adequate sentence splitter with an abbreviation guard.

    Sentence boundaries matter to the AI detector, which measures the variance
    of per-sentence perplexity. A splitter that fires inside et al. 2019
    inflates that variance and biases the score toward human.
    """
    spans: list[Span] = []
    start = 0
    for m in _SENT_END.finditer(text):
        head = text[max(0, m.start() - 12):m.start()].split()
        last = head[-1].casefold().rstrip(".") if head else ""
        if last in _ABBREV or (len(last) == 1 and last.isalpha()):
            continue
        end = m.end()
        if end > start and text[start:end].strip():
            spans.append(Span(offsets[start], offsets[end - 1] + 1, "sentence"))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append(Span(offsets[start], offsets[len(text) - 1] + 1, "sentence"))
    return spans


# --------------------------------------------------------------------------
# exclusion taggers
# --------------------------------------------------------------------------

_INLINE_QUOTE = re.compile(r'"([^"]{8,600})"')
_BLOCK_QUOTE = re.compile(r"(?:^|\n)(?:[ ]{4,}|\t)(\S[^\n]{20,})", re.MULTILINE)

_CITATION = re.compile(
    r"\((?:[A-Z][A-Za-z'-]+(?:\s+(?:et\s+al\.?|and|&)\s+[A-Z][A-Za-z'-]+)?"
    r"(?:,\s*)?(?:19|20)\d{2}[a-z]?(?:,\s*(?:p{1,2}\.?\s*)?\d+(?:-\d+)?)?)\)"
    r"|\[\d{1,3}(?:\s*[,-]\s*\d{1,3})*\]"
)

_BIB_HEADING = re.compile(
    r"^\s*(?:references|bibliography|works\s+cited|literature\s+cited)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def tag_quotes(text: str, offsets: list[int]) -> list[Span]:
    spans: list[Span] = []
    for pattern in (_INLINE_QUOTE, _BLOCK_QUOTE):
        for m in pattern.finditer(text):
            s, e = m.span(1)
            spans.append(Span(offsets[s], offsets[e - 1] + 1, "quote"))
    return spans


def tag_citations(text: str, offsets: list[int]) -> list[Span]:
    return [
        Span(offsets[m.start()], offsets[m.end() - 1] + 1, "citation")
        for m in _CITATION.finditer(text)
    ]


def tag_bibliography(text: str, offsets: list[int]) -> list[Span]:
    """Everything from the last reference heading to the end of the document.

    A reference list is a dense pile of other people's words by definition.
    Leaving it in the denominator makes the index meaningless.
    """
    matches = list(_BIB_HEADING.finditer(text))
    if not matches:
        return []
    start = matches[-1].start()
    return [Span(offsets[start], offsets[len(text) - 1] + 1, "bibliography")]


def merge_spans(spans: list[Span]) -> list[Span]:
    """Merge overlapping spans of the same kind so nothing is counted twice."""
    by_kind: dict[str, list[Span]] = {}
    for s in spans:
        by_kind.setdefault(s.kind, []).append(s)
    merged: list[Span] = []
    for kind, group in by_kind.items():
        group.sort(key=lambda s: (s.start, s.end))
        cur = group[0]
        for nxt in group[1:]:
            if nxt.start <= cur.end:
                cur = Span(cur.start, max(cur.end, nxt.end), kind)
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
    merged.sort(key=lambda s: (s.start, s.end))
    return merged


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def normalize(raw: str) -> NormalizedDoc:
    """Fold, dehyphenate, tokenize, segment and tag a raw document string."""
    text, offsets = fold(raw)
    text, offsets = dehyphenate(text, offsets)
    tokens = tokenize(text, offsets)
    sentences = split_sentences(text, offsets)
    exclusions = merge_spans(
        tag_quotes(text, offsets)
        + tag_citations(text, offsets)
        + tag_bibliography(text, offsets)
    )
    return NormalizedDoc(
        raw=raw,
        text=text,
        offsets=offsets,
        tokens=tokens,
        sentences=sentences,
        exclusions=exclusions,
    )
