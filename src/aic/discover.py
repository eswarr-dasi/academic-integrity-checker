"""Source discovery against free, no-key scholarly APIs.

The similarity engine can only find what is in its index, so an empty index
returns 0.0 percent and that number means nothing at all. This module fills
the index from openly queryable sources:

    OpenAlex   https://api.openalex.org   abstracts, rebuilt from the
                                          inverted index OpenAlex publishes
    Crossref   https://api.crossref.org   abstracts where the publisher
                                          deposited one, as JATS markup

What this is not: it is not web scale, and it is not the licensed reference
corpus a commercial originality service compares against. It indexes titles
and abstracts, not full text, so it can catch a reused abstract or a lifted
definition and will miss a paraphrased body paragraph completely. Treat a
match as a pointer to a source worth reading, and treat the absence of
matches as no information rather than as a clean result.

It also sends fragments of the submission to a third party in order to search
for them. That is a real disclosure, not a footnote: nothing here runs
offline, and it must not be pointed at someone else's paper without them
knowing. The browser page deliberately never calls it.
"""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .ingest import RawDocument
from .normalize import NormalizedDoc

OPENALEX = "https://api.openalex.org/works"
CROSSREF = "https://api.crossref.org/works"
USER_AGENT = (
    "academic-integrity-checker/0.1 "
    "(+https://github.com/eswarr-dasi/academic-integrity-checker)"
)
DEFAULT_TIMEOUT = 20.0

# Only used to keep query phrases from being all glue. Deliberately short:
# this is query construction, not linguistics.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can did do does for from had has have
    he her his how in into is it its may might must not of on or our she so
    such than that the their them then there these they this those to was
    we were what when where which who will with would you your
    """.split()
)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Candidate:
    """One retrieved record, ready to be added to a corpus index."""

    source_id: str
    title: str
    url: str
    text: str
    provider: str
    year: int | None = None
    open_access: bool = False

    def as_raw(self) -> RawDocument:
        return RawDocument(
            doc_id=self.source_id,
            text=self.text,
            source_name=self.title or self.source_id,
            media_type="text/plain",
        )


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    """Rebuild running text from an OpenAlex abstract_inverted_index."""
    if not inverted:
        return ""
    spots: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        spots.extend((int(position), word) for position in positions)
    spots.sort()
    return " ".join(word for _, word in spots)


def strip_jats(raw: str | None) -> str:
    """Flatten the JATS fragment Crossref returns into plain text."""
    if not raw:
        return ""
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", raw))).strip()


def query_phrases(doc: NormalizedDoc, count: int = 6, words: int = 6) -> list[str]:
    """Pick spread out, content bearing phrases to search for.

    Windows are scored by how many of their words are not glue, the best
    scoring windows win, and chosen windows may not overlap. Spreading them
    matters: six phrases from the same paragraph only ever find one source.
    """
    tokens = [token.text.lower() for token in doc.tokens]
    if not tokens:
        return []
    if len(tokens) <= words:
        return [" ".join(tokens)]
    scored = [
        (
            sum(
                1
                for word in tokens[start : start + words]
                if word not in STOPWORDS and len(word) > 3
            ),
            start,
        )
        for start in range(len(tokens) - words + 1)
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    chosen: list[int] = []
    for _, start in scored:
        if all(abs(start - taken) >= words for taken in chosen):
            chosen.append(start)
        if len(chosen) >= count:
            break
    chosen.sort()
    return [" ".join(tokens[start : start + words]) for start in chosen]


def _get(url: str, timeout: float = DEFAULT_TIMEOUT) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def search_openalex(
    query: str,
    per_page: int = 10,
    mailto: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[Candidate]:
    """Search OpenAlex. Passing mailto gets you their faster polite pool."""
    params = {
        "search": query,
        "per-page": str(max(1, min(per_page, 50))),
        "select": (
            "id,doi,title,publication_year,abstract_inverted_index,open_access"
        ),
    }
    if mailto:
        params["mailto"] = mailto
    payload = _get(OPENALEX + "?" + urllib.parse.urlencode(params), timeout)
    out: list[Candidate] = []
    for work in payload.get("results", []):
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        if not abstract:
            continue
        title = (work.get("title") or "").strip()
        access = work.get("open_access") or {}
        out.append(
            Candidate(
                source_id=work.get("doi") or work.get("id") or title,
                title=title,
                url=work.get("id") or "",
                text=(title + ". " + abstract) if title else abstract,
                provider="openalex",
                year=work.get("publication_year"),
                open_access=bool(access.get("is_oa")),
            )
        )
    return out


def search_crossref(
    query: str, rows: int = 10, timeout: float = DEFAULT_TIMEOUT
) -> list[Candidate]:
    """Search Crossref. Many records carry no abstract and are skipped."""
    params = {
        "query.bibliographic": query,
        "rows": str(max(1, min(rows, 50))),
        "select": "DOI,title,abstract,issued,URL",
    }
    payload = _get(CROSSREF + "?" + urllib.parse.urlencode(params), timeout)
    out: list[Candidate] = []
    for item in payload.get("message", {}).get("items", []):
        abstract = strip_jats(item.get("abstract"))
        if not abstract:
            continue
        titles = item.get("title") or [""]
        title = (titles[0] or "").strip()
        issued = (item.get("issued") or {}).get("date-parts") or [[None]]
        out.append(
            Candidate(
                source_id=item.get("DOI") or title,
                title=title,
                url=item.get("URL") or "",
                text=(title + ". " + abstract) if title else abstract,
                provider="crossref",
                year=issued[0][0] if issued and issued[0] else None,
            )
        )
    return out


def _key(candidate: Candidate) -> str:
    return candidate.source_id.lower().replace("https://doi.org/", "")


def discover(
    doc: NormalizedDoc,
    queries: int = 6,
    per_query: int = 10,
    mailto: str | None = None,
    providers: tuple[str, ...] = ("openalex", "crossref"),
    timeout: float = DEFAULT_TIMEOUT,
) -> list[Candidate]:
    """Retrieve candidate sources for one submission, deduplicated by DOI.

    Network failures on a single query are swallowed on purpose. Discovery is
    best effort: a flaky API should narrow the index, not abort the check.
    """
    found: dict[str, Candidate] = {}
    for phrase in query_phrases(doc, count=queries):
        if "openalex" in providers:
            try:
                for candidate in search_openalex(phrase, per_query, mailto, timeout):
                    found.setdefault(_key(candidate), candidate)
            except (OSError, ValueError):
                pass
        if "crossref" in providers:
            try:
                for candidate in search_crossref(phrase, per_query, timeout):
                    found.setdefault(_key(candidate), candidate)
            except (OSError, ValueError):
                pass
    return list(found.values())
