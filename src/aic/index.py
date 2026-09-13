"""Corpus index: banded LSH plus a fingerprint posting list.

Two structures are kept side by side because they answer different questions,
and a checker that only has one of them fails in a predictable way.

* The banded LSH tables answer "which documents resemble this one as a whole",
  in roughly constant time, without touching most of the corpus.
* The fingerprint posting list answers "which documents share this exact
  phrase, and where". That is the question coursework actually poses: a paper
  with three copied paragraphs and twelve original ones has low whole-document
  Jaccard against the source it stole from, so LSH alone would never surface
  it. Postings also give aic.similarity its alignment seeds for free.

Banding is the usual trade. With a MinHash signature of n slots split into b
bands of r rows, two documents collide in at least one band with probability
1 - (1 - s**r)**b for true Jaccard s. That is an S-curve whose knee sits near
(1/b)**(1/r), so the band layout is derived from the recall threshold rather
than picked by feel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .fingerprint import (
    DEFAULT_K,
    DEFAULT_W,
    DocumentFingerprint,
    estimate_jaccard,
)


@dataclass
class SourceMeta:
    """Provenance for one corpus document. Every match must be attributable."""

    doc_id: str
    title: str = ""
    author: str = ""
    url: str = ""
    kind: str = "corpus"  # corpus | student | web | internal
    year: int | None = None
    n_tokens: int = 0

    def label(self) -> str:
        return self.title or self.url or self.doc_id


@dataclass
class Candidate:
    doc_id: str
    estimated_jaccard: float
    shared_fingerprints: int
    meta: SourceMeta
    via: str = "lsh"  # lsh | postings | both


def choose_bands(n_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) whose S-curve knee is closest to threshold.

    Searches every exact factorization of n_perm, so the returned layout uses
    the whole signature and no slot is wasted.
    """
    best: tuple[float, int, int] | None = None
    for rows in range(1, n_perm + 1):
        if n_perm % rows:
            continue
        bands = n_perm // rows
        knee = (1.0 / bands) ** (1.0 / rows)
        err = abs(knee - threshold)
        if best is None or err < best[0]:
            best = (err, bands, rows)
    assert best is not None
    return best[1], best[2]


@dataclass
class CorpusIndex:
    """In-memory reference index. Swap the dicts for Redis or FAISS at scale."""

    k: int = DEFAULT_K
    w: int = DEFAULT_W
    n_perm: int = 128
    threshold: float = 0.10
    min_shared_fingerprints: int = 2

    bands: int = field(init=False)
    rows: int = field(init=False)

    meta: dict[str, SourceMeta] = field(default_factory=dict)
    signatures: dict[str, list[int]] = field(default_factory=dict)
    buckets: dict[str, set[str]] = field(default_factory=dict)
    postings: dict[int, list[tuple[str, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bands, self.rows = choose_bands(self.n_perm, self.threshold)

    # -- writing ----------------------------------------------------------

    def _band_keys(self, signature: list[int]) -> list[str]:
        keys = []
        for b in range(self.bands):
            chunk = signature[b * self.rows:(b + 1) * self.rows]
            keys.append(str(b) + ":" + ",".join(str(x) for x in chunk))
        return keys

    def add(self, fp: DocumentFingerprint, meta: SourceMeta | None = None) -> None:
        if fp.k != self.k or fp.w != self.w:
            raise ValueError(
                "fingerprint built with k=%d w=%d cannot enter an index built "
                "with k=%d w=%d" % (fp.k, fp.w, self.k, self.w)
            )
        doc_id = fp.doc_id
        self.meta[doc_id] = meta or SourceMeta(doc_id=doc_id, n_tokens=fp.n_tokens)
        self.meta[doc_id].n_tokens = fp.n_tokens
        self.signatures[doc_id] = fp.signature
        for key in self._band_keys(fp.signature):
            self.buckets.setdefault(key, set()).add(doc_id)
        for f in fp.fingerprints:
            self.postings.setdefault(f.value, []).append((doc_id, f.pos))

    def remove(self, doc_id: str) -> None:
        """Drop a document. Needed for the student-paper takedown path: a
        repository that cannot forget is a liability, not a feature."""
        sig = self.signatures.pop(doc_id, None)
        self.meta.pop(doc_id, None)
        if sig is not None:
            for key in self._band_keys(sig):
                bucket = self.buckets.get(key)
                if bucket:
                    bucket.discard(doc_id)
                    if not bucket:
                        del self.buckets[key]
        for value in list(self.postings):
            kept = [p for p in self.postings[value] if p[0] != doc_id]
            if kept:
                self.postings[value] = kept
            else:
                del self.postings[value]

    # -- reading ----------------------------------------------------------

    def candidates(
        self,
        fp: DocumentFingerprint,
        limit: int = 50,
        exclude: set[str] | None = None,
        min_shared: int | None = None,
    ) -> list[Candidate]:
        """Union of the two retrieval paths, ranked by strength of evidence.

        Ranking puts shared fingerprint count ahead of estimated Jaccard on
        purpose. A source that shares forty selected fingerprints with the
        submission is worth aligning even if the two documents look nothing
        alike overall, which is exactly the partial-copy case.
        """
        exclude = exclude or set()
        floor = self.min_shared_fingerprints if min_shared is None else min_shared

        lsh_ids: set[str] = set()
        for key in self._band_keys(fp.signature):
            lsh_ids |= self.buckets.get(key, set())

        shared = self.shared_counts(fp)
        posting_ids = {d for d, n in shared.items() if n >= floor}

        hit_ids = (lsh_ids | posting_ids) - exclude
        hit_ids.discard(fp.doc_id)

        out: list[Candidate] = []
        for doc_id in hit_ids:
            if doc_id not in self.signatures:
                continue
            in_lsh = doc_id in lsh_ids
            in_post = doc_id in posting_ids
            out.append(
                Candidate(
                    doc_id=doc_id,
                    estimated_jaccard=estimate_jaccard(
                        fp.signature, self.signatures[doc_id]
                    ),
                    shared_fingerprints=shared.get(doc_id, 0),
                    meta=self.meta[doc_id],
                    via="both" if in_lsh and in_post else ("lsh" if in_lsh else "postings"),
                )
            )
        out.sort(key=lambda c: (-c.shared_fingerprints, -c.estimated_jaccard))
        return out[:limit]

    def shared_counts(self, fp: DocumentFingerprint) -> dict[str, int]:
        """How many selected fingerprints each corpus document has in common."""
        counts: dict[str, int] = {}
        for value in fp.values:
            for doc_id, _pos in self.postings.get(value, ()):
                counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

    def seeds(self, fp: DocumentFingerprint, doc_id: str) -> list[tuple[int, int]]:
        """Matching (query_pos, source_pos) pairs to seed local alignment."""
        pairs: list[tuple[int, int]] = []
        for f in fp.fingerprints:
            for other_id, pos in self.postings.get(f.value, ()):
                if other_id == doc_id:
                    pairs.append((f.pos, pos))
        pairs.sort()
        return pairs

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(
                {
                    "k": self.k,
                    "w": self.w,
                    "n_perm": self.n_perm,
                    "threshold": self.threshold,
                    "min_shared_fingerprints": self.min_shared_fingerprints,
                    "bands": self.bands,
                    "rows": self.rows,
                },
                indent=2,
            )
        )
        (path / "meta.json").write_text(
            json.dumps({d: asdict(m) for d, m in self.meta.items()}, indent=2)
        )
        (path / "signatures.json").write_text(json.dumps(self.signatures))
        (path / "postings.json").write_text(
            json.dumps({str(v): p for v, p in self.postings.items()})
        )

    @classmethod
    def load(cls, path: str | Path) -> "CorpusIndex":
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        idx = cls(
            k=cfg["k"],
            w=cfg["w"],
            n_perm=cfg["n_perm"],
            threshold=cfg["threshold"],
            min_shared_fingerprints=cfg.get("min_shared_fingerprints", 2),
        )
        idx.meta = {
            d: SourceMeta(**m)
            for d, m in json.loads((path / "meta.json").read_text()).items()
        }
        idx.signatures = json.loads((path / "signatures.json").read_text())
        idx.postings = {
            int(v): [(d, int(p)) for d, p in pairs]
            for v, pairs in json.loads((path / "postings.json").read_text()).items()
        }
        for doc_id, sig in idx.signatures.items():
            for key in idx._band_keys(sig):
                idx.buckets.setdefault(key, set()).add(doc_id)
        return idx

    def __len__(self) -> int:
        return len(self.signatures)

    def stats(self) -> dict[str, object]:
        return {
            "documents": len(self.signatures),
            "distinct_fingerprints": len(self.postings),
            "buckets": len(self.buckets),
            "bands": self.bands,
            "rows": self.rows,
            "lsh_knee": round((1.0 / self.bands) ** (1.0 / self.rows), 4),
            "min_shared_fingerprints": self.min_shared_fingerprints,
        }
"""Corpus index: banded LSH for candidate retrieval, plus a posting list.

Two structures are kept side by side because they answer different questions.

* The banded LSH tables answer "which documents are worth looking at", in
  roughly constant time, without touching most of the corpus.
* The fingerprint posting list answers "where exactly", by mapping a shingle
  hash to every (document, token position) that selected it. That is what
  lets aic.similarity seed its alignment instead of scanning.

Banding is the usual trade. With a MinHash signature of n slots split into b
bands of r rows, two documents collide in at least one band with probability
1 - (1 - s**r)**b for true Jaccard s. That is an S-curve whose knee sits near
(1/b)**(1/r), so the band layout is chosen from the recall threshold rather
than picked by feel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .fingerprint import (
    DEFAULT_K,
    DEFAULT_W,
    DocumentFingerprint,
    Fingerprint,
    estimate_jaccard,
)


@dataclass
class SourceMeta:
    """Provenance for one corpus document. Every match must be attributable."""

    doc_id: str
    title: str = ""
    author: str = ""
    url: str = ""
    kind: str = "corpus"  # corpus | student | web | internal
    year: int | None = None
    n_tokens: int = 0

    def label(self) -> str:
        return self.title or self.url or self.doc_id


@dataclass
class Candidate:
    doc_id: str
    estimated_jaccard: float
    shared_fingerprints: int
    meta: SourceMeta


def choose_bands(n_perm: int, threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) whose S-curve knee is closest to threshold.

    Searches every exact factorization of n_perm, so the returned layout uses
    the whole signature and no slot is wasted.
    """
    best: tuple[float, int, int] | None = None
    for rows in range(1, n_perm + 1):
        if n_perm % rows:
            continue
        bands = n_perm // rows
        knee = (1.0 / bands) ** (1.0 / rows)
        err = abs(knee - threshold)
        if best is None or err < best[0]:
            best = (err, bands, rows)
    assert best is not None
    return best[1], best[2]


@dataclass
class CorpusIndex:
    """In-memory reference index. Swap the dicts for Redis or FAISS at scale."""

    k: int = DEFAULT_K
    w: int = DEFAULT_W
    n_perm: int = 128
    threshold: float = 0.10

    bands: int = field(init=False)
    rows: int = field(init=False)

    meta: dict[str, SourceMeta] = field(default_factory=dict)
    signatures: dict[str, list[int]] = field(default_factory=dict)
    buckets: dict[str, set[str]] = field(default_factory=dict)
    postings: dict[int, list[tuple[str, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bands, self.rows = choose_bands(self.n_perm, self.threshold)

    # -- writing ----------------------------------------------------------

    def _band_keys(self, signature: list[int]) -> list[str]:
        keys = []
        for b in range(self.bands):
            chunk = signature[b * self.rows:(b + 1) * self.rows]
            keys.append(str(b) + ":" + ",".join(str(x) for x in chunk))
        return keys

    def add(self, fp: DocumentFingerprint, meta: SourceMeta | None = None) -> None:
        if fp.k != self.k or fp.w != self.w:
            raise ValueError(
                "fingerprint built with k=%d w=%d cannot enter an index built "
                "with k=%d w=%d" % (fp.k, fp.w, self.k, self.w)
            )
        doc_id = fp.doc_id
        self.meta[doc_id] = meta or SourceMeta(doc_id=doc_id, n_tokens=fp.n_tokens)
        self.meta[doc_id].n_tokens = fp.n_tokens
        self.signatures[doc_id] = fp.signature
        for key in self._band_keys(fp.signature):
            self.buckets.setdefault(key, set()).add(doc_id)
        for f in fp.fingerprints:
            self.postings.setdefault(f.value, []).append((doc_id, f.pos))

    def remove(self, doc_id: str) -> None:
        """Drop a document. Needed for the student-paper takedown path: a
        repository that cannot forget is a liability, not a feature."""
        sig = self.signatures.pop(doc_id, None)
        self.meta.pop(doc_id, None)
        if sig is not None:
            for key in self._band_keys(sig):
                bucket = self.buckets.get(key)
                if bucket:
                    bucket.discard(doc_id)
                    if not bucket:
                        del self.buckets[key]
        for value in list(self.postings):
            kept = [p for p in self.postings[value] if p[0] != doc_id]
            if kept:
                self.postings[value] = kept
            else:
                del self.postings[value]

    # -- reading ----------------------------------------------------------

    def candidates(
        self,
        fp: DocumentFingerprint,
        limit: int = 50,
        exclude: set[str] | None = None,
    ) -> list[Candidate]:
        exclude = exclude or set()
        hit_ids: set[str] = set()
        for key in self._band_keys(fp.signature):
            hit_ids |= self.buckets.get(key, set())
        hit_ids -= exclude
        hit_ids.discard(fp.doc_id)

        shared = self.shared_counts(fp)
        out = [
            Candidate(
                doc_id=doc_id,
                estimated_jaccard=estimate_jaccard(fp.signature, self.signatures[doc_id]),
                shared_fingerprints=shared.get(doc_id, 0),
                meta=self.meta[doc_id],
            )
            for doc_id in hit_ids
        ]
        out.sort(key=lambda c: (-c.estimated_jaccard, -c.shared_fingerprints))
        return out[:limit]

    def shared_counts(self, fp: DocumentFingerprint) -> dict[str, int]:
        """How many selected fingerprints each corpus document has in common."""
        counts: dict[str, int] = {}
        for value in fp.values:
            for doc_id, _pos in self.postings.get(value, ()):  # noqa: B007
                counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

    def seeds(self, fp: DocumentFingerprint, doc_id: str) -> list[tuple[int, int]]:
        """Matching (query_pos, source_pos) pairs to seed local alignment."""
        pairs: list[tuple[int, int]] = []
        for f in fp.fingerprints:
            for other_id, pos in self.postings.get(f.value, ()):
                if other_id == doc_id:
                    pairs.append((f.pos, pos))
        pairs.sort()
        return pairs

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(
                {
                    "k": self.k,
                    "w": self.w,
                    "n_perm": self.n_perm,
                    "threshold": self.threshold,
                    "bands": self.bands,
                    "rows": self.rows,
                },
                indent=2,
            )
        )
        (path / "meta.json").write_text(
            json.dumps({d: asdict(m) for d, m in self.meta.items()}, indent=2)
        )
        (path / "signatures.json").write_text(json.dumps(self.signatures))
        (path / "postings.json").write_text(
            json.dumps({str(v): p for v, p in self.postings.items()})
        )

    @classmethod
    def load(cls, path: str | Path) -> "CorpusIndex":
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        idx = cls(
            k=cfg["k"], w=cfg["w"], n_perm=cfg["n_perm"], threshold=cfg["threshold"]
        )
        idx.meta = {
            d: SourceMeta(**m)
            for d, m in json.loads((path / "meta.json").read_text()).items()
        }
        idx.signatures = json.loads((path / "signatures.json").read_text())
        idx.postings = {
            int(v): [(d, int(p)) for d, p in pairs]
            for v, pairs in json.loads((path / "postings.json").read_text()).items()
        }
        for doc_id, sig in idx.signatures.items():
            for key in idx._band_keys(sig):
                idx.buckets.setdefault(key, set()).add(doc_id)
        return idx

    def __len__(self) -> int:
        return len(self.signatures)

    def stats(self) -> dict[str, object]:
        return {
            "documents": len(self.signatures),
            "distinct_fingerprints": len(self.postings),
            "buckets": len(self.buckets),
            "bands": self.bands,
            "rows": self.rows,
            "lsh_knee": round((1.0 / self.bands) ** (1.0 / self.rows), 4),
        }
