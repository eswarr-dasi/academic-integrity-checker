"""End to end orchestration: corpus in, originality report out.

Everything below is glue. The interesting decisions live in the modules this
imports, and this file exists so that the CLI, the API and the tests all drive
the engine through exactly one code path. If a report can be produced two
different ways, the two ways will disagree eventually.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .ai_detect import AIDetector, LanguageModel
from .fingerprint import DEFAULT_K, DEFAULT_W, fingerprint_tokens
from .index import CorpusIndex, SourceMeta
from .ingest import RawDocument, load_dir, load_path
from .normalize import normalize
from .report import OriginalityReport
from .scoring import ScoringSettings, score_similarity
from .similarity import find_matches


@dataclass
class TokenStore:
    """Token streams for indexed documents, needed for alignment.

    Fingerprints alone cannot produce a readable match span, so the corpus
    token stream has to be retrievable. For a real deployment this is object
    storage keyed by doc_id, not a dict, and it is the component with the
    hardest retention questions attached to it: keeping every student paper
    forever is a choice, not a requirement.
    """

    data: dict[str, list[str]] = field(default_factory=dict)

    def put(self, doc_id: str, tokens: list[str]) -> None:
        self.data[doc_id] = tokens

    def tokens(self, doc_id: str) -> list[str]:
        return list(self.data.get(doc_id, []))

    def drop(self, doc_id: str) -> None:
        self.data.pop(doc_id, None)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.data))

    @classmethod
    def load(cls, path: str | Path) -> TokenStore:
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(json.loads(p.read_text()))


@dataclass
class Engine:
    index: CorpusIndex
    store: TokenStore = field(default_factory=TokenStore)
    detector: AIDetector = field(default_factory=AIDetector)
    settings: ScoringSettings = field(default_factory=ScoringSettings)

    # -- corpus side ------------------------------------------------------

    def add_source(
        self,
        raw: RawDocument,
        title: str = "",
        url: str = "",
        kind: str = "corpus",
        keep_tokens: bool = True,
    ) -> str:
        doc = normalize(raw.text)
        fp = fingerprint_tokens(raw.doc_id, doc.words, self.index.k, self.index.w)
        self.index.add(
            fp,
            SourceMeta(
                doc_id=raw.doc_id,
                title=title or raw.source_name,
                url=url,
                kind=kind,
                n_tokens=len(doc.words),
            ),
        )
        if keep_tokens:
            self.store.put(raw.doc_id, doc.words)
        return raw.doc_id

    def add_corpus_dir(self, root: str | Path) -> list[str]:
        return [self.add_source(raw) for raw in load_dir(root)]

    def forget(self, doc_id: str) -> None:
        """Remove a source from the index and the token store.

        Exposed deliberately. A checker that can ingest a student paper but
        never delete it cannot satisfy a deletion request, and that is a legal
        problem long before it is an engineering one.
        """
        self.index.remove(doc_id)
        self.store.drop(doc_id)

    # -- submission side --------------------------------------------------

    def check(
        self,
        raw: RawDocument,
        settings: ScoringSettings | None = None,
        add_to_corpus: bool = False,
    ) -> OriginalityReport:
        settings = settings or self.settings
        doc = normalize(raw.text)
        fp = fingerprint_tokens(raw.doc_id, doc.words, self.index.k, self.index.w)

        candidates = self.index.candidates(fp, exclude={raw.doc_id})
        matches = find_matches(
            doc,
            fp,
            self.index,
            self.store,
            candidates=candidates,
            min_words=settings.min_match_words,
        )
        similarity = score_similarity(doc, matches, settings)
        ai = self.detector.analyze(doc)

        if add_to_corpus:
            # Self-submission must never be a match against itself, which is
            # why indexing happens after the check and not before.
            self.add_source(raw, kind="student")

        return OriginalityReport(
            document=raw,
            similarity=similarity,
            ai=ai,
            settings=settings,
        )

    def check_path(self, path: str | Path, **kwargs: object) -> OriginalityReport:
        return self.check(load_path(path), **kwargs)  # type: ignore[arg-type]

    # -- persistence ------------------------------------------------------

    def save(self, root: str | Path) -> None:
        root = Path(root)
        self.index.save(root)
        self.store.save(root / "tokens.json")

    @classmethod
    def load(
        cls,
        root: str | Path,
        lm: LanguageModel | None = None,
        settings: ScoringSettings | None = None,
    ) -> Engine:
        root = Path(root)
        return cls(
            index=CorpusIndex.load(root),
            store=TokenStore.load(root / "tokens.json"),
            detector=AIDetector(lm=lm),
            settings=settings or ScoringSettings(),
        )

    @classmethod
    def blank(
        cls,
        k: int = DEFAULT_K,
        w: int = DEFAULT_W,
        lm: LanguageModel | None = None,
    ) -> Engine:
        return cls(index=CorpusIndex(k=k, w=w), detector=AIDetector(lm=lm))


def build_index(
    corpus_dir: str | Path,
    out_dir: str | Path,
    k: int = DEFAULT_K,
    w: int = DEFAULT_W,
) -> dict[str, object]:
    engine = Engine.blank(k=k, w=w)
    added = engine.add_corpus_dir(corpus_dir)
    engine.save(out_dir)
    stats = dict(engine.index.stats())
    stats["added"] = len(added)
    return stats


def quick_check(
    submission: str | Path,
    index_dir: str | Path,
    lm: LanguageModel | None = None,
) -> OriginalityReport:
    """One call convenience wrapper used by the CLI and the smoke tests."""
    return Engine.load(index_dir, lm=lm).check_path(submission)


def in_memory_check(
    submission_text: str,
    sources: dict[str, str],
    lm: LanguageModel | None = None,
) -> OriginalityReport:
    """Check a string against a dict of {label: text}. No files, no index dir.

    This is the path the unit tests use, and the one to reach for when
    reproducing a disputed report from stored text.
    """
    from .ingest import load_bytes

    engine = Engine.blank(lm=lm)
    for label, text in sources.items():
        raw = load_bytes(text.encode("utf-8"), label + ".txt")
        engine.add_source(raw, title=label)
    return engine.check(load_bytes(submission_text.encode("utf-8"), "submission.txt"))
