"""Document ingest: bytes in, one text string plus a page map out.

The page map is the part that is easy to skip and expensive to retrofit. A
report that says "281 matched words" is useless if the reader cannot find them
in the original PDF, so every loader records the character offset at which
each page starts. A character offset in the extracted text can then be turned
back into a page number in constant time.

Heavy parsers are imported lazily. The core engine must stay importable on a
machine with nothing but the standard library, which is what makes the unit
tests fast and the container small.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".txt", ".md", ".text", ".pdf", ".docx", ".rtf", ".html", ".htm"}
MAX_BYTES = 40 * 1024 * 1024


class UnsupportedDocument(ValueError):
    pass


class MissingDependency(RuntimeError):
    pass


@dataclass
class RawDocument:
    doc_id: str
    text: str
    source_name: str
    page_starts: list[int] = field(default_factory=list)
    media_type: str = "text/plain"
    sha256: str = ""

    @property
    def n_pages(self) -> int:
        return max(1, len(self.page_starts))

    def page_of(self, offset: int) -> int:
        """1-based page containing a character offset of self.text."""
        page = 1
        for i, start in enumerate(self.page_starts):
            if offset >= start:
                page = i + 1
            else:
                break
        return page

    def locate(self, start: int, end: int) -> dict[str, int]:
        return {
            "page_start": self.page_of(start),
            "page_end": self.page_of(max(start, end - 1)),
            "char_start": start,
            "char_end": end,
        }


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc_id_for(data: bytes, name: str) -> str:
    """Content addressed id so that resubmitting the same file is detectable.

    Using the hash rather than a random uuid also means a corpus can be
    deduplicated without comparing text, which matters once the reference set
    is large enough that near-duplicates of the same paper are common.
    """
    return "sha256:" + digest(data)[:24]


# --------------------------------------------------------------------------
# per format loaders
# --------------------------------------------------------------------------

def _load_text(data: bytes) -> tuple[str, list[int]]:
    for encoding in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding), [0]
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), [0]


def _load_pdf(data: bytes) -> tuple[str, list[int]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise MissingDependency(
            "PDF ingest needs pypdf. Install with: pip install pypdf"
        ) from exc
    import io

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    starts: list[int] = []
    cursor = 0
    for page in reader.pages:
        starts.append(cursor)
        body = page.extract_text() or ""
        parts.append(body)
        cursor += len(body) + 1
    return "\n".join(parts), starts


def _load_docx(data: bytes) -> tuple[str, list[int]]:
    try:
        import docx  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependency(
            "DOCX ingest needs python-docx. Install with: pip install python-docx"
        ) from exc
    import io

    document = docx.Document(io.BytesIO(data))
    blocks = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            blocks.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(blocks), [0]


def _load_rtf(data: bytes) -> tuple[str, list[int]]:
    try:
        from striprtf.striprtf import rtf_to_text  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependency(
            "RTF ingest needs striprtf. Install with: pip install striprtf"
        ) from exc
    text, _ = _load_text(data)
    return rtf_to_text(text), [0]


_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)


def _load_html(data: bytes) -> tuple[str, list[int]]:
    text, _ = _load_text(data)
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text("\n"), [0]
    except ImportError:
        # Regex fallback. Good enough for a smoke test, not for production.
        stripped = _SCRIPT.sub(" ", text)
        return _TAG.sub(" ", stripped), [0]


_LOADERS = {
    ".txt": _load_text,
    ".text": _load_text,
    ".md": _load_text,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".rtf": _load_rtf,
    ".html": _load_html,
    ".htm": _load_html,
}

_MEDIA = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".rtf": "application/rtf",
    ".html": "text/html",
    ".htm": "text/html",
}


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def load_bytes(data: bytes, name: str) -> RawDocument:
    if len(data) > MAX_BYTES:
        raise UnsupportedDocument(
            "file is %d bytes, limit is %d" % (len(data), MAX_BYTES)
        )
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedDocument(
            "%s is not a supported extension, expected one of %s"
            % (suffix or "(none)", ", ".join(sorted(SUPPORTED)))
        )
    text, starts = _LOADERS[suffix](data)
    return RawDocument(
        doc_id=doc_id_for(data, name),
        text=_tidy(text),
        source_name=Path(name).name,
        page_starts=starts or [0],
        media_type=_MEDIA.get(suffix, "text/plain"),
        sha256=digest(data),
    )


def load_path(path: str | Path) -> RawDocument:
    path = Path(path)
    return load_bytes(path.read_bytes(), path.name)


def load_dir(root: str | Path, recursive: bool = True) -> list[RawDocument]:
    """Load every supported file under root, skipping the ones that fail.

    Corpus building must not abort on one corrupt PDF out of fifty thousand,
    so failures are dropped here and counted by the caller.
    """
    root = Path(root)
    pattern = "**/*" if recursive else "*"
    out: list[RawDocument] = []
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            try:
                out.append(load_path(path))
            except (UnsupportedDocument, MissingDependency, OSError):
                continue
    return out


_WS_RUN = re.compile(r"[ \t\x0b\f\r]+")
_BLANK_RUN = re.compile(r"\n{3,}")


def _tidy(text: str) -> str:
    """Collapse runs of spaces and blank lines without touching offsets that
    matter. Called once, before normalization, so the offset table built in
    aic.normalize refers to this tidied text and stays consistent."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RUN.sub(" ", text)
    return _BLANK_RUN.sub("\n\n", text).strip()
