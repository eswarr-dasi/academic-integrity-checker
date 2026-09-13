"""HTTP surface for the originality engine.

Run with:  uvicorn api.main:app --reload

Scope note. This service has no authentication, no tenancy and an in-process
report store, because those three things are deployment decisions and a
reference implementation that guesses at them is worse than one that leaves
the hole visible. Before this touches real coursework it needs, at minimum:

* an authenticated caller identity, with instructors scoped to their own
  sections and students able to read only their own reports;
* durable storage for reports and an explicit retention window;
* a legal basis for retaining student work in the reference corpus, and a
  working deletion path (the engine exposes one, see DELETE below);
* rate limiting, because fingerprinting is CPU bound and trivially abusable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from aic.ingest import MissingDependency, UnsupportedDocument, load_bytes
from aic.pipeline import Engine
from aic.scoring import ScoringSettings

app = FastAPI(
    title="academic-integrity-checker",
    version="0.1.0",
    description="Plagiarism and AI-writing detection with auditable reports.",
)


@dataclass
class State:
    engine: Engine = field(default_factory=Engine.blank)
    reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    html: dict[str, str] = field(default_factory=dict)


state = State()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"ok": True, "index": state.engine.index.stats()}


@app.post("/v1/corpus/documents", status_code=201)
async def add_corpus_document(
    file: UploadFile = File(...),
    title: str = Form(""),
    url: str = Form(""),
    kind: str = Form("corpus"),
) -> dict[str, Any]:
    """Add one reference document to the index."""
    raw = await _read(file)
    doc_id = state.engine.add_source(raw, title=title or raw.source_name, url=url, kind=kind)
    return {"doc_id": doc_id, "documents": len(state.engine.index)}


@app.delete("/v1/corpus/documents/{doc_id}")
def forget_corpus_document(doc_id: str) -> dict[str, Any]:
    """Remove a document from the reference corpus.

    Kept as a first class endpoint rather than an admin script. A reference
    corpus of student work that cannot be reduced is a compliance incident
    waiting to happen.
    """
    if doc_id not in state.engine.index.meta:
        raise HTTPException(status_code=404, detail="unknown doc_id")
    state.engine.forget(doc_id)
    return {"doc_id": doc_id, "removed": True, "documents": len(state.engine.index)}


@app.post("/v1/submissions", status_code=201)
async def create_submission(
    file: UploadFile = File(...),
    exclude_quotes: bool = Form(True),
    exclude_citations: bool = Form(True),
    exclude_bibliography: bool = Form(True),
    min_match_words: int = Form(8),
    add_to_corpus: bool = Form(False),
) -> JSONResponse:
    """Check a submission and store its report.

    Returns the full report rather than only an id, because the common client
    is a marking tool that wants the indices immediately and the spans later.
    """
    raw = await _read(file)
    settings = ScoringSettings(
        exclude_quotes=exclude_quotes,
        exclude_citations=exclude_citations,
        exclude_bibliography=exclude_bibliography,
        min_match_words=min_match_words,
    )
    report = state.engine.check(raw, settings=settings, add_to_corpus=add_to_corpus)
    payload = report.to_dict()
    state.reports[raw.doc_id] = payload
    state.html[raw.doc_id] = report.to_html()
    return JSONResponse(status_code=201, content=payload)


@app.get("/v1/reports/{doc_id}")
def get_report(doc_id: str) -> dict[str, Any]:
    report = state.reports.get(doc_id)
    if report is None:
        raise HTTPException(status_code=404, detail="unknown report id")
    return report


@app.get("/v1/reports/{doc_id}/html", response_class=HTMLResponse)
def get_report_html(doc_id: str) -> HTMLResponse:
    body = state.html.get(doc_id)
    if body is None:
        raise HTTPException(status_code=404, detail="unknown report id")
    return HTMLResponse(content=body)


@app.get("/v1/index/stats")
def index_stats() -> dict[str, Any]:
    return dict(state.engine.index.stats())


async def _read(file: UploadFile):
    data = await file.read()
    try:
        return load_bytes(data, file.filename or "upload.txt")
    except UnsupportedDocument as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except MissingDependency as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
