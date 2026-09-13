"""The originality report: versioned JSON first, HTML rendered from it.

JSON is the source of truth. The HTML view is a pure function of the JSON, so
anything an instructor can see in the browser is also in the machine readable
record, and an appeal can be re-examined years later from the stored document
alone.

REPORT_VERSION changes whenever a field is removed or its meaning changes.
Additive changes do not bump it. Consumers should reject a report whose major
version they do not recognise rather than guess.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .ai_detect import AIAnalysis
from .ingest import RawDocument
from .scoring import ScoringSettings, SimilarityResult, review_priority

REPORT_VERSION = "1.0"

_BAND_COLOR = {
    "human": "#2f6f3e",
    "unclear": "#8a6d1f",
    "likely-ai": "#a8541b",
    "very-likely-ai": "#8f2020",
}


@dataclass
class OriginalityReport:
    document: RawDocument
    similarity: SimilarityResult
    ai: AIAnalysis
    settings: ScoringSettings
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    engine_version: str = REPORT_VERSION

    # -- machine readable -------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        doc = self.document
        priority = review_priority(
            self.similarity.index, self.ai.index, self.ai.band
        )
        return {
            "report_version": REPORT_VERSION,
            "generated_at": self.generated_at,
            "document": {
                "id": doc.doc_id,
                "name": doc.source_name,
                "sha256": doc.sha256,
                "media_type": doc.media_type,
                "pages": doc.n_pages,
                "total_words": self.similarity.total_words,
                "scorable_words": self.similarity.scorable_words,
                "excluded_words": self.similarity.excluded_words,
            },
            "similarity": self.similarity.to_dict(),
            "ai_writing": self.ai.to_dict(),
            "settings": self.settings.to_dict(),
            "review_priority": priority,
            "caveats": self._caveats(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def _caveats(self) -> list[str]:
        out = list(self.ai.caveats)
        out.append(
            "A similarity index is not a plagiarism finding. Quoted, cited and "
            "boilerplate text can all raise it legitimately."
        )
        out.append(
            "Matches can only be found against documents present in the "
            "reference index, so a low index does not mean original."
        )
        return out

    # -- human readable ---------------------------------------------------

    def to_html(self) -> str:
        d = self.to_dict()
        sim = self.similarity
        ai = self.ai
        body = _highlight(self.document.text, self)

        rows = "".join(
            "<tr><td>%s</td><td class=num>%d</td><td class=num>%.1f%%</td>"
            "<td class=num>%.2f</td><td>%s</td></tr>"
            % (
                html.escape(s.source_label),
                s.words,
                100 * s.share,
                s.best_similarity,
                html.escape(", ".join(s.match_types)),
            )
            for s in sim.by_source
        ) or "<tr><td colspan=5>No sources matched above the minimum length.</td></tr>"

        caveats = "".join(
            "<li>%s</li>" % html.escape(c) for c in self._caveats()
        )

        return _TEMPLATE % {
            "name": html.escape(self.document.source_name),
            "generated": html.escape(self.generated_at),
            "sim_pct": 100 * sim.index,
            "matched": sim.matched_words,
            "scorable": sim.scorable_words,
            "excluded": sim.excluded_words,
            "ai_pct": 100 * ai.index,
            "ai_band": html.escape(ai.band),
            "ai_color": _BAND_COLOR.get(ai.band, "#444"),
            "ci_lo": 100 * ai.confidence_interval[0],
            "ci_hi": 100 * ai.confidence_interval[1],
            "calibrated": "calibrated" if ai.calibrated else "NOT calibrated",
            "priority": html.escape(str(d["review_priority"])),
            "rows": rows,
            "body": body,
            "caveats": caveats,
        }


def _highlight(text: str, report: OriginalityReport) -> str:
    """Wrap matched character spans in mark elements, HTML escaping the rest.

    Spans are applied in a single left to right pass over disjoint,
    pre-merged ranges. Nesting is not attempted, because overlapping
    highlights in a document a human has to read are worse than useless.
    """
    spans: list[tuple[int, int, str, str]] = []
    for m in report.similarity.matches:
        start, end = m.query_chars
        kind = "excl" if m.excluded else "match"
        label = "%s (%s)" % (m.source_label, m.match_type)
        if m.excluded:
            label += " excluded: %s" % m.exclusion_reason
        spans.append((start, end, kind, label))

    spans.sort()
    out: list[str] = []
    cursor = 0
    for start, end, kind, label in spans:
        if start < cursor:
            continue
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        out.append(html.escape(text[cursor:start]))
        out.append(
            '<mark class="%s" title="%s">%s</mark>'
            % (kind, html.escape(label, quote=True), html.escape(text[start:end]))
        )
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "".join(out).replace("\n", "<br>\n")


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Originality report - %(name)s</title>
<style>
 body { font: 15px/1.6 -apple-system, Segoe UI, Roboto, sans-serif;
        margin: 0 auto; max-width: 980px; padding: 32px 20px; color: #1b1b1b; }
 h1 { font-size: 22px; margin: 0 0 4px; }
 .meta { color: #666; font-size: 13px; margin-bottom: 24px; }
 .cards { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px; }
 .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px 20px;
         min-width: 210px; }
 .big { font-size: 30px; font-weight: 600; line-height: 1.1; }
 .label { font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
          color: #666; }
 .sub { font-size: 12px; color: #666; margin-top: 6px; }
 table { border-collapse: collapse; width: 100%%; margin: 10px 0 26px; }
 th, td { border-bottom: 1px solid #eee; padding: 7px 8px; text-align: left;
          font-size: 13px; }
 td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
 mark.match { background: #ffe08a; padding: 0 1px; }
 mark.excl { background: #e6e6e6; color: #555; padding: 0 1px; }
 .doc { border: 1px solid #ddd; border-radius: 10px; padding: 20px;
        white-space: normal; }
 .warn { background: #fff8e1; border: 1px solid #f0d68a; border-radius: 10px;
         padding: 12px 18px; font-size: 13px; }
 .warn li { margin: 4px 0; }
 h2 { font-size: 16px; margin-top: 30px; }
</style>
</head>
<body>
<h1>Originality report</h1>
<div class="meta">%(name)s &middot; generated %(generated)s &middot;
 review priority: <strong>%(priority)s</strong></div>

<div class="cards">
  <div class="card">
    <div class="label">Similarity index</div>
    <div class="big">%(sim_pct).1f%%</div>
    <div class="sub">%(matched)d of %(scorable)d scorable words<br>
      %(excluded)d words excluded by policy</div>
  </div>
  <div class="card">
    <div class="label">AI-writing index</div>
    <div class="big" style="color:%(ai_color)s">%(ai_pct).1f%%</div>
    <div class="sub">band: %(ai_band)s<br>
      dispersion range %(ci_lo).0f%% to %(ci_hi).0f%% &middot; %(calibrated)s</div>
  </div>
</div>

<h2>Matched sources</h2>
<table>
<tr><th>Source</th><th class="num">Words</th><th class="num">Share</th>
    <th class="num">Similarity</th><th>Types</th></tr>
%(rows)s
</table>

<h2>How to read this</h2>
<div class="warn"><ul>%(caveats)s</ul></div>

<h2>Submission</h2>
<div class="doc">%(body)s</div>
</body>
</html>
"""
