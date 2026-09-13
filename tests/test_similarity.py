"""End to end tests for normalization, matching and the similarity index.

All source text in this file is written for the test suite, so the corpus is
unencumbered and the expected offsets are known.
"""

from __future__ import annotations

import json

from aic.ingest import load_bytes
from aic.normalize import normalize
from aic.pipeline import Engine, in_memory_check
from aic.scoring import ScoringSettings, review_priority
from aic.similarity import smith_waterman

SOURCE_A = (
    "Coastal saltmarsh sediment stores carbon at rates an order of magnitude "
    "above those of temperate forest, and the mechanism is burial rather than "
    "growth. Vertical accretion keeps pace with sea level rise wherever the "
    "suspended sediment supply is adequate, which means the marsh surface "
    "climbs and the older organic layers pass below the oxygenated zone where "
    "decomposition slows almost to a halt. The practical consequence is that a "
    "marsh allowed to migrate landward retains its carbon function, while one "
    "pinned against a seawall drowns and releases it."
)

COPIED = (
    "the marsh surface climbs and the older organic layers pass below the "
    "oxygenated zone where decomposition slows almost to a halt"
)

ORIGINAL_B = (
    "Urban tree canopy reduces peak summer surface temperature mainly through "
    "transpiration and not through shade, which is why a young street planting "
    "with a small leaf area delivers little measurable cooling for its first "
    "decade. Species choice matters less than water availability. A drought "
    "stressed plane tree closes its stomata by midday and behaves thermally "
    "like a lamp post, so irrigation planning belongs in the canopy strategy "
    "rather than in the maintenance budget where it is usually filed."
)

SUBMISSION_COPIED = ORIGINAL_B + " " + COPIED + ". A closing sentence of my own."


def _engine_with(sources: dict[str, str]) -> Engine:
    engine = Engine.blank()
    for label, text in sources.items():
        engine.add_source(load_bytes(text.encode("utf-8"), label + ".txt"), title=label)
    return engine


def _check(text: str, sources: dict[str, str], settings=None):
    engine = _engine_with(sources)
    raw = load_bytes(text.encode("utf-8"), "submission.txt")
    return engine.check(raw, settings=settings)


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

def test_dehyphenation_rejoins_words_split_across_lines():
    doc = normalize("the maintenance bud-\nget is dominated by irrigation")
    assert "budget" in doc.words
    assert "bud" not in doc.words


def test_abbreviations_do_not_end_a_sentence():
    doc = normalize("Smith et al. 2019 reported gains. A second sentence follows.")
    assert len(doc.sentences) == 2


def test_token_offsets_point_back_into_the_raw_text():
    raw = "Managed realignment buys flood attenuation."
    doc = normalize(raw)
    for token in doc.tokens:
        assert raw[token.start:token.end].casefold() == token.text


def test_bibliography_is_tagged_and_removed_from_the_denominator():
    raw = ORIGINAL_B + "\n\nReferences\n\nSomebody. A paper title. A journal, 2019.\n"
    doc = normalize(raw)
    assert any(s.kind == "bibliography" for s in doc.exclusions)
    mask = doc.scorable_mask({"bibliography"})
    assert not all(mask)


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

def test_smith_waterman_recovers_an_exact_embedded_run():
    a = "keeps pace with sea level rise".split()
    b = "and so it keeps pace with sea level rise wherever supply".split()
    qa, qb, sa, sb, identity = smith_waterman(a, b)
    assert (qa, qb) == (0, len(a) - 1)
    assert b[sa:sb + 1] == a
    assert identity == 1.0


def test_smith_waterman_returns_nothing_for_disjoint_text():
    qa, qb, _, _, identity = smith_waterman("alpha beta".split(), "gamma delta".split())
    assert qb < qa
    assert identity == 0.0


# --------------------------------------------------------------------------
# matching and scoring
# --------------------------------------------------------------------------

def test_copied_passage_is_found_and_attributed():
    report = _check(SUBMISSION_COPIED, {"Source A": SOURCE_A})
    sim = report.similarity
    assert sim.index > 0.12
    assert sim.matched_words >= 15
    assert sim.by_source
    assert sim.by_source[0].source_label == "Source A"
    assert sim.by_source[0].match_types == ["verbatim"]


def test_an_original_submission_scores_zero():
    report = _check(ORIGINAL_B, {"Source A": SOURCE_A})
    assert report.similarity.index == 0.0
    assert report.similarity.matches == []


def test_index_never_exceeds_one_with_many_overlapping_sources():
    sources = {
        "Source A": SOURCE_A,
        "Source A reprint": SOURCE_A + " An extra trailing remark.",
        "Source A mirror": "A different opening remark. " + SOURCE_A,
    }
    report = _check(SUBMISSION_COPIED, sources)
    sim = report.similarity
    assert 0.0 <= sim.index <= 1.0
    assert sim.matched_words <= sim.scorable_words
    # every plausible origin is listed, and the shares still sum to the index
    assert len(sim.by_source) >= 2
    assert abs(sum(s.share for s in sim.by_source) - sim.index) < 1e-9


def test_quoted_and_excluded_passage_lowers_the_index():
    quoted = (
        ORIGINAL_B
        + ' One review notes that "'
        + COPIED
        + '." A closing sentence of my own.'
    )
    with_exclusion = _check(quoted, {"Source A": SOURCE_A})
    without_exclusion = _check(
        quoted, {"Source A": SOURCE_A}, settings=ScoringSettings(exclude_quotes=False)
    )
    assert with_exclusion.similarity.index < without_exclusion.similarity.index
    assert any(m.excluded for m in with_exclusion.similarity.matches)
    assert any(
        m.exclusion_reason == "quote" for m in with_exclusion.similarity.matches
    )


def test_short_overlap_below_the_minimum_is_not_reported():
    submission = ORIGINAL_B + " The marsh surface climbs. And then it stops."
    report = _check(submission, {"Source A": SOURCE_A})
    assert report.similarity.index == 0.0


def test_homoglyph_and_zero_width_evasion_is_still_detected():
    """Character level obfuscation must not defeat the fingerprints.

    Cyrillic lookalikes and zero width joiners are what the consumer grade
    evasion tools reach for first, and both are cheap to normalize away.
    """
    evaded = "\u200b".join(COPIED.replace("o", "\u043e").replace("a", "\u0430"))
    submission = ORIGINAL_B + " " + evaded + ". A closing sentence of my own."
    report = _check(submission, {"Source A": SOURCE_A})
    assert report.similarity.index > 0.12


def test_allowlisted_source_is_ignored():
    engine = _engine_with({"Assignment brief": SOURCE_A})
    brief_id = next(iter(engine.index.meta))
    raw = load_bytes(SUBMISSION_COPIED.encode("utf-8"), "submission.txt")
    report = engine.check(raw, settings=ScoringSettings(allowlist={brief_id}))
    assert report.similarity.index == 0.0


def test_self_submission_does_not_match_itself():
    engine = _engine_with({"Source A": SOURCE_A})
    raw = load_bytes(SUBMISSION_COPIED.encode("utf-8"), "submission.txt")
    first = engine.check(raw, add_to_corpus=True)
    second = engine.check(raw)
    # indexing happens after scoring, so the first report is unaffected, and
    # the resubmission finds the stored copy rather than silently ignoring it
    assert first.similarity.index > 0.12
    assert second.similarity.index >= first.similarity.index


def test_forget_removes_a_source_from_future_reports():
    engine = _engine_with({"Source A": SOURCE_A})
    doc_id = next(iter(engine.index.meta))
    raw = load_bytes(SUBMISSION_COPIED.encode("utf-8"), "submission.txt")
    assert engine.check(raw).similarity.index > 0.0
    engine.forget(doc_id)
    assert len(engine.index) == 0
    assert engine.check(raw).similarity.index == 0.0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def test_report_json_is_versioned_and_carries_caveats():
    report = in_memory_check(SUBMISSION_COPIED, {"Source A": SOURCE_A})
    payload = json.loads(report.to_json())
    assert payload["report_version"] == "1.0"
    assert payload["document"]["total_words"] > 0
    assert payload["similarity"]["index"] >= 0
    assert payload["ai_writing"]["band"] in (
        "human", "unclear", "likely-ai", "very-likely-ai"
    )
    assert payload["caveats"], "a report without caveats is a report that overclaims"
    assert payload["review_priority"] in ("read-first", "worth-a-look", "no-flags")


def test_report_html_highlights_the_matched_span():
    report = in_memory_check(SUBMISSION_COPIED, {"Source A": SOURCE_A})
    body = report.to_html()
    assert "Originality report" in body
    assert "<mark" in body
    assert "Source A" in body


def test_review_priority_is_a_triage_hint_not_a_verdict():
    assert review_priority(0.45, 0.1, "human") == "read-first"
    assert review_priority(0.12, 0.92, "likely-ai") == "read-first"
    assert review_priority(0.16, 0.1, "human") == "worth-a-look"
    assert review_priority(0.01, 0.05, "human") == "no-flags"


def test_index_round_trips_through_disk(tmp_path):
    engine = _engine_with({"Source A": SOURCE_A})
    engine.save(tmp_path)
    reloaded = Engine.load(tmp_path)
    assert len(reloaded.index) == len(engine.index)
    raw = load_bytes(SUBMISSION_COPIED.encode("utf-8"), "submission.txt")
    before = engine.check(raw).similarity.index
    after = reloaded.check(raw).similarity.index
    assert abs(before - after) < 1e-9
