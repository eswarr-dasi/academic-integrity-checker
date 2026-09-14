"""Tests for source discovery.

No test here touches the network. The payload fixtures are trimmed copies of
real OpenAlex and Crossref responses, so the parsing is exercised against the
shapes those APIs actually return rather than against a shape that would be
convenient.
"""

from __future__ import annotations

import pytest

from aic import discover
from aic.normalize import normalize

OPENALEX_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/W2741809807",
            "doi": "https://doi.org/10.1234/blue.carbon",
            "title": "Blue carbon burial in temperate saltmarsh",
            "publication_year": 2019,
            "abstract_inverted_index": {
                "Saltmarsh": [0],
                "sediments": [1],
                "bury": [2],
                "carbon": [3, 6],
                "at": [4],
                "measurable": [5],
                "rates": [7],
            },
            "open_access": {"is_oa": True},
        },
        {
            "id": "https://openalex.org/W9999999999",
            "doi": None,
            "title": "A record with no abstract at all",
            "publication_year": 2020,
            "abstract_inverted_index": None,
            "open_access": {"is_oa": False},
        },
    ]
}

CROSSREF_PAYLOAD = {
    "message": {
        "items": [
            {
                "DOI": "10.5194/oos2025-560",
                "URL": "https://doi.org/10.5194/oos2025-560",
                "title": ["Enhancing blue carbon sequestration"],
                "abstract": (
                    "<jats:p>Mangroves &amp; saltmarsh store"
                    " <jats:italic>organic</jats:italic> carbon.</jats:p>"
                ),
                "issued": {"date-parts": [[2025, 3]]},
            },
            {
                "DOI": "10.7185/gold2021.4039",
                "title": ["Porewater exchange"],
                "issued": {"date-parts": [[2021]]},
            },
        ]
    }
}

PROSE = (
    "Saltmarsh sediments accumulate organic carbon at rates that exceed "
    "terrestrial forest soils, and the resulting stock persists for "
    "centuries when the marsh surface keeps pace with relative sea level "
    "rise. Porewater exchange moves alkalinity offshore, which complicates "
    "any simple accounting of the sequestration benefit claimed for "
    "restoration projects in temperate estuaries."
)


def test_an_inverted_abstract_is_rebuilt_in_position_order():
    inverted = {"carbon": [3, 6], "the": [0, 2], "is": [1], "buried": [4, 5]}
    assert discover.reconstruct_abstract(inverted) == (
        "the is the carbon buried buried carbon"
    )


def test_a_missing_inverted_abstract_is_empty_rather_than_an_error():
    assert discover.reconstruct_abstract(None) == ""
    assert discover.reconstruct_abstract({}) == ""


def test_jats_markup_is_flattened_and_unescaped():
    raw = "<jats:p>Alpha &amp; <jats:italic>beta</jats:italic>\n  gamma</jats:p>"
    assert discover.strip_jats(raw) == "Alpha & beta gamma"
    assert discover.strip_jats(None) == ""


def test_query_phrases_are_content_bearing_and_do_not_overlap():
    doc = normalize(PROSE)
    phrases = discover.query_phrases(doc, count=4, words=6)
    assert len(phrases) == 4
    assert all(len(p.split()) == 6 for p in phrases)
    assert len(set(phrases)) == len(phrases)
    for phrase in phrases:
        content = [
            w for w in phrase.split() if w not in discover.STOPWORDS and len(w) > 3
        ]
        assert len(content) >= 3


def test_a_document_shorter_than_the_window_still_yields_one_phrase():
    phrases = discover.query_phrases(normalize("short marsh note"), count=4, words=6)
    assert phrases == ["short marsh note"]


def test_an_empty_document_yields_no_phrases():
    assert discover.query_phrases(normalize(""), count=4) == []


def test_openalex_records_without_an_abstract_are_skipped(monkeypatch):
    monkeypatch.setattr(discover, "_get", lambda url, timeout=20.0: OPENALEX_PAYLOAD)
    found = discover.search_openalex("blue carbon")
    assert len(found) == 1
    candidate = found[0]
    assert candidate.source_id == "https://doi.org/10.1234/blue.carbon"
    assert candidate.provider == "openalex"
    assert candidate.year == 2019
    assert candidate.open_access is True
    assert candidate.text.startswith("Blue carbon burial in temperate saltmarsh. ")
    assert "bury carbon" in candidate.text


def test_crossref_records_without_an_abstract_are_skipped(monkeypatch):
    monkeypatch.setattr(discover, "_get", lambda url, timeout=20.0: CROSSREF_PAYLOAD)
    found = discover.search_crossref("blue carbon")
    assert len(found) == 1
    candidate = found[0]
    assert candidate.source_id == "10.5194/oos2025-560"
    assert candidate.year == 2025
    assert candidate.open_access is False
    assert "Mangroves & saltmarsh store organic carbon." in candidate.text


def test_a_candidate_converts_to_a_raw_document():
    candidate = discover.Candidate(
        source_id="10.1/x",
        title="Title",
        url="https://doi.org/10.1/x",
        text="Title. Body text.",
        provider="crossref",
    )
    raw = candidate.as_raw()
    assert raw.doc_id == "10.1/x"
    assert raw.source_name == "Title"
    assert raw.text == "Title. Body text."


def test_discovery_deduplicates_the_same_doi_across_providers(monkeypatch):
    shared = discover.Candidate(
        source_id="https://doi.org/10.1/dup",
        title="Shared",
        url="",
        text="Shared. Body.",
        provider="openalex",
    )
    twin = discover.Candidate(
        source_id="10.1/DUP",
        title="Shared",
        url="",
        text="Shared. Body.",
        provider="crossref",
    )
    monkeypatch.setattr(discover, "search_openalex", lambda *a, **k: [shared])
    monkeypatch.setattr(discover, "search_crossref", lambda *a, **k: [twin])
    found = discover.discover(normalize(PROSE), queries=3)
    assert len(found) == 1


def test_a_failing_provider_narrows_the_index_instead_of_aborting(monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("connection reset")

    survivor = discover.Candidate(
        source_id="10.1/ok",
        title="Survivor",
        url="",
        text="Survivor. Body.",
        provider="crossref",
    )
    monkeypatch.setattr(discover, "search_openalex", explode)
    monkeypatch.setattr(discover, "search_crossref", lambda *a, **k: [survivor])
    found = discover.discover(normalize(PROSE), queries=2)
    assert [c.source_id for c in found] == ["10.1/ok"]


def test_discovery_asks_for_at_most_the_requested_number_of_queries(monkeypatch):
    seen: list[str] = []

    def record(phrase, *args, **kwargs):
        seen.append(phrase)
        return []

    monkeypatch.setattr(discover, "search_openalex", record)
    monkeypatch.setattr(discover, "search_crossref", lambda *a, **k: [])
    discover.discover(normalize(PROSE), queries=3)
    assert len(seen) == 3


@pytest.mark.parametrize(
    "doi,expected",
    [
        ("https://doi.org/10.1/A", "10.1/a"),
        ("10.1/A", "10.1/a"),
    ],
)
def test_doi_keys_normalise_to_the_same_string(doi, expected):
    candidate = discover.Candidate(
        source_id=doi, title="t", url="", text="t", provider="openalex"
    )
    assert discover._key(candidate) == expected
