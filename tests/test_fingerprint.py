"""Tests for the fingerprinting layer.

The winnowing guarantee is the load bearing claim of the whole similarity
engine, so it gets tested directly rather than through the pipeline.
"""

from __future__ import annotations

from aic.fingerprint import (
    exact_jaccard,
    estimate_jaccard,
    fingerprint_tokens,
    guaranteed_match_length,
    minhash,
    shingle_hashes,
    winnow,
)

SAMPLE = (
    "Groundwater recharge in semi arid basins depends less on total annual "
    "rainfall than on the intensity distribution of individual storms. A "
    "single convective event that delivers forty millimetres in an hour "
    "produces far more runoff than the same volume spread across a week, and "
    "runoff is what reaches the wadi channels where transmission losses "
    "dominate the recharge budget. Field campaigns in the eastern basin "
    "measured channel infiltration rates between eleven and twenty three "
    "millimetres per hour, with the spread explained mostly by the depth of "
    "the sandy alluvium rather than by antecedent moisture. That result "
    "matters for management because it implies recharge can be increased by "
    "widening the wetted channel area rather than by attempting to capture "
    "hillslope runoff behind small dams, which simply relocates the losses "
    "to an evaporating reservoir surface."
)

TOKENS = SAMPLE.lower().split()


def test_shingle_count_and_rolling_hash_agree_with_direct_hash():
    k = 5
    hashes = shingle_hashes(TOKENS, k)
    assert len(hashes) == len(TOKENS) - k + 1
    # The rolling update must produce the same value as hashing the k-gram
    # from scratch, otherwise fingerprints are not portable between runs.
    for i in (0, 7, 23, len(hashes) - 1):
        assert hashes[i] == shingle_hashes(TOKENS[i:i + k], k)[0]


def test_shingling_degenerate_inputs():
    assert shingle_hashes([], 5) == []
    assert shingle_hashes(["one", "two"], 5) == []
    assert shingle_hashes(TOKENS, 0) == []
    assert winnow([], 4) == []


def test_winnow_selects_sparse_subset_and_keeps_positions():
    hashes = shingle_hashes(TOKENS, 5)
    fps = winnow(hashes, 4)
    assert 0 < len(fps) < len(hashes)
    assert all(0 <= f.pos < len(hashes) for f in fps)
    assert all(hashes[f.pos] == f.value for f in fps)
    # positions are strictly increasing, one selection per new window minimum
    assert [f.pos for f in fps] == sorted({f.pos for f in fps})


def test_winnowing_detects_every_shared_run_of_t_tokens():
    k, w = 5, 4
    t = guaranteed_match_length(k, w)
    assert t == 8
    shared = "transmission losses dominate the recharge budget in wadi channels".split()
    assert len(shared) >= t

    a = "an unrelated opening clause".split() + shared + "and a closing remark".split()
    b = "a completely different preamble sentence".split() + shared + "then more text".split()

    fa = fingerprint_tokens("a", a, k, w)
    fb = fingerprint_tokens("b", b, k, w)
    assert fa.values & fb.values, "a shared run of t tokens must share a fingerprint"


def test_short_shared_run_below_k_is_not_reported():
    k, w = 5, 4
    shared = "recharge budget".split()
    a = "one two three four".split() + shared + "five six seven".split()
    b = "eight nine ten eleven".split() + shared + "twelve thirteen".split()
    fa = fingerprint_tokens("a", a, k, w)
    fb = fingerprint_tokens("b", b, k, w)
    assert not (fa.values & fb.values)


def test_fingerprints_are_stable_under_a_single_insertion():
    """The whole point of winnowing over fixed-stride selection.

    Inserting one word near the start must not invalidate the fingerprints of
    everything after it, or every match past the first edit is lost.
    """
    base = fingerprint_tokens("base", TOKENS)
    edited = fingerprint_tokens("edited", ["however"] + TOKENS)
    retained = len(base.values & edited.values) / len(base.values)
    assert retained > 0.85


def test_fingerprints_are_deterministic_across_calls():
    first = fingerprint_tokens("d", TOKENS)
    second = fingerprint_tokens("d", TOKENS)
    assert first.signature == second.signature
    assert first.values == second.values


def test_minhash_estimates_jaccard():
    a = set(range(0, 400))
    b = set(range(200, 600))
    exact = exact_jaccard(a, b)
    estimate = estimate_jaccard(minhash(a), minhash(b))
    assert abs(estimate - exact) < 0.12


def test_minhash_identical_sets_agree_exactly():
    a = {11, 22, 33, 44}
    assert estimate_jaccard(minhash(a), minhash(set(a))) == 1.0


def test_minhash_disjoint_sets_rarely_agree():
    a = set(range(0, 500))
    b = set(range(10_000, 10_500))
    assert estimate_jaccard(minhash(a), minhash(b)) < 0.05


def test_positions_map_hash_to_every_occurrence():
    tokens = TOKENS + TOKENS
    fp = fingerprint_tokens("doubled", tokens)
    positions = fp.positions()
    assert all(isinstance(v, list) and v for v in positions.values())
    repeated = [v for v in positions.values() if len(v) > 1]
    assert repeated, "a document containing itself twice must repeat fingerprints"
