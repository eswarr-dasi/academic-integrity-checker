"""Fingerprinting: k-shingles, a Rabin-Karp rolling hash, winnowing, MinHash.

Hashing every k-gram of every corpus document is wasteful and, worse, it is
position sensitive: insert one word near the start of a paragraph and every
downstream fingerprint shifts. Winnowing fixes both problems by selecting a
sparse, content defined subset of the k-gram hashes.

The guarantee that matters, from Schleimer, Wilkerson and Aiken (2003):

    with shingle size k and window size w, winnowing detects every shared
    substring of length at least t = k + w - 1 tokens, and never reports a
    match shorter than k tokens.

So the two knobs have a direct policy meaning. k is the shortest phrase the
system is willing to call a match at all, and t is the shortest phrase it is
guaranteed to find. Defaults of k = 5 and w = 4 give t = 8 words, which is
roughly where a shared phrase stops being a coincidence of English and starts
being evidence of a shared source.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Mersenne prime 2**61 - 1. Big enough that collisions are not a practical
# concern, small enough to stay in one machine word on 64 bit CPython.
MERSENNE = (1 << 61) - 1
BASE = 257

DEFAULT_K = 5
DEFAULT_W = 4
DEFAULT_PERMUTATIONS = 128


@dataclass(frozen=True)
class Fingerprint:
    """A selected shingle hash and where it starts in the token stream."""

    value: int
    pos: int

    @property
    def end(self) -> int:
        return self.pos


@dataclass
class DocumentFingerprint:
    """Everything the index needs to know about one document."""

    doc_id: str
    k: int
    w: int
    n_tokens: int
    fingerprints: list[Fingerprint]
    signature: list[int]

    @property
    def values(self) -> set[int]:
        return {f.value for f in self.fingerprints}

    def positions(self) -> dict[int, list[int]]:
        """hash value -> every token position where it was selected."""
        out: dict[int, list[int]] = {}
        for f in self.fingerprints:
            out.setdefault(f.value, []).append(f.pos)
        return out


# --------------------------------------------------------------------------
# shingling
# --------------------------------------------------------------------------

def _token_value(token: str) -> int:
    """Stable per token integer. Python hash() is salted per process, so a
    fingerprint built today would not match one built tomorrow."""
    h = 0
    for ch in token:
        h = (h * 31 + ord(ch)) % MERSENNE
    return h


def shingle_hashes(tokens: list[str], k: int = DEFAULT_K) -> list[int]:
    """Rolling hash of every contiguous k-gram, in O(n) total.

    Returns a list of length max(0, len(tokens) - k + 1) where element i is the
    hash of tokens[i:i + k].
    """
    n = len(tokens)
    if n < k or k <= 0:
        return []

    values = [_token_value(t) for t in tokens]
    high = pow(BASE, k - 1, MERSENNE)

    h = 0
    for i in range(k):
        h = (h * BASE + values[i]) % MERSENNE
    out = [h]

    for i in range(1, n - k + 1):
        h = (h - values[i - 1] * high) % MERSENNE
        h = (h * BASE + values[i + k - 1]) % MERSENNE
        out.append(h)
    return out


# --------------------------------------------------------------------------
# winnowing
# --------------------------------------------------------------------------

def winnow(hashes: list[int], w: int = DEFAULT_W) -> list[Fingerprint]:
    """Select the minimum hash of every window of w consecutive hashes.

    Ties are broken by taking the rightmost minimum, which is what makes the
    selection stable under insertions: the same window content always yields
    the same choice regardless of what came before it.
    """
    if w <= 0 or not hashes:
        return []
    if len(hashes) <= w:
        best = min(range(len(hashes)), key=lambda i: (hashes[i], -i))
        return [Fingerprint(hashes[best], best)]

    selected: list[Fingerprint] = []
    # Monotonic deque of candidate indices, increasing hash value.
    deque: list[int] = []
    last_selected = -1

    for i, h in enumerate(hashes):
        while deque and hashes[deque[-1]] >= h:
            deque.pop()
        deque.append(i)
        while deque[0] <= i - w:
            deque.pop(0)
        if i >= w - 1:
            chosen = deque[0]
            if chosen != last_selected:
                selected.append(Fingerprint(hashes[chosen], chosen))
                last_selected = chosen
    return selected


# --------------------------------------------------------------------------
# MinHash
# --------------------------------------------------------------------------

def permutations(num: int = DEFAULT_PERMUTATIONS, seed: int = 20240101) -> list[tuple[int, int]]:
    """Coefficients for num universal hash functions h(x) = (a*x + b) mod p.

    The seed is fixed and part of the on-disk index format. Change it and
    every signature in the corpus has to be rebuilt.
    """
    rng = random.Random(seed)
    return [(rng.randrange(1, MERSENNE), rng.randrange(0, MERSENNE)) for _ in range(num)]


PERMS = permutations()


def minhash(values: set[int], perms: list[tuple[int, int]] | None = None) -> list[int]:
    """MinHash signature of a set of fingerprint values.

    Estimating Jaccard similarity from two signatures costs O(num_perm)
    instead of O(|A| + |B|), which is what makes the banded LSH lookup in
    aic.index affordable over millions of documents.
    """
    perms = perms or PERMS
    if not values:
        return [MERSENNE] * len(perms)
    return [min(((a * v + b) % MERSENNE) for v in values) for a, b in perms]


def estimate_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    """Fraction of agreeing signature slots, an unbiased Jaccard estimate."""
    if not sig_a or len(sig_a) != len(sig_b):
        return 0.0
    agree = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return agree / len(sig_a)


def exact_jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def fingerprint_tokens(
    doc_id: str,
    tokens: list[str],
    k: int = DEFAULT_K,
    w: int = DEFAULT_W,
    perms: list[tuple[int, int]] | None = None,
) -> DocumentFingerprint:
    hashes = shingle_hashes(tokens, k)
    fps = winnow(hashes, w)
    return DocumentFingerprint(
        doc_id=doc_id,
        k=k,
        w=w,
        n_tokens=len(tokens),
        fingerprints=fps,
        signature=minhash({f.value for f in fps}, perms),
    )


def guaranteed_match_length(k: int = DEFAULT_K, w: int = DEFAULT_W) -> int:
    """t = k + w - 1, the shortest shared phrase this configuration will find."""
    return k + w - 1
