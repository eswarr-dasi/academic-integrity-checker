# Scoring

Two numbers leave this system. Both are defined here, once, in prose that an
appeals committee can read. If the code and this document disagree, the code
is wrong.

---

## 1. Similarity index

### Definition

~~~
similarity_index = |union of matched scorable token positions|
                   ---------------------------------------------
                   |scorable token positions|
~~~

A token position is **scorable** unless it falls inside a span the normalizer
tagged as quotation, in-text citation, or reference list, and the
corresponding exclusion is enabled.

A token position is **matched** if it lies inside an accepted match. A match
is accepted when all of the following hold:

1. it is at least **min_match_words** long (default 8 words);
2. its source is not on the allowlist;
3. it is not itself wholly inside an excluded span (at least 80 percent
   overlap counts as wholly).

### Properties that hold by construction

| Property | Why |
| --- | --- |
| index is in [0, 1] | positions are unioned, never summed |
| a sentence found in N sources counts once | union, and per-source shares subtract already-claimed positions |
| excluded text leaves both numerator and denominator | the scorable set is computed before matching is counted |
| sub-threshold matches never inflate the index | filtering happens before scoring, not after |

The third property is the one people get wrong. If quoted text were removed
from the numerator only, a paper that is 40 percent correctly attributed block
quotation would score 0 percent and look cleaner than an original one, which
is absurd in the opposite direction. Both sides come out.

### Worked example

A 1000 word essay. The reference list is 120 words. One 60 word passage is
copied verbatim from a journal article. A 30 word passage is quoted correctly
with quotation marks and a citation.

~~~
total words                       1000
bibliography                      -120
quoted span                        -30
                                  ----
scorable                           850

matched, not excluded               60   (the verbatim passage)
matched, excluded                   30   (the quotation, reported with a reason)

similarity_index = 60 / 850 = 0.0706  ->  7.1%
~~~

The quotation still appears in the report, greyed out, labelled
"excluded: quote". Hiding it would be dishonest, counting it would be unfair.

### What the number does not mean

There is no threshold at which the index becomes misconduct. A methods section
in a lab report can legitimately exceed 40 percent. A fabricated essay that
was written to order by a human can sit at 0 percent. Any institution that
writes a number into its regulations is making a policy choice and should
publish the reasoning, because the tool cannot supply it.

### Knobs and their meaning

| Setting | Default | Effect |
| --- | --- | --- |
| min_match_words | 8 | shortest reported match. Below about 6 words, shared phrasing in English is common enough to be noise |
| k (shingle size) | 5 | shortest phrase the engine can match at all |
| w (winnow window) | 4 | with k, fixes the guaranteed detection length t = k + w - 1 = 8 |
| exclude_quotes | true | drop quoted spans from both sides of the fraction |
| exclude_citations | true | drop parenthetical and numeric citations |
| exclude_bibliography | true | drop everything after the last reference heading |
| allowlist | empty | source ids that are legitimately shared, such as the assignment brief |

---

## 2. AI-writing index

### Definition

The AI-writing index is the calibrated output of a logistic model over twelve
named features. It estimates the probability that a passage of this length,
with these statistical properties, was machine generated **in the population
the calibration set was drawn from**. That last clause is not decoration. A
model calibrated on first-year undergraduate essays says nothing reliable
about a doctoral methods chapter.

### Bands

Bands exist so that a marker never has to interpret a bare decimal.

| Index | Band | Intended reading |
| --- | --- | --- |
| < 0.20 | human | no signal worth acting on |
| 0.20 to 0.75 | unclear | the detector does not know. This is a large band on purpose |
| 0.75 to 0.90 | likely-ai | worth a conversation, never a conclusion |
| >= 0.90 | very-likely-ai | strong signal, still requires evidence of process |

The unclear band is wide because the alternative is worse. A detector forced
to choose on a coin flip produces accusations at the rate of its own error.

### Reported interval

The report carries an interval alongside the index. It is a **dispersion
estimate** over the per-segment scores, widened when the model is
uncalibrated and widened again when no reference language model was available.
It is labelled as such and is not a frequentist confidence interval. Its job
is to make the point estimate look as uncertain as it actually is.

### Calibration procedure

1. Collect a labelled set: human-written and machine-generated documents from
   the same assignment types, same disciplines, same length distribution.
2. Split by **author**, not by document, or the model memorises writing style
   and reports inflated accuracy.
3. Fit the logistic weights on the training split.
4. Fit isotonic regression on a held-out split to map raw scores to observed
   positive rates.
5. Report true-positive rate at a fixed 1 percent false-positive rate,
   stratified by discipline and by whether the writer is a non-native English
   speaker.
6. Re-run steps 1 to 5 whenever a new generation of language model appears.
   Detection accuracy decays as generators improve, and an uncalibrated
   detector that used to work is more dangerous than one that never did.

Until step 4 has run, the report says "UNCALIBRATED" and every consumer of the
JSON can see it in the "calibrated" field.

---

## 3. Review priority

A triage hint, computed from both indices, for a marker with eighty papers and
one afternoon. It orders a queue and decides nothing.

| Priority | Condition |
| --- | --- |
| read-first | similarity >= 0.30, or AI band is likely/very-likely with similarity >= 0.10 |
| worth-a-look | similarity >= 0.15 or AI index >= 0.75 |
| no-flags | everything else |

"no-flags" means the tool found nothing, not that the work is original. See
docs/ETHICS.md for why that distinction matters.
