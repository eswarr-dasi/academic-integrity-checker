# Evaluation

A detector with no published numbers is a rumour. This document specifies the
harness, the datasets, the metrics and the operating points. Until the tables
at the bottom are filled in, the correct summary of this project is "the
algorithms are implemented and the accuracy is unknown".

---

## 1. Similarity engine

### Datasets

| Set | Source | What it tests |
| --- | --- | --- |
| PAN plagiarism corpora | public research corpora with character-level ground truth | verbatim, obfuscated and summary plagiarism |
| synthetic injections | built by tools/make_injections.py from a public-domain corpus | recall at known offsets and lengths |
| clean control set | documents with no injected reuse | false-positive rate |
| adversarial set | synonym substitution, sentence reordering, homoglyphs, invisible characters, back-translation | evasion resistance |

Synthetic injection is the workhorse because the ground truth offsets are
exact, so recall can be measured per obfuscation class and per length rather
than as a single averaged number that hides the failure modes.

### Metrics

Character-level, following the PAN convention, because a match that finds the
right paragraph but the wrong sentence boundary is only partly useful:

- **precision** = matched characters that are truly reused / matched characters
- **recall** = matched characters that are truly reused / truly reused characters
- **granularity** = detections per true case (1.0 is ideal, higher means one
  passage was reported as several fragments)
- **plagdet** = the PAN combined score, F1 discounted by granularity

Report these broken out by obfuscation class, never pooled.

### Targets

| Obfuscation class | Recall target | Precision target |
| --- | --- | --- |
| verbatim, >= 8 words | 0.95 | 0.99 |
| light edit (synonyms, word order) | 0.85 | 0.97 |
| heavy paraphrase | 0.60 | 0.90 |
| back-translated | 0.55 | 0.90 |
| summary | 0.30 | 0.85 |
| clean control (false positives) | n/a | index <= 0.03 |

Precision targets are higher than recall targets throughout. A missed copy is
a marking failure, a false match is an accusation.

### Throughput

Measured as wall clock for one 5000 word submission against a 10 million
document index, single process, warm index:

| Stage | Budget |
| --- | --- |
| ingest + normalize | 300 ms |
| fingerprint | 100 ms |
| candidate retrieval | 400 ms |
| alignment (50 candidates) | 3000 ms |
| scoring + report | 200 ms |
| **total** | **< 5 s** |

---

## 2. AI-writing detector

### Datasets

Balanced human and generated sets, stratified along every axis that is known
to move the score:

- **generator family and decoding settings**: several model families, greedy
  and sampled, low and high temperature, with and without a humanizer pass.
- **discipline**: humanities essay, lab report, literature review, reflective
  writing. Formulaic technical prose is the hardest case and must be its own
  stratum.
- **writer background**: first-language and additional-language English
  writers, reported separately. This is the single most important split in
  the whole harness.
- **length**: 150, 300, 600, 1200 words. Accuracy below roughly 300 words is
  poor and the report already warns about it.
- **mixed authorship**: human draft with machine-edited paragraphs, which is
  the realistic case and the one single-score detectors handle worst.

### Splitting rule

Split by **author**, never by document. A random document split lets the model
learn individual writing styles and inflates every metric, sometimes by tens
of points.

### Metrics

- **ROC-AUC**, for a summary that is threshold independent.
- **TPR at 1 percent FPR**. This is the number that matters. Any institution
  running a detector at scale will be operating at a very low false-positive
  tolerance, and average accuracy at the 50 percent threshold is irrelevant
  to that regime.
- **Expected calibration error** in ten bins, before and after isotonic
  calibration.
- **FPR by writer background**, as an explicit fairness measurement. A
  detector whose false positives concentrate on additional-language writers
  is not usable regardless of its aggregate score.
- **Segment-level accuracy** on the mixed authorship set.

### Targets

| Metric | Target |
| --- | --- |
| ROC-AUC, 600+ words, in-distribution generators | >= 0.90 |
| TPR at 1 percent FPR, 600+ words | >= 0.55 |
| TPR at 1 percent FPR, unseen generator family | >= 0.30 |
| Expected calibration error after isotonic | <= 0.05 |
| FPR gap between writer-background strata | <= 1 percentage point |
| Segment F1 on mixed authorship | >= 0.60 |

The unseen-generator target is deliberately low. Out-of-distribution
generalisation for this task is weak and claiming otherwise is how detectors
end up causing harm.

---

## 3. Drift monitoring

Detection accuracy is not a fixed property. Re-run the whole AI harness:

- whenever a major new generation of language model is released;
- quarterly regardless;
- whenever the observed score distribution on live submissions shifts by more
  than 0.05 in mean, which is a cheap early warning that the population or
  the generators changed.

If a re-run cannot be done, the correct action is to raise the flag threshold
or to switch the AI index off, not to keep reporting a stale number.

---

## 4. Results

Empty until the harness has been run. Do not fill these in by hand.

### Similarity

| Obfuscation class | Precision | Recall | Granularity | Plagdet |
| --- | --- | --- | --- | --- |
| verbatim | - | - | - | - |
| light edit | - | - | - | - |
| heavy paraphrase | - | - | - | - |
| back-translated | - | - | - | - |

### AI writing

| Stratum | n | ROC-AUC | TPR at 1% FPR | ECE |
| --- | --- | --- | --- | --- |
| all | - | - | - | - |
| first-language English | - | - | - | - |
| additional-language English | - | - | - | - |
| unseen generator | - | - | - | - |
