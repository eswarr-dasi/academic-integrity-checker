# academic-integrity-checker

An originality engine that answers two questions about a submitted document:

1. **Is it copied?** Near-duplicate, quote and paraphrase detection against a reference corpus.
2. **Is it machine-written?** Statistical and model-based detection of AI-generated prose.

The output is a single **Originality Report** containing a similarity index, an AI-writing index and a list of highlighted, source-attributed matches. That is the same report shape instructors expect from commercial tools such as Turnitin, iThenticate or Copyleaks.

> **Status:** reference implementation / scaffold. The algorithms are real and runnable. What a production deployment needs on top is a large licensed reference corpus and a calibration dataset. See [Limitations](#limitations-read-this-before-you-trust-a-number).

---

## Why build this rather than call an API

Commercial detectors are closed boxes. You cannot see why a passage was flagged, you cannot tune the exclusion rules for your discipline, and you cannot audit the false-positive rate for your student population. This project makes every stage inspectable: the fingerprints, the candidate sources, the alignment spans, the feature vector behind an AI score, and the calibration curve that turns that score into a percentage.

---

## Target standards

The bar being aimed at, stated as measurable requirements rather than marketing copy.

| Capability | Commercial baseline | Target here |
| --- | --- | --- |
| Verbatim / near-duplicate recall | very high | >= 0.95 recall on >= 8-word matches |
| Paraphrase detection | moderate | >= 0.80 recall on back-translated paraphrase |
| Quote and bibliography exclusion | yes | yes, rule + citation-parser based |
| Per-match source attribution | yes | yes, with character offsets |
| AI-writing detection | reported, opaque | reported with confidence band + feature breakdown |
| AI false-positive rate on human text | disputed | published per-cohort, target <= 1% at the flag threshold |
| Throughput | batch, seconds | < 5 s for a 5000-word paper against a 10M-document index |
| Report format | proprietary HTML | JSON (machine) + HTML (human), both versioned |

The false-positive target is the important one. A plagiarism flag is an accusation, and the cost of a wrong flag is borne by a student, so the system is tuned for precision and reports uncertainty instead of hiding it.

---

## Architecture

```
            submission (pdf | docx | txt | md)
                          |
                   [ 1. ingest ]  text + layout + per-char source offsets
                          |
                   [ 2. normalize ]  unicode fold, dehyphenate, sentence split,
                          |           quote / citation / bibliography tagging
                          |
         +----------------+----------------+
         |                                 |
  [ 3. similarity engine ]          [ 4. ai-writing engine ]
         |                                 |
  a. k-shingle + rolling hash       a. token log-prob features
  b. winnowing -> fingerprints         (perplexity, burstiness, rank)
  c. MinHash + LSH candidate       b. stylometric features
     retrieval                        (sentence length variance, function
  d. embedding rerank for              word ratio, punctuation entropy)
     paraphrase                     c. calibrated classifier -> probability
  e. Smith-Waterman span               + confidence band
     alignment                         |
         |                             |
         +----------------+------------+
                          |
                   [ 5. scoring ]  similarity index, ai index, exclusions
                          |
                   [ 6. report ]  JSON + HTML, highlighted spans, sources
```

### 1. Ingest
Extracts text while preserving a mapping from every output character back to its page and offset in the original file, so the report can highlight the right region of a PDF. Handles PDF, DOCX, ODT, RTF, TXT and Markdown.

### 2. Normalize
Case and unicode folding, ligature expansion, soft-hyphen and line-break dehyphenation, whitespace collapse, then sentence segmentation. Runs three taggers whose spans are later excluded from the similarity index on request: quoted material, in-text citations, and the reference list / bibliography.

### 3. Similarity engine
Content-defined chunking so that an inserted word does not shift every downstream fingerprint:

- **k-shingles** over normalized word tokens, default k = 5.
- **Rabin-Karp rolling hash** for O(n) shingle hashing.
- **Winnowing** with window w = 4 to select a stable, position-independent fingerprint subset. Guarantees detection of any shared substring of length >= k + w - 1 tokens.
- **MinHash** signatures (128 permutations) plus **banded LSH** for sub-linear candidate retrieval from the corpus index.
- **Embedding rerank** on sentence vectors to catch paraphrase that shares no n-grams.
- **Smith-Waterman local alignment** on the candidate pairs to produce exact, human-readable match spans with start and end offsets.

### 4. AI-writing engine
Deliberately not a single black-box classifier. Three feature families feed one calibrated model:

- **Likelihood features** from a reference language model: mean token log-probability (perplexity), log-rank, and the variance of per-sentence perplexity (burstiness). Human writing is bursty; sampled text tends to sit in a narrow band near the model mode.
- **Stylometric features**: sentence-length distribution, function-word ratios, type-token ratio, punctuation entropy, discourse-marker frequency, hapax rate.
- **Perturbation features**: a DetectGPT-style curvature probe. Rewrite the passage slightly and measure whether log-likelihood drops. Machine text sits on a local maximum, human text does not.

The classifier is logistic regression over those features, which keeps the decision auditable, then **isotonic calibration** so that a reported 0.90 actually means 90% of such passages were machine-written in the validation set. Scores are returned as bands (`human`, `unclear`, `likely-ai`, `very-likely-ai`), never as a bare verdict.

### 5. Scoring
The similarity index is the fraction of *scorable* words covered by at least one accepted match, where scorable excludes anything the tagger marked as quotation, citation or bibliography when those exclusions are enabled. Overlapping matches are merged before counting, so the index cannot exceed 100%. Matches shorter than the minimum span, or against sources on the allow-list (the students own prior submission, the assignment prompt, a shared template), are dropped before scoring.

### 6. Report
Versioned JSON is the source of truth; the HTML view is rendered from it.

```json
{
  "report_version": "1.0",
  "document": { "id": "sub_8f21", "word_count": 1842, "scorable_words": 1655 },
  "similarity": {
    "index": 0.17,
    "matched_words": 281,
    "matches": [
      {
        "source_id": "corpus:arxiv:2104.08653",
        "source_title": "A Survey of Near-Duplicate Detection",
        "submission_span": [1204, 1487],
        "source_span": [88120, 88401],
        "match_type": "verbatim",
        "similarity": 0.98,
        "excluded": false,
        "exclusion_reason": null
      }
    ]
  },
  "ai_writing": {
    "index": 0.62,
    "band": "unclear",
    "confidence_interval": [0.41, 0.80],
    "segments": [
      { "span": [0, 640], "score": 0.22, "band": "human" },
      { "span": [641, 1840], "score": 0.81, "band": "likely-ai" }
    ],
    "features": {
      "mean_log_prob": -2.41,
      "burstiness": 0.38,
      "curvature": 0.71,
      "sentence_len_variance": 12.4
    }
  },
  "settings": { "exclude_quotes": true, "exclude_bibliography": true, "min_match_words": 8 },
  "caveats": ["ai_writing index is probabilistic and must not be used as sole evidence"]
}
```

---

## Repository layout

```
src/aic/
  ingest.py       document loading, text extraction, offset mapping
  normalize.py    folding, dehyphenation, sentence split, quote/citation tagging
  fingerprint.py  k-shingles, Rabin-Karp rolling hash, winnowing, MinHash
  index.py        corpus store + banded LSH retrieval
  similarity.py   candidate rerank, Smith-Waterman alignment, span merging
  ai_detect.py    likelihood + stylometric + curvature features, calibrated model
  scoring.py      similarity index, ai index, exclusion rules, bands
  report.py       JSON schema + HTML renderer
  pipeline.py     end-to-end orchestration
  cli.py          command line entry point
api/main.py       FastAPI service
tests/            unit tests per module
docs/             ARCHITECTURE.md, SCORING.md, EVALUATION.md, ETHICS.md
samples/          small public-domain corpus for smoke tests
```

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# build an index from a corpus directory
python -m aic.cli index build --corpus samples/corpus --out .aic/index

# check one document
python -m aic.cli check samples/submissions/essay.txt --index .aic/index --html report.html

# serve the API
uvicorn api.main:app --reload
```

### API

```
POST /v1/submissions        multipart file upload -> { submission_id }
GET  /v1/reports/{id}       originality report JSON
GET  /v1/reports/{id}/html  rendered report
POST /v1/corpus/documents   add a document to the reference index
```

---

## Evaluation

Do not ship a detector you have not measured. `docs/EVALUATION.md` defines the harness:

- **Similarity**: PAN plagiarism-detection corpora, plus synthetic verbatim / shuffled / back-translated injections at known offsets. Reported as precision, recall and character-level F1 per obfuscation class.
- **AI detection**: balanced human and generated sets across several model families and decoding settings, stratified by discipline and by whether the writer is a non-native English speaker. Reported as ROC-AUC plus true-positive rate at a fixed 1% false-positive rate, which is the only operating point that matters in practice.
- **Drift**: the AI detector is re-evaluated whenever a new generation of language model appears, because detection accuracy decays as generators improve.

---

## Limitations, read this before you trust a number

- **Coverage is the ceiling on plagiarism detection.** A match can only be found against documents in the index. Without licensed access to journal archives and a student-paper repository, recall against real-world sources is far below a commercial product regardless of how good the algorithms are.
- **AI detection is probabilistic and it is not evidence of misconduct.** Published false-positive rates are non-trivial and are measurably worse for non-native English writers and for formulaic technical prose. This project therefore reports bands and intervals, refuses to output a binary verdict, and expects a human to make the decision.
- **A high similarity index is not plagiarism.** Correctly quoted and cited material, boilerplate methods sections, shared assignment prompts and reference lists all raise the number legitimately. The index is a pointer for a reader, not a judgment.
- **Adversarial evasion works.** Synonym substitution, sentence reordering, homoglyph injection and humanizer tools all degrade detection. Homoglyph and invisible-character normalization is implemented; the rest is an open arms race.

---

## Roadmap

- [x] Repository and architecture
- [ ] Ingest + normalization with offset preservation
- [ ] Fingerprinting and LSH index
- [ ] Alignment and span merging
- [ ] AI feature extraction and calibration
- [ ] Report JSON schema + HTML renderer
- [ ] FastAPI service and persistence
- [ ] Evaluation harness and published metrics
- [ ] LMS integration (LTI 1.3)

---

## License

MIT. See [LICENSE](LICENSE).
