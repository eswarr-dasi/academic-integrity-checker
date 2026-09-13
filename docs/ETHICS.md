# Operating limits

This is a document about harm, not about compliance. The engineering
decisions in this repository were made with the contents of this file in mind,
and changing them without reading it will produce a tool that hurts people.

---

## 1. The asymmetry that shapes everything

A missed case of plagiarism costs a marker some accuracy in one grade. A false
accusation costs a student a disciplinary record, sometimes a visa, sometimes a
degree. The two errors are not comparable, so the system is not tuned to
maximise accuracy. It is tuned for precision, it reports uncertainty, and it
refuses to output a verdict.

Concretely, in this codebase:

- the AI detector returns bands and an interval, and there is a test asserting
  that no field named "verdict" exists;
- the unclear band spans 0.20 to 0.75, which is wide enough that many real
  documents land in it, which is the honest outcome;
- an uncalibrated model announces itself in the JSON and in the rendered
  report rather than quietly producing a number that looks like a probability;
- excluded matches are shown with a reason rather than deleted, so the reader
  can disagree with the exclusion policy.

---

## 2. Known bias in AI-writing detection

The likelihood features that drive this class of detector measure how
predictable text is to a reference language model. Several populations write
predictable prose for reasons that have nothing to do with machine
generation:

- **Writers working in an additional language.** Simpler syntax, a smaller
  and more frequent vocabulary, and more conventional phrasing all raise
  likelihood scores. Published studies have found large disparities in
  false-positive rates for non-native English writers. This is the single
  most serious known defect of the approach.
- **Formulaic technical and scientific prose.** A methods section is supposed
  to be predictable. Lab reports, clinical write-ups and legal drafting all
  score high for good reasons.
- **Autistic and neurodivergent writers**, and writers using assistive
  writing software, whose style may be more regular than the calibration
  population.
- **Writers who followed the rubric closely.** Structural advice pushes prose
  toward the mode.

Mitigations implemented here: bands instead of thresholds, an interval,
per-segment rather than whole-document scores, a required per-cohort
false-positive measurement in docs/EVALUATION.md, and a hard requirement that
the calibration set match the population being scored.

Mitigations not implemented and left to the deploying institution: a policy
that a detector score alone is never sufficient grounds for an allegation.

---

## 3. What a similarity index does not tell you

- **A high index is not plagiarism.** Correct quotation, standard methods
  language, shared assignment prompts, templates and reference lists all
  raise it.
- **A low index is not originality.** Coverage is the ceiling. Nothing can be
  matched against sources that are not in the index, which includes most
  paywalled scholarship, most of the web, contract-written essays, and work
  copied from a friend who never submitted.
- **The index cannot see intent.** A student who paraphrased badly and a
  student who copied deliberately produce the same number.

---

## 4. Student data

Building a reference corpus out of student work is the feature that makes
these systems effective and the one that carries the most risk. If this
project is deployed:

- Have a lawful basis for retention and state it to students before their
  first submission, not in a terms-of-service update afterwards.
- Set a retention period and enforce it automatically. "Forever" is not a
  retention period.
- Support deletion. The engine exposes Engine.forget and the API exposes
  DELETE /v1/corpus/documents/{doc_id} precisely so that a deletion request
  is a supported operation rather than a database migration.
- Do not make indexing a submission the default. In this codebase
  add_to_corpus defaults to false, and indexing happens after scoring so a
  resubmission is never matched against itself.
- Keep reports access controlled. An originality report is an allegation in
  waiting and should not be readable by other students, other markers, or
  analytics pipelines.

---

## 5. Process over scores

The only robust evidence of authorship is process: drafts, version history,
notes, an oral defence of the argument. A detector can tell a marker where to
look. It cannot answer the question, and an institution that treats it as an
answer has outsourced a judgement it is not allowed to outsource.

Suggested sequence when a report flags something:

1. Read the flagged spans. Most flags resolve here as quotation or
   boilerplate.
2. Check the sources actually say what the match claims.
3. Talk to the student before any process starts, and show them the report.
4. Ask about process, not about the score.
5. If a case proceeds, the evidence is the passages and the conversation. The
   percentages are context, not exhibits.

---

## 6. Uses this project will not support

Feature requests along these lines will be declined:

- a single pass/fail output, or anything that removes the human decision;
- surveillance features such as keystroke logging, webcam proctoring or
  browser monitoring;
- ranking or scoring of students, cohorts or staff from detector output;
- suppressing the caveats block in the report;
- exporting student text to third-party services that retain it.

---

## 7. Further reading

Rather than citing specific figures that will date badly, the areas to read
before deploying anything in this repository: the PAN series of plagiarism
detection evaluations, the DetectGPT line of work on likelihood curvature,
the published studies on bias in GPT detectors against non-native English
writers, and your own institution's academic misconduct regulations, which
almost certainly say that automated tools are indicative and not
determinative.
