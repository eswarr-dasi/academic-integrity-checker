"""Fitted parameters for the style only AI-writing detector.

Provenance
----------
Dataset: HC3 (Hello-SimpleAI/HC3), read through the public Hugging Face
datasets-server REST API. Each row carries a human answer and a ChatGPT
answer to the same question, which controls for topic.

Sample: 2,640 answers of at least 120 words, balanced 1,320 human against
1,320 machine, with the machine half stratified to match the human half mix
of HC3 sources (reddit_eli5, finance, medicine, wiki_csai). Split in two by a
seeded shuffle: one half to fit on, one half never touched until the numbers
in METRICS were measured.

Method: stylometric features only, measured per scoring window of
SEGMENT_WORDS words, standardised with the MEANS and STDS below, then a
ridge penalised logistic regression solved by Newton-Raphson. Window scores
are mapped to probabilities by an isotonic calibrator fitted on the same
half. A document index is the word weighted mean of its window scores.

Read METRICS before quoting a number at anybody. At the likely-ai boundary
this model still mislabels about one human document in fourteen, and on
formal technical prose (the HC3 medicine split) it is far worse than that.

Scope: fitted for the configuration with no language model attached.
Supplying a LanguageModel adds features this fit never saw, so the profile
has to be refitted before those numbers mean anything.
"""

from __future__ import annotations

PROFILE_ID = "hc3-style-v1"
DATASET = "Hello-SimpleAI/HC3"
DATASET_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3"
FITTED_ON = "2026-09-14"

FEATURES: tuple[str, ...] = (
    "sent_len_mean",
    "sent_len_cv",
    "type_token_ratio",
    "hapax_ratio",
    "function_word_ratio",
    "punct_entropy",
    "discourse_marker_rate",
)

BIAS = -0.20956

WEIGHTS: dict[str, float] = {
    "sent_len_mean": 0.53231,
    "sent_len_cv": -0.476628,
    "type_token_ratio": -0.552988,
    "hapax_ratio": -0.342835,
    "function_word_ratio": -0.042403,
    "punct_entropy": -1.414988,
    "discourse_marker_rate": 0.299503,
}

MEANS: dict[str, float] = {
    "sent_len_mean": 22.833767,
    "sent_len_cv": 0.459068,
    "type_token_ratio": 0.627374,
    "hapax_ratio": 0.452414,
    "function_word_ratio": 0.472133,
    "punct_entropy": 1.888929,
    "discourse_marker_rate": 0.002449,
}

STDS: dict[str, float] = {
    "sent_len_mean": 11.252466,
    "sent_len_cv": 0.202412,
    "type_token_ratio": 0.098899,
    "hapax_ratio": 0.121797,
    "function_word_ratio": 0.055512,
    "punct_entropy": 0.596965,
    "discourse_marker_rate": 0.005001,
}

# Isotonic step function, runs of equal values merged. Maps the logistic
# output of one scoring window to an empirical probability.
CALIBRATION_BREAKPOINTS: list[float] = [
    0.016951, 0.083116, 0.092362, 0.111049, 0.1152, 0.144535, 0.214769,
    0.256052, 0.279542, 0.385707, 0.408655, 0.422646, 0.465908, 0.468712,
    0.601939, 0.630176, 0.706211, 0.712777, 0.76816, 0.772931, 0.800222,
    0.815829, 0.872429, 0.944432, 0.949359, 0.998739, 0.999279,
]

CALIBRATION_VALUES: list[float] = [
    0.0, 0.043478, 0.066667, 0.092105, 0.111111, 0.142857, 0.149701,
    0.164557, 0.245902, 0.272727, 0.357143, 0.4, 0.482353, 0.5, 0.550661,
    0.657895, 0.675214, 0.714286, 0.792453, 0.833333, 0.836066, 0.866667,
    0.904762, 0.921296, 0.923077, 0.932432, 1.0,
]

# Measured on the held out half only. tpr is the share of machine documents
# at or above the threshold, fpr the share of human documents at or above it.
METRICS: dict[str, object] = {
    "fit_documents": 1320,
    "heldout_documents": 1320,
    "fit_windows": 2265,
    "heldout_windows": 2293,
    "window_auc": 0.8821,
    "document_auc": 0.9096,
    "document_accuracy_at_0_5": 0.8409,
    "operating_points": (
        {"threshold": 0.20, "tpr": 0.9811, "fpr": 0.5131},
        {"threshold": 0.50, "tpr": 0.8754, "fpr": 0.1910},
        {"threshold": 0.75, "tpr": 0.6309, "fpr": 0.0700},
        {"threshold": 0.90, "tpr": 0.3864, "fpr": 0.0321},
    ),
    "fpr_by_source_at_0_75": {
        "reddit_eli5": 0.0522,
        "wiki_csai": 0.0571,
        "finance": 0.0640,
        "medicine": 0.3158,
    },
    "fpr_by_length_at_0_75": {
        "one_window": 0.1356,
        "two_windows": 0.0232,
        "three_or_more_windows": 0.0152,
    },
}

HEADLINE_CAVEAT = (
    "Calibrated on HC3 question answering prose, not on student coursework. "
    "On held out data 7.0 percent of human documents reached the likely-ai "
    "boundary, and 31.6 percent of human medical writing did."
)
