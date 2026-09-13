"""academic-integrity-checker: an inspectable originality engine.

Two independent engines, one report:

* similarity  - fingerprint + alignment based detection of copied text against
  a reference corpus. See aic.fingerprint, aic.index, aic.similarity.
* ai_detect   - calibrated detection of machine-generated prose from
  likelihood, stylometric and curvature features. See aic.ai_detect.

Nothing in this package emits a verdict. Both engines emit scores, spans and
the features behind them, aic.scoring turns those into indices and confidence
bands, and a human makes the decision.
"""

__version__ = "0.1.0"

__all__ = [
    "ingest",
    "normalize",
    "fingerprint",
    "index",
    "similarity",
    "ai_detect",
    "scoring",
    "report",
    "pipeline",
]
