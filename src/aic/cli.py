"""Command line interface. Standard library only, so it runs anywhere.

    python -m aic.cli index build --corpus samples/corpus --out .aic/index
    python -m aic.cli index stats --index .aic/index
    python -m aic.cli check paper.pdf --index .aic/index --html report.html
    python -m aic.cli forget sha256:abc123 --index .aic/index

The --lm flag opts in to the reference language model. Without it the AI
detector runs on stylometric features only, and says so in the report instead
of pretending the score is as good.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import Engine, build_index
from .scoring import ScoringSettings


def _settings_from_args(args: argparse.Namespace) -> ScoringSettings:
    return ScoringSettings(
        exclude_quotes=not args.include_quotes,
        exclude_citations=not args.include_citations,
        exclude_bibliography=not args.include_bibliography,
        min_match_words=args.min_words,
        allowlist=set(args.allow or []),
    )


def _language_model(name: str | None):
    if not name:
        return None
    from .ai_detect import HuggingFaceLM

    return HuggingFaceLM(model_name=name)


def cmd_index_build(args: argparse.Namespace) -> int:
    stats = build_index(args.corpus, args.out, k=args.k, w=args.w)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_index_stats(args: argparse.Namespace) -> int:
    engine = Engine.load(args.index)
    print(json.dumps(engine.index.stats(), indent=2))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    engine = Engine.load(args.index, lm=_language_model(args.lm))
    report = engine.check_path(
        args.path,
        settings=_settings_from_args(args),
        add_to_corpus=args.add_to_corpus,
    )
    payload = report.to_dict()

    if args.json:
        Path(args.json).write_text(report.to_json())
    if args.html:
        Path(args.html).write_text(report.to_html())

    sim = payload["similarity"]
    ai = payload["ai_writing"]
    print(
        "similarity %.1f%%  (%d / %d scorable words, %d sources)"
        % (
            100 * sim["index"],
            sim["matched_words"],
            sim["scorable_words"],
            len(sim["sources"]),
        )
    )
    print(
        "ai-writing %.1f%%  band=%s  range %.0f%%-%.0f%%  %s"
        % (
            100 * ai["index"],
            ai["band"],
            100 * ai["confidence_interval"][0],
            100 * ai["confidence_interval"][1],
            "calibrated" if ai["calibrated"] else "UNCALIBRATED",
        )
    )
    print("review priority: %s" % payload["review_priority"])
    for line in payload["caveats"]:
        print("  note: %s" % line)

    if args.print_json:
        print(report.to_json())
    if args.add_to_corpus:
        engine.save(args.index)
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    engine = Engine.load(args.index)
    if args.doc_id not in engine.index.meta:
        print("no such document: %s" % args.doc_id, file=sys.stderr)
        return 1
    engine.forget(args.doc_id)
    engine.save(args.index)
    print("removed %s, index now holds %d documents" % (args.doc_id, len(engine.index)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aic", description="Academic integrity checker."
    )
    subs = parser.add_subparsers(dest="command", required=True)

    index_parser = subs.add_parser("index", help="build or inspect a corpus index")
    index_subs = index_parser.add_subparsers(dest="index_command", required=True)

    build = index_subs.add_parser("build", help="fingerprint a directory of sources")
    build.add_argument("--corpus", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--k", type=int, default=5, help="shingle size in words")
    build.add_argument("--w", type=int, default=4, help="winnowing window")
    build.set_defaults(func=cmd_index_build)

    stats = index_subs.add_parser("stats", help="print index statistics")
    stats.add_argument("--index", required=True)
    stats.set_defaults(func=cmd_index_stats)

    check = subs.add_parser("check", help="produce an originality report")
    check.add_argument("path")
    check.add_argument("--index", required=True)
    check.add_argument("--json", help="write the report JSON here")
    check.add_argument("--html", help="write the rendered report here")
    check.add_argument("--print-json", action="store_true")
    check.add_argument("--lm", help="reference LM name, for example gpt2-medium")
    check.add_argument("--min-words", type=int, default=8)
    check.add_argument("--include-quotes", action="store_true")
    check.add_argument("--include-citations", action="store_true")
    check.add_argument("--include-bibliography", action="store_true")
    check.add_argument("--allow", action="append", help="source id to ignore")
    check.add_argument(
        "--add-to-corpus",
        action="store_true",
        help="index this submission after checking it",
    )
    check.set_defaults(func=cmd_check)

    forget = subs.add_parser("forget", help="delete a document from the index")
    forget.add_argument("doc_id")
    forget.add_argument("--index", required=True)
    forget.set_defaults(func=cmd_forget)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
