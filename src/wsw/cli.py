"""Command line: ``wsw fetch | lfs | app | bench | report``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import sys
from pathlib import Path

import numpy as np

from wsw.data import available_datasets, downloaded_datasets, load_dataset, wrench


def _cmd_fetch(args: argparse.Namespace) -> int:
    names = wrench.WRENCH_DATASETS if "all" in args.datasets else tuple(args.datasets)
    observed: dict[str, str] = {}
    for name in names:
        observed.update(wrench.fetch(name, record=args.record))
    if args.record:
        print(json.dumps(dict(sorted(observed.items())), indent=2))
    print(f"data directory: {wrench.wrench_dir()}")
    return 0


def _cmd_lfs(args: argparse.Namespace) -> int:
    from wsw.diagnostics import lf_stats, matrix_summary

    ds = load_dataset(args.dataset)
    s = matrix_summary(ds.train.L, ds.k)
    print(
        f"{ds.name}: {s.n_rows} train rows, {s.n_lfs} LFs, coverage {s.coverage:.3f}, "
        f"{s.mean_votes_per_covered_row:.2f} votes per covered row, "
        f"conflicting rows {s.conflict_rate:.3f}"
    )
    header = ("LF", "votes for", "coverage", "overlap", "conflict", "gold acc")
    print("{:<16} {:<18} {:>9} {:>8} {:>9} {:>8}".format(*header))
    stats = lf_stats(ds.train.L, ds.k, ds.lf_names, np.arange(len(ds.train)), ds.train.y)
    for st in stats:
        votes = ",".join(ds.label_names[c] for c in st.polarity) or "-"
        acc = "-" if st.empirical_accuracy is None else f"{st.empirical_accuracy:.3f}"
        print(
            f"{st.name:<16} {votes[:18]:<18} {st.coverage:>9.3f} {st.overlap:>8.3f} "
            f"{st.conflict:>9.3f} {acc:>8}"
        )
    print("(gold acc scores each LF against the training split's gold labels)")
    return 0


def _cmd_app(args: argparse.Namespace) -> int:
    from wsw.app import serve

    ready = downloaded_datasets()
    if args.dataset not in ready:
        print(
            f"{args.dataset} is not downloaded; run `wsw fetch {args.dataset}` "
            f"(ready now: {', '.join(ready)})",
            file=sys.stderr,
        )
        return 2
    serve(
        args.dataset,
        host=args.host,
        port=args.port,
        max_rows=args.max_rows or None,
        label_model=args.label_model,
        strategy=args.strategy,
    )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from wsw.benchmark.runner import SUITES, run_suite

    suite = SUITES[args.suite]
    if args.datasets:
        suite = dataclasses.replace(suite, datasets=tuple(args.datasets))
    if args.seeds:
        suite = dataclasses.replace(suite, seeds=tuple(range(args.seeds)))
    out = (
        Path(args.out)
        if args.out
        else Path("bench/results")
        / ("full.json" if suite.name == "full" else f"smoke-{'-'.join(suite.datasets)}.json")
    )
    command = "wsw " + shlex.join(args.argv)
    run_suite(suite, out, command, use_snorkel=not args.no_snorkel)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from wsw.benchmark.report import learning_curve_svg, markdown_report

    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the tables use "±"
    print(markdown_report(res), end="")
    if args.svg:
        Path(args.svg).parent.mkdir(parents=True, exist_ok=True)
        Path(args.svg).write_text(learning_curve_svg(res), encoding="utf-8")
        print(f"\n(wrote {args.svg})", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from wsw.active.strategies import STRATEGIES
    from wsw.labelmodels import LABEL_MODELS

    parser = argparse.ArgumentParser(prog="wsw", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download WRENCH datasets and verify their sha256")
    p.add_argument("datasets", nargs="+", choices=[*wrench.WRENCH_DATASETS, "all"])
    p.add_argument(
        "--record", action="store_true", help="print digests of files with none committed"
    )
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("lfs", help="print LF diagnostics for a dataset")
    p.add_argument("dataset", choices=available_datasets())
    p.set_defaults(func=_cmd_lfs)

    p = sub.add_parser("app", help="start the web app")
    p.add_argument("--dataset", default="demo", choices=available_datasets())
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-rows", type=int, default=5000, help="cap on training rows (0 = all)")
    p.add_argument("--label-model", default="dawid_skene", choices=sorted(LABEL_MODELS))
    p.add_argument("--strategy", default="conflict_entropy", choices=sorted(STRATEGIES))
    p.set_defaults(func=_cmd_app)

    p = sub.add_parser("bench", help="run the benchmark and write a results JSON")
    p.add_argument("--suite", default="full", choices=["full", "smoke"])
    p.add_argument("--datasets", nargs="+", choices=available_datasets())
    p.add_argument("--seeds", type=int, help="number of seeds (0..n-1)")
    p.add_argument("--out", help="results path (default bench/results/<suite>...json)")
    p.add_argument("--no-snorkel", action="store_true", help="skip the Snorkel reference")
    p.set_defaults(func=_cmd_bench)

    p = sub.add_parser("report", help="render Markdown tables (and an SVG) from a results JSON")
    p.add_argument("results")
    p.add_argument("--svg", help="also write the learning-curve figure here")
    p.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    args.argv = argv
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
