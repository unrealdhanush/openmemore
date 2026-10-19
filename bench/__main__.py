"""`uv run python -m bench <dataset>`: measure OpenMCP's retrieval (SPEC.md M7.6).

    uv run python -m bench longmemeval-s --limit 100      # an even spread over question types
    uv run python -m bench locomo
    uv run python -m bench longmemeval-s --keywords-only  # without the embedding model
    uv run python -m bench locomo --judge-endpoint http://127.0.0.1:11434/v1 --judge-model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from bench import datasets, runner

RESULTS = Path(__file__).resolve().parent / "results"
LOG = Path(__file__).resolve().parent / "RESULTS.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench", description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(datasets.SOURCES))
    parser.add_argument("--limit", type=int, help="At most this many histories, spread evenly over question types.")
    parser.add_argument("--k", default="5,10", help="Cutoffs, comma-separated (default 5,10).")
    parser.add_argument("--keywords-only", action="store_true", help="Without the embedding model.")
    parser.add_argument("--model-dir", default=os.environ.get("OPENMCP_MODEL_DIR")
                        or str(Path.home() / ".cache" / "openmcp-test-models"), help="Where the embedding model is cached.")
    parser.add_argument("--judge-endpoint", help="Also answer and judge each question with this model endpoint.")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-api", default="responses", choices=["responses", "chat-completions"])
    parser.add_argument("--record", action="store_true", help="Append the summary to bench/RESULTS.md.")
    parser.add_argument("--set", action="append", default=[], metavar="MODULE.NAME=VALUE",
                        help="Override a constant for tuning, e.g. openmcp.tools.search.CHAT_ACTIVATION_WEIGHT=0.002.")
    args = parser.parse_args(argv)
    ks = sorted({int(k) for k in args.k.split(",")})
    settings = apply_settings(args.set)

    if args.keywords_only:
        runner.use_embedder(None)
        embedder_name = "none (keywords only)"
    else:
        from openmcp import embed

        model = embed.FastEmbedder(Path(args.model_dir), allow_download=True)
        runner.use_embedder(runner.CachingEmbedder(model, runner.cache_path()))
        embedder_name = embed.MODEL
    answer = None
    if args.judge_endpoint:
        from bench.judge import answerer
        from openmcp import llm

        if not args.judge_model:
            parser.error("--judge-model is required with --judge-endpoint")
        answer = answerer(llm.Model(endpoint=args.judge_endpoint, model=args.judge_model, api=args.judge_api,
                                    api_key=os.environ.get("OPENMCP_MODEL_API_KEY")), max(ks))

    haystacks = runner.sample(datasets.load(args.dataset), args.limit)
    print(f"{args.dataset}: {len(haystacks)} histories, {sum(len(h.questions) for h in haystacks)} questions, "
          f"embedder {embedder_name}", file=sys.stderr)

    def progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 20) == 0:
            print(f"  {done}/{total}", file=sys.stderr, flush=True)

    report = runner.run(args.dataset, haystacks, ks, embedder_name=embedder_name, answer=answer, progress=progress)
    if args.limit:
        report["limit"] = args.limit
    if settings:
        report["settings"] = settings
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"{stamp}-{args.dataset}-{report['commit']}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    table = render(report)
    print(table)
    print(f"report: {out}", file=sys.stderr)
    if args.record:
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n## {report['ran_at'][:10]} {args.dataset}"
                      f"{f' (limit {args.limit})' if args.limit else ''} at {report['commit']}"
                      f"{''.join(f', {k}={v}' for k, v in settings.items())}\n\n{table}\n")
    return 0


def apply_settings(pairs: list[str]) -> dict[str, object]:
    """Set module constants from MODULE.NAME=VALUE pairs (VALUE a Python literal)."""
    import ast
    import importlib

    applied: dict[str, object] = {}
    for pair in pairs:
        target, _, raw = pair.partition("=")
        module_name, _, name = target.rpartition(".")
        module = importlib.import_module(module_name)
        if not hasattr(module, name):
            raise SystemExit(f"{module_name} has no {name}")
        value = ast.literal_eval(raw)
        setattr(module, name, value)
        applied[name] = value
    return applied


def render(report: dict) -> str:
    ks = report["ks"]
    columns = [f"{m}@{k}" for k in ks for m in ("session_recall", "any_hit", "ndcg", "turn_recall")]
    columns += ["correct"] if any("correct" in s for s in report["summary"].values()) else []
    lines = [
        f"Embedder: {report['embedder']}. {report['haystacks']} histories, {report['questions']} questions. "
        f"Ingest {report['latency']['ingest_s_mean']} s per history; query p50 {report['latency']['query_ms_p50']} ms, "
        f"p95 {report['latency']['query_ms_p95']} ms.",
        "",
        "| type | n | " + " | ".join(columns) + " |",
        "|---|---:|" + "---:|" * len(columns),
    ]
    for name, row in report["summary"].items():
        cells = [f"{row[c]:.3f}" if c in row else "–" for c in columns]
        lines.append(f"| {name} | {row['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
