"""Print a results table:  python -m eval.cli --dataset scifact --retrievers bm25 --k 10

--format markdown prints a table ready to paste into the README,
--format json prints every metric plus the per-query scores.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from eval.runner import RetrieverReport, evaluate
from retrieval.registry import REGISTRY


def text_table(dataset: str, k: int, reports: list[RetrieverReport]) -> str:
    header = f"{'retriever':<15}{'recall@' + str(k):>11}{'MRR@' + str(k):>9}{'nDCG@' + str(k):>10}{'p50 ms':>9}{'p95 ms':>9}"
    lines = [f"\n{dataset}, {len(reports[0].per_query)} queries\n", header, "-" * len(header)]
    for r in reports:
        lines.append(f"{r.retriever:<15}{r.recall:>11.3f}{r.mrr:>9.3f}{r.ndcg:>10.3f}{r.p50_ms:>9.2f}{r.p95_ms:>9.2f}")
    return "\n".join(lines)


def markdown_table(k: int, reports: list[RetrieverReport]) -> str:
    best = {m: max(getattr(r, m) for r in reports) for m in ("recall", "mrr", "ndcg")}
    bold = lambda r, m: f"**{getattr(r, m):.3f}**" if getattr(r, m) == best[m] else f"{getattr(r, m):.3f}"  # noqa: E731
    lines = [
        f"| Retriever | recall@{k} | MRR@{k} | nDCG@{k} | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(f"| {r.retriever} | {bold(r, 'recall')} | {bold(r, 'mrr')} | {bold(r, 'ndcg')} | {r.p50_ms:.1f} | {r.p95_ms:.1f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--retrievers", default=",".join(REGISTRY), help="comma-separated")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    args = parser.parse_args()

    reports = sorted(evaluate(args.dataset, args.retrievers.split(","), args.k), key=lambda r: r.ndcg, reverse=True)
    if args.format == "markdown":
        print(markdown_table(args.k, reports))
    elif args.format == "json":
        print(json.dumps({"dataset": args.dataset, "k": args.k, "retrievers": [asdict(r) for r in reports]}, indent=2))
    else:
        print(text_table(args.dataset, args.k, reports))


if __name__ == "__main__":
    main()
