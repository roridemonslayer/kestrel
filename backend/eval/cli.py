"""Print a results table:  python -m eval.cli --dataset scifact --retrievers bm25 --k 10"""

from __future__ import annotations

import argparse

from eval.runner import evaluate
from retrieval.registry import REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--retrievers", default=",".join(REGISTRY), help="comma-separated")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    reports = evaluate(args.dataset, args.retrievers.split(","), args.k)
    k = args.k
    header = f"{'retriever':<12}{'recall@' + str(k):>11}{'MRR@' + str(k):>9}{'nDCG@' + str(k):>10}{'p50 ms':>9}{'p95 ms':>9}"
    print(f"\n{args.dataset}, {len(reports[0].per_query)} queries\n")
    print(header)
    print("-" * len(header))
    for r in sorted(reports, key=lambda r: r.ndcg, reverse=True):
        print(f"{r.retriever:<12}{r.recall:>11.3f}{r.mrr:>9.3f}{r.ndcg:>10.3f}{r.p50_ms:>9.2f}{r.p95_ms:>9.2f}")


if __name__ == "__main__":
    main()
