"""Search quality metrics, written to be read and learned from.

Every metric here answers a version of the same question:
"For one query, did the retriever put the relevant documents near the top?"

Each function takes the same two inputs:

    ranked_ids    the retriever's results for one query, best first,
                  e.g. ["d7", "d2", "d9"]  means d7 came 1st, d2 2nd, d9 3rd
    relevant      the documents a human judged relevant for that query,
                  as a dict {doc_id: grade}. A grade of 1 means relevant;
                  some datasets use 2 or 3 for "very relevant".

To score a whole dataset, compute the metric for every query and average them
(see `mean`). That average is the number you see on the leaderboard.

Everything is plain Python on purpose: no numpy, no libraries, just loops.
"""

from __future__ import annotations

import math


def recall_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """What fraction of the relevant documents showed up in the top k results?

    Worked example:
        ranked_ids = ["d1", "d2", "d3", "d4"]
        relevant   = {"d2": 1, "d4": 1, "d9": 1}     # 3 relevant documents
        recall_at_k(ranked_ids, relevant, k=3)

        The top 3 results are d1, d2, d3.
        Of the 3 relevant documents, only d2 is in there  -> found 1
        recall@3 = found / total relevant = 1 / 3 = 0.333

    Recall ignores order inside the top k: finding d2 at rank 1 or rank 3
    counts the same. It only asks "did we find it at all within k?"
    """
    if not relevant:
        return 0.0  # nothing to find, so there is nothing to score

    # Only the first k results count; anything ranked lower is out of view.
    top_k = ranked_ids[:k]

    # Count how many of those top results are relevant.
    found = 0
    for doc_id in top_k:
        if doc_id in relevant:
            found += 1

    # Divide by how many relevant documents exist in total, found or not.
    return found / len(relevant)


def reciprocal_rank(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """1 / (position of the FIRST relevant result), or 0 if none is in the top k.

    MRR ("mean reciprocal rank") is this number averaged over all queries.

    Worked example:
        ranked_ids = ["d5", "d8", "d2", "d1"]
        relevant   = {"d2": 1, "d1": 1}
        reciprocal_rank(ranked_ids, relevant, k=10)

        d5 at rank 1: not relevant
        d8 at rank 2: not relevant
        d2 at rank 3: relevant!  -> stop here
        reciprocal rank = 1 / 3 = 0.333

    So: first result relevant -> 1.0, second -> 0.5, third -> 0.333, ...
    It rewards putting a good answer right at the top, and it only cares
    about the first hit (d1 at rank 4 changes nothing).
    """
    # enumerate(..., start=1) gives positions 1, 2, 3... the way people count ranks.
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0  # no relevant document anywhere in the top k


def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain: a 0-to-1 score for the whole top k.

    It combines two ideas:
      * relevant documents earn points (more points for higher grades)
      * points shrink the lower a document is ranked ("discounted")
    Then it divides by the best score that was possible ("normalized"),
    so 1.0 means "perfect ordering" no matter how many relevant docs exist.

    The discount for rank r is 1 / log2(r + 1):
        rank 1 -> 1 / log2(2) = 1.0
        rank 2 -> 1 / log2(3) = 0.631
        rank 3 -> 1 / log2(4) = 0.5
        rank 4 -> 1 / log2(5) = 0.431

    Worked example:
        ranked_ids = ["d1", "d2", "d3"]
        relevant   = {"d2": 1, "d3": 1}
        ndcg_at_k(ranked_ids, relevant, k=3)

        DCG (what the retriever got):
            d1 at rank 1: grade 0 -> 0 * 1.0   = 0
            d2 at rank 2: grade 1 -> 1 * 0.631 = 0.631
            d3 at rank 3: grade 1 -> 1 * 0.5   = 0.5
            DCG = 1.131

        IDCG (the best possible: both relevant docs at ranks 1 and 2):
            1 * 1.0 + 1 * 0.631 = 1.631

        nDCG@3 = DCG / IDCG = 1.131 / 1.631 = 0.693
    """
    if not relevant:
        return 0.0

    # Step 1: DCG, the discounted score of the actual ranking.
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        grade = relevant.get(doc_id, 0)  # documents that aren't relevant are worth 0
        dcg += grade / math.log2(rank + 1)

    # Step 2: IDCG, the discounted score of a perfect ranking:
    # every relevant document, highest grade first, packed into the top k.
    best_grades = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, grade in enumerate(best_grades, start=1):
        idcg += grade / math.log2(rank + 1)

    # Step 3: normalize, so the answer is always between 0 and 1.
    return dcg / idcg


def mean(values: list[float]) -> float:
    """Average of per-query scores; this is what turns one score per query into one score per retriever."""
    if not values:
        return 0.0
    return sum(values) / len(values)
