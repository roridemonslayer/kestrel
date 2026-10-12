from eval.cli import markdown_table
from eval.runner import RetrieverReport


def report(name, recall, mrr, ndcg):
    return RetrieverReport(retriever=name, recall=recall, mrr=mrr, ndcg=ndcg, p50_ms=1.0, p95_ms=2.0)


def test_markdown_table_bolds_the_best_score_in_each_column():
    table = markdown_table(10, [report("a", 0.9, 0.5, 0.7), report("b", 0.8, 0.6, 0.7)])
    lines = table.splitlines()
    assert lines[0] == "| Retriever | recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |"
    assert lines[2] == "| a | **0.900** | 0.500 | **0.700** | 1.0 | 2.0 |"
    assert lines[3] == "| b | 0.800 | **0.600** | **0.700** | 1.0 | 2.0 |"
