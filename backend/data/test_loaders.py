from data.loaders import load_dataset


def test_scifact_loads_with_known_sizes():
    ds = load_dataset("scifact")
    assert len(ds.corpus) == 5183
    assert len(ds.queries) == 300
    # Every query has at least one relevant document, and every judged document exists in the corpus.
    for query_id, judged in ds.qrels.items():
        assert query_id in ds.queries
        assert judged
        assert all(doc_id in ds.corpus for doc_id in judged)


def test_nfcorpus_loads_with_graded_relevance():
    ds = load_dataset("nfcorpus")
    assert len(ds.corpus) == 3633
    assert len(ds.queries) == 323
    # NFCorpus grades relevance 1 or 2, which nDCG uses to reward the more relevant documents.
    grades = {grade for judged in ds.qrels.values() for grade in judged.values()}
    assert grades == {1, 2}


def test_arguana_queries_are_corpus_documents():
    ds = load_dataset("arguana")
    assert len(ds.corpus) == 8674
    assert len(ds.queries) == 1406
    # Most queries are arguments taken from the corpus, which is why their own document is dropped from results.
    assert sum(query_id in ds.corpus for query_id in ds.queries) == 1298
    assert not any(query_id in judged for query_id, judged in ds.qrels.items())
