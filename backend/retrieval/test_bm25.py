from retrieval.bm25 import BM25Retriever, tokenize

DOCS = {
    "cats": "Cats are small furry pets that purr.",
    "dogs": "Dogs are loyal pets that bark.",
    "stars": "Stars are massive balls of burning gas in space.",
}


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Cats ARE purring!") == ["cats", "purring"]


def test_bm25_ranks_the_matching_document_first():
    bm25 = BM25Retriever()
    bm25.index(DOCS)
    results = bm25.search("which pets bark", k=3)
    assert results[0].doc_id == "dogs"
    assert results[0].score > results[1].score
    # "stars" shares no words with the query, so it isn't returned at all.
    assert "stars" not in [r.doc_id for r in results]
