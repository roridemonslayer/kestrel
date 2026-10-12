"""Load a BEIR dataset as three plain dicts: corpus, queries and qrels.

BEIR publishes each dataset as a zip with:
  corpus.jsonl      one document per line: {"_id", "title", "text"}
  queries.jsonl     one query per line:    {"_id", "text"}
  qrels/test.tsv    relevance judgments:   query-id, corpus-id, score

The zip is downloaded once into .cache/datasets/ and read from disk after that.
"""

from __future__ import annotations

import csv
import io
import json
import ssl
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import certifi

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "datasets"

# Datasets the app offers. Any BEIR dataset with a test split can be added here by name.
DATASETS = {
    "scifact": "SciFact: scientific claims checked against research abstracts",
    "nfcorpus": "NFCorpus: nutrition questions answered from medical papers",
}


@dataclass
class Dataset:
    name: str
    corpus: dict[str, dict[str, str]]  # doc_id -> {"title": ..., "text": ...}
    queries: dict[str, str]  # query_id -> query text (only queries that have judgments)
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance}, relevance > 0 means relevant


def documents_for_indexing(ds: Dataset) -> dict[str, str]:
    """doc_id -> the text a retriever should index: the title followed by the body."""
    return {doc_id: f"{doc['title']}. {doc['text']}".strip(". ") for doc_id, doc in ds.corpus.items()}


def _download(name: str) -> Path:
    folder = CACHE_DIR / name
    if (folder / "corpus.jsonl").exists():
        return folder
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {name} from BEIR...")
    # certifi's CA bundle, since python.org builds of Python on macOS ship without one.
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(BEIR_URL.format(name=name), context=context) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    # The zip holds a single top-level folder named after the dataset.
    archive.extractall(CACHE_DIR)
    return folder


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_dataset(name: str, split: str = "test") -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Available: {', '.join(DATASETS)}")
    folder = _download(name)

    corpus = {
        row["_id"]: {"title": row.get("title", ""), "text": row.get("text", "")}
        for row in _read_jsonl(folder / "corpus.jsonl")
    }

    qrels: dict[str, dict[str, int]] = {}
    with (folder / "qrels" / f"{split}.tsv").open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header: query-id, corpus-id, score
        for query_id, doc_id, score in reader:
            if int(score) > 0:
                qrels.setdefault(query_id, {})[doc_id] = int(score)

    # queries.jsonl holds every split's queries; keep only the ones judged in this split.
    queries = {row["_id"]: row["text"] for row in _read_jsonl(folder / "queries.jsonl") if row["_id"] in qrels}
    return Dataset(name=name, corpus=corpus, queries=queries, qrels=qrels)


if __name__ == "__main__":
    import sys

    ds = load_dataset(sys.argv[1] if len(sys.argv) > 1 else "scifact")
    relevant = sum(len(docs) for docs in ds.qrels.values())
    print(f"{ds.name}: {len(ds.corpus):,} documents, {len(ds.queries):,} queries, {relevant:,} relevant pairs")
    qid = next(iter(ds.queries))
    doc_id = next(iter(ds.qrels[qid]))
    print(f"\nExample query {qid}: {ds.queries[qid]}")
    print(f"A relevant doc ({doc_id}): {ds.corpus[doc_id]['title']}")
