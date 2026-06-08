"""Task 6 — Lexical Search Module (BM25) bằng Python thuần."""

import json
import math
from collections import Counter

try:
    from src.task4_chunking_indexing import (
        VECTOR_INDEX_PATH,
        chunk_documents,
        embed_chunks,
        index_to_vectorstore,
        load_documents,
        tokenize,
    )
except ImportError:  # Cho phép chạy trực tiếp: python src/task6_lexical_search.py
    from task4_chunking_indexing import (
        VECTOR_INDEX_PATH,
        chunk_documents,
        embed_chunks,
        index_to_vectorstore,
        load_documents,
        tokenize,
    )


CORPUS: list[dict] = []
BM25_INDEX = None


class BM25Index:
    """
    Minimal BM25Okapi implementation.

    BM25 chấm điểm theo TF, IDF và chuẩn hóa độ dài document. k1=1.5 kiểm
    soát bão hòa term frequency; b=0.75 giảm lợi thế của chunk dài.
    """

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        doc_count = len(self.tokenized_corpus)
        doc_freq = Counter()

        for doc in self.tokenized_corpus:
            doc_freq.update(set(doc))

        return {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []

        for index, term_freq in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_len[index]
            for term in query_tokens:
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue

                idf = self.idf.get(term, 0.0)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / (self.avgdl or 1)
                )
                score += idf * (tf * (self.k1 + 1)) / denominator

            scores.append(score)

        return scores


def load_corpus() -> list[dict]:
    """Load chunks from Task 4's local vector index, building it if needed."""
    if not VECTOR_INDEX_PATH.exists():
        docs = load_documents()
        chunks = chunk_documents(docs)
        chunks = embed_chunks(chunks)
        index_to_vectorstore(chunks)

    payload = json.loads(VECTOR_INDEX_PATH.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    return [
        {
            "content": chunk.get("content", ""),
            "metadata": chunk.get("metadata", {}),
        }
        for chunk in chunks
        if chunk.get("content")
    ]


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Index(tokenized_corpus)


def get_bm25_index():
    """Lazily initialize corpus and BM25 index once per process."""
    global CORPUS, BM25_INDEX

    if not CORPUS:
        CORPUS = load_corpus()

    if BM25_INDEX is None:
        BM25_INDEX = build_bm25_index(CORPUS)

    return BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0:
        return []

    bm25 = get_bm25_index()
    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )

    results = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue

        results.append(
            {
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"],
            }
        )
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
