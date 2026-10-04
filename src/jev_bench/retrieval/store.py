from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Retriever:
    """Ranks a query's fixed document pool by TF-IDF cosine similarity.

    Deliberately not a vector database: each HotpotQA question ships with its own small
    (10-paragraph) pool already, so there is nothing to index at scale — ranking ten items needs
    no ANN infrastructure. This is pure, deterministic, and dependency-free, which matters more
    here than retrieval sophistication: both variants must read the identical ranking, and it
    must not require a network call or a model download to stay reproducible offline.
    """

    def rank(self, query: str, doc_pool: dict[str, str]) -> list[str]:
        doc_ids = list(doc_pool)
        doc_tokens = {doc_id: _tokenize(doc_pool[doc_id]) for doc_id in doc_ids}
        query_tokens = _tokenize(query)

        document_frequency: Counter[str] = Counter()
        for tokens in doc_tokens.values():
            document_frequency.update(set(tokens))
        n_docs = len(doc_ids)

        def idf(term: str) -> float:
            return math.log((n_docs + 1) / (document_frequency.get(term, 0) + 1)) + 1.0

        def vectorize(tokens: list[str]) -> dict[str, float]:
            counts = Counter(tokens)
            return {term: count * idf(term) for term, count in counts.items()}

        query_vector = vectorize(query_tokens)
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0

        def score(doc_id: str) -> float:
            doc_vector = vectorize(doc_tokens[doc_id])
            doc_norm = math.sqrt(sum(weight * weight for weight in doc_vector.values())) or 1.0
            dot = sum(weight * query_vector.get(term, 0.0) for term, weight in doc_vector.items())
            return dot / (doc_norm * query_norm)

        return sorted(doc_ids, key=lambda doc_id: (-score(doc_id), doc_id))
