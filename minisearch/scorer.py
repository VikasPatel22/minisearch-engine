"""
Scoring algorithms: TF-IDF and BM25.
BM25 (Best Match 25) is the industry standard used by Elasticsearch, Lucene, etc.
"""
import math
from typing import Dict, List, Tuple


def tf_idf_score(
    query_tokens: List[str],
    index: Dict,
) -> List[Tuple[str, float]]:
    """
    Classic TF-IDF scoring.
    Returns list of (doc_id, score) sorted by descending score.
    """
    docs = index["docs"]
    inverted = index["inverted"]
    N = index["total_docs"]

    scores: Dict[str, float] = {}

    for token in query_tokens:
        if token not in inverted:
            continue

        postings = inverted[token]
        df = len(postings)
        idf = math.log((N + 1) / (df + 1)) + 1  # smoothed IDF

        for doc_id, tf in postings.items():
            token_count = docs[doc_id]["token_count"]
            tf_norm = tf / token_count  # normalized TF
            scores[doc_id] = scores.get(doc_id, 0) + tf_norm * idf

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def bm25_score(
    query_tokens: List[str],
    index: Dict,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[str, float]]:
    """
    BM25 scoring (Okapi BM25).
    k1 controls term frequency saturation (1.2–2.0 typical).
    b controls document length normalization (0 = off, 1 = full).
    Returns list of (doc_id, score) sorted by descending score.
    """
    docs = index["docs"]
    inverted = index["inverted"]
    N = index["total_docs"]

    # Average document length
    if N == 0:
        return []
    avg_dl = sum(d["token_count"] for d in docs.values()) / N

    scores: Dict[str, float] = {}

    for token in query_tokens:
        if token not in inverted:
            continue

        postings = inverted[token]
        df = len(postings)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        for doc_id, tf in postings.items():
            dl = docs[doc_id]["token_count"]
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            scores[doc_id] = scores.get(doc_id, 0) + idf * tf_norm

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
