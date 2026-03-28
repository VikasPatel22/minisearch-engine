"""
Query engine: parses user query, runs scoring, returns ranked results with snippets.
"""
import re
from typing import List, Dict
from .indexer import tokenize, stem
from .scorer import bm25_score, tf_idf_score


def highlight(text: str, terms: List[str], width: int = 200) -> str:
    """Extract a relevant snippet and highlight matched terms."""
    text_lower = text.lower()
    best_pos = 0
    for term in terms:
        pos = text_lower.find(term)
        if pos != -1:
            best_pos = max(0, pos - 40)
            break

    snippet = text[best_pos: best_pos + width]
    if best_pos > 0:
        snippet = "..." + snippet
    if best_pos + width < len(text):
        snippet += "..."

    # Bold matched terms (ANSI for terminal, ** for plain)
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        snippet = pattern.sub(lambda m: f"\033[1m{m.group()}\033[0m", snippet)

    return snippet


def search(
    query: str,
    index: Dict,
    algorithm: str = "bm25",
    top_k: int = 10,
) -> List[Dict]:
    """
    Search the index for the given query.

    Args:
        query: user query string
        index: loaded index dict
        algorithm: "bm25" or "tfidf"
        top_k: max number of results

    Returns:
        List of result dicts with keys: rank, doc_id, title, path, score, snippet
    """
    raw_tokens = tokenize(query)
    tokens = [stem(t) for t in raw_tokens]

    if not tokens:
        return []

    if algorithm == "tfidf":
        ranked = tf_idf_score(tokens, index)
    else:
        ranked = bm25_score(tokens, index)

    results = []
    for rank, (doc_id, score) in enumerate(ranked[:top_k], start=1):
        doc = index["docs"].get(doc_id, {})
        snippet = highlight(doc.get("snippet", ""), raw_tokens)
        results.append({
            "rank": rank,
            "doc_id": doc_id,
            "title": doc.get("title", doc_id),
            "path": doc.get("path", ""),
            "score": round(score, 4),
            "snippet": snippet,
        })

    return results
