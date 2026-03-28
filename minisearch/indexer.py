"""
Tokenizer and inverted index builder.
Reads text/markdown files and builds a persistent JSON index.
"""
import json
import math
import os
import re
import string
from pathlib import Path
from typing import Dict, List

# Simple English stop words (no NLTK needed)
STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","it","its","this","that","are","was","were","be","been","have",
    "has","had","do","does","did","will","would","could","should","may",
    "might","not","no","so","if","as","by","from","up","about","into",
    "than","then","they","their","there","we","our","you","your",
}


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split, remove stop words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def stem(word: str) -> str:
    """
    Minimal suffix stripping (Porter-lite).
    Good enough for demos without any dependency.
    """
    suffixes = ["ing", "tion", "ness", "ment", "ly", "ed", "er", "est", "s"]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def read_file(path: Path) -> str:
    """Read a file and strip markdown formatting."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Strip markdown headers, links, code fences
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text


class Indexer:
    def __init__(self, index_path: str = "index.json"):
        self.index_path = index_path
        self.index: Dict = {
            "docs": {},        # doc_id -> {path, title, snippet, token_count}
            "inverted": {},    # token -> {doc_id: term_freq}
            "total_docs": 0,
        }

    def add_document(self, path: Path, doc_id: str = None):
        doc_id = doc_id or str(path)
        raw = read_file(path)
        tokens = [stem(t) for t in tokenize(raw)]

        if not tokens:
            return

        # Term frequency map
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # Extract title (first non-empty line or filename)
        first_line = raw.strip().split("\n")[0].strip()[:80]
        title = first_line if first_line else path.stem

        # Snippet: first 200 chars of clean text
        snippet = re.sub(r"\s+", " ", raw).strip()[:200]

        self.index["docs"][doc_id] = {
            "path": str(path),
            "title": title,
            "snippet": snippet,
            "token_count": len(tokens),
        }

        for token, freq in tf.items():
            if token not in self.index["inverted"]:
                self.index["inverted"][token] = {}
            self.index["inverted"][token][doc_id] = freq

        self.index["total_docs"] += 1
        print(f"  Indexed: {path.name} ({len(tokens)} tokens)")

    def build(self, folder: str, extensions: tuple = (".txt", ".md")):
        folder_path = Path(folder)
        files = [
            f for f in folder_path.rglob("*")
            if f.is_file() and f.suffix in extensions
        ]
        print(f"Indexing {len(files)} files from '{folder}'...")
        for f in files:
            self.add_document(f)

        self.save()
        print(f"\nDone. {self.index['total_docs']} documents indexed → {self.index_path}")

    def save(self):
        with open(self.index_path, "w") as fh:
            json.dump(self.index, fh, indent=2)

    def load(self):
        with open(self.index_path) as fh:
            self.index = json.load(fh)
        return self
