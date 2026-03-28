# minisearch-engine

> **A from-scratch search engine in pure Python — TF-IDF and BM25 scoring, inverted index, snippet extraction, and term highlighting. Zero dependencies.**

Index any folder of text or markdown files and search them from the CLI in milliseconds. Built to demonstrate core Information Retrieval algorithms without hiding them behind a library.

---

## Features

- **BM25 scoring** — the algorithm behind Elasticsearch and Lucene
- **TF-IDF scoring** — classic algorithm, included for comparison
- **Built-in benchmark** — run both algorithms side-by-side with timing
- **Inverted index** — stored as a portable JSON file, no database needed
- **Snippet extraction** — shows the most relevant passage with matched terms highlighted
- **Minimal stemmer** — suffix stripping without NLTK/spaCy
- **Zero dependencies** — pure Python 3.9+ stdlib

---

## Install & Run

```bash
git clone https://github.com/VikasPatel22/minisearch-engine
cd minisearch-engine
pip install -e .

# Index a folder of docs
minisearch index ./data

# Search it
minisearch search "machine learning gradient descent"

# Compare BM25 vs TF-IDF
minisearch benchmark "neural network training"

# Use TF-IDF explicitly
minisearch search "python async" --algo tfidf

# Get top 5 results
minisearch search "cloudflare workers" --top 5
```

---

## Example Output

```
Top 3 results for 'gradient descent' [BM25]

  #1 Understanding Neural Networks  (score: 4.2817)
       data/ml/neural_nets.md
       ...The core of training is gradient descent — iteratively nudging weights
       in the direction that reduces loss...

  #2 Optimization Algorithms  (score: 3.1204)
       data/ml/optimization.md
       ...gradient descent variants like SGD, Adam, and RMSProp all...
```

---

## How BM25 Works

```
score(q, d) = Σ IDF(t) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × dl/avgdl))

Where:
  tf   = term frequency in document
  IDF  = inverse document frequency (penalizes common words)
  dl   = document length
  avgdl = average document length across corpus
  k1   = term saturation constant (default 1.5)
  b    = length normalization (default 0.75)
```

BM25 outperforms TF-IDF by preventing very long documents from dominating just due to raw term count.

---

## File Structure

```
minisearch-engine/
├── minisearch/
│   ├── indexer.py         # Tokenize, stem, build inverted index
│   ├── scorer.py          # TF-IDF + BM25 implementations
│   ├── query.py           # Query parser, snippet extractor
│   └── cli.py             # CLI commands: index, search, benchmark
├── data/                  # Sample documents to try
├── pyproject.toml
└── README.md
```

---

## License

MIT © Vikas Patel
