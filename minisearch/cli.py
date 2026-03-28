#!/usr/bin/env python3
"""
minisearch CLI — index folders and search them from the terminal.

Usage:
  minisearch index <folder>               # Build index from folder
  minisearch search <query>               # Search the index
  minisearch search <query> --algo tfidf  # Use TF-IDF instead of BM25
  minisearch benchmark <query>            # Compare BM25 vs TF-IDF
"""
import argparse
import json
import sys
import time

from minisearch.indexer import Indexer
from minisearch.query import search

INDEX_FILE = "index.json"


def cmd_index(args):
    indexer = Indexer(INDEX_FILE)
    indexer.build(args.folder)


def cmd_search(args):
    indexer = Indexer(INDEX_FILE)
    try:
        indexer.load()
    except FileNotFoundError:
        print("No index found. Run: minisearch index <folder>")
        sys.exit(1)

    results = search(args.query, indexer.index, algorithm=args.algo, top_k=args.top)

    if not results:
        print("No results found.")
        return

    print(f"\n\033[1mTop {len(results)} results for '{args.query}' [{args.algo.upper()}]\033[0m\n")
    for r in results:
        print(f"  \033[93m#{r['rank']}\033[0m \033[1m{r['title']}\033[0m  (score: {r['score']})")
        print(f"       {r['path']}")
        print(f"       {r['snippet']}\n")


def cmd_benchmark(args):
    indexer = Indexer(INDEX_FILE)
    try:
        indexer.load()
    except FileNotFoundError:
        print("No index found. Run: minisearch index <folder>")
        sys.exit(1)

    print(f"\n\033[1mBenchmark: '{args.query}'\033[0m\n")

    for algo in ("bm25", "tfidf"):
        t0 = time.perf_counter()
        results = search(args.query, indexer.index, algorithm=algo, top_k=5)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\033[96m{algo.upper()}\033[0m ({elapsed:.2f}ms)")
        for r in results:
            print(f"  #{r['rank']}  {r['title']}  — score: {r['score']}")
        print()


def main():
    parser = argparse.ArgumentParser(prog="minisearch", description="Tiny but powerful search engine.")
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="Index a folder of text/markdown files")
    p_index.add_argument("folder", help="Folder path to index")

    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query", help="Query string")
    p_search.add_argument("--algo", choices=["bm25", "tfidf"], default="bm25")
    p_search.add_argument("--top", type=int, default=10)

    p_bench = sub.add_parser("benchmark", help="Compare BM25 vs TF-IDF")
    p_bench.add_argument("query", help="Query string")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
