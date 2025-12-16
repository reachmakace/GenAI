"""Indexer: read chunks.jsonl, embed with OpenAI (or fallback), and store into Redis.

Usage:
  python -m src.indexer --input data/chunks.jsonl --batch-size 64

Environment:
  - `OPENAI_API_KEY` should be set to use OpenAI embeddings. If not set,
    `sentence-transformers` will be used if available.
  - `REDIS_URL` may be set (defaults to redis://localhost:6379)
"""
import argparse
import json
import os
from typing import List

from .embeddings import embed_texts
from .redis_client import get_redis_client


def read_chunks(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def store_chunk(redis_client, chunk: dict, vector: List[float]):
    key = f"chunk:{chunk['id']}"
    # store metadata and text; embedding as JSON string
    payload = {
        "id": chunk.get("id"),
        "source": chunk.get("source"),
        "chunk_index": str(chunk.get("chunk_index")),
        "start": str(chunk.get("start", "")),
        "end": str(chunk.get("end", "")),
        "text": chunk.get("text")[:4000],  # truncate stored text for safety
        "embedding": json.dumps(vector),
    }
    # write hash
    redis_client.hset(key, mapping=payload)
    print(f"Stored chunk {chunk['id']} (source: {chunk.get('source')})")
    # add to a set of chunk ids for quick listing
    redis_client.sadd("chunks:all", chunk.get("id"))
    print(f"Added chunk ID {chunk['id']} to set chunks:all")    


def index_file(input_path: str, batch_size: int = 64):
    items = read_chunks(input_path)
    print(f"Read {len(items)} chunks from {input_path}")

    r = get_redis_client()

    # Process in batches, batch_size is 64 as per input argument
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        texts = [b.get("text", "") for b in batch]
        print(f"Embedding batch {i}..{i+len(batch)-1} (size {len(batch)})")
        vectors = embed_texts(texts)
        for chunk, vec in zip(batch, vectors):
            store_chunk(r, chunk, vec)

    print("Indexing complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    index_file(args.input, args.batch_size)


if __name__ == "__main__":
    main()
