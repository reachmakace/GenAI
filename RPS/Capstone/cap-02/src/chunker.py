"""Simple text chunking utilities."""
from typing import List, Dict
from uuid import uuid4


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Split `text` into chunks of approx `chunk_size` with `overlap`.

    Returns a list of dicts with keys: `id`, `text`, `start`, `end`.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >=0 and < chunk_size")

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end]
        chunks.append({
            "id": str(uuid4()),
            "text": chunk_text,
            "start": start,
            "end": end,
        })
        if end == text_len:
            break
        start = end - overlap

    return chunks
