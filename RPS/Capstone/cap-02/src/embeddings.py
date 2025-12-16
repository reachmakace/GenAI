"""Embeddings helpers.

This module chooses between OpenAI embeddings (when `OPENAI_API_KEY` is set)
or a local `sentence-transformers` fallback. The exported function is
`embed_texts(texts)` which returns a list of vectors (list of floats).
"""
import os
import json
from typing import List, Optional

OPENAI_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _use_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def embed_texts(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Embed a list of texts.

    - If `OPENAI_API_KEY` is set, calls OpenAI Embeddings API using `openai`.
    - Otherwise falls back to `sentence-transformers` (`all-MiniLM-L6-v2`).

    Returns: list of vectors (list of floats).
    """
    model = model or OPENAI_MODEL

    if _use_openai():
        try:
            # new OpenAI python client (>=1.0.0)
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("OpenAI package not available: " + str(e))

        # create client (it reads OPENAI_API_KEY from env by default)
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print(f"Using OpenAI embeddings model: {model}")
        # New API uses client.embeddings.create(...)
        # It accepts a list for `input` to batch multiple texts.
        resp = client.embeddings.create(model=model, input=texts)
        vectors = [d.embedding for d in resp.data]
        # ensure plain python lists
        print(f"Obtained {len(vectors)} embeddings from OpenAI")
        return [list(map(float, v)) for v in vectors]

    # Fallback: sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("sentence-transformers not available: " + str(e))

    model_name = os.environ.get("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    st = SentenceTransformer(model_name)
    arr = st.encode(texts, show_progress_bar=False)
    # convert numpy arrays to plain python lists
    return [list(map(float, v)) for v in arr]

