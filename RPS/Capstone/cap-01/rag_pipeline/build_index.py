"""Build a Chroma index from JSONL chunk file.

This is a starter implementation. It uses LangChain OpenAI embeddings when
available and stores vectors into a local Chroma collection.
"""
import json
import os
import hashlib
from pathlib import Path
from typing import Iterable


def iter_chunks(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def _get_openai_embeddings():
    """Try multiple import locations for OpenAIEmbeddings to support LangChain versions."""
    try:
        # LangChain v0.0x layout
        print("Call to langchain.embeddings.openai.OpenAIEmbeddings")
        from langchain_openai import OpenAIEmbeddings
        print("Call to langchain.embeddings.openai.OpenAIEmbeddings succeeded")
        return OpenAIEmbeddings(model="text-embedding-3-large")
    except Exception:
        print("Exception in langchain.embeddings.openai.OpenAIEmbeddings import")
        pass
    try:
        # older layout
        print("Call to langchain.embeddings.OpenAIEmbeddings")
        from langchain.embeddings import OpenAIEmbeddings
        print("Call to langchain.embeddings.OpenAIEmbeddings succeeded")
        return OpenAIEmbeddings()
    except Exception:
        print("Exception in langchain.embeddings.OpenAIEmbeddings import" + str(Exception))
        return None
        


def _mock_embed_documents(docs, dim: int = 64):
    vecs = []
    for d in docs:
        h = hashlib.sha256(d.encode("utf-8", errors="ignore")).digest()
        vals = [((b % 128) / 64.0) - 1.0 for b in h]
        if len(vals) < dim:
            vals = (vals * ((dim // len(vals)) + 1))[:dim]
        vecs.append([float(x) for x in vals[:dim]])
    return vecs


def build_index_from_jsonl(jsonl_path: str, persist_dir: str = ".chromadb", collection_name: str = "capstone"):
    try:
        import chromadb
    except Exception as e:
        raise RuntimeError("Missing 'chromadb' dependency. Install requirements.txt") from e

    emb = None
    use_openai = bool(os.environ.get("OPENAI_API_KEY"))
    print("Key OPENAI_API_KEY found: " + os.environ.get("OPENAI_API_KEY"))
    if use_openai:
        print("Call to _get_openai_embeddings()")
        emb = _get_openai_embeddings()
        print("OpenAIEmbeddings instance: " + str(emb))

    # Instantiate chromadb client in a way that supports multiple versions
    client = None
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        print("ChromaDB client created with path kwarg" + persist_dir)
    except TypeError:
        # Some chromadb versions do not accept `path` kwarg
        try:
            client = chromadb.Client()
            print("ChromaDB client created without path kwarg")
            # attempt to set persist directory if supported
            if hasattr(client, "persist_directory"):
                try:
                    client.persist_directory = persist_dir
                    print("ChromaDB client persist_directory set to " + persist_dir)
                except Exception:
                    pass
        except Exception:
            client = None
    except Exception:
        client = None

    if client is None:
        # try chromadb.config.Settings + chromadb.Client(settings=...)
        try:
            from chromadb.config import Settings

            settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
            client = chromadb.Client(settings=settings)
            print("ChromaDB client created with Settings" + settings)
        except Exception:
            # last resort: default client()
            try:
                client = chromadb.Client()
            except Exception as e:
                raise RuntimeError("Unable to construct chromadb client for this environment") from e

    try:
        collection = client.get_or_create_collection(name=collection_name)
        print(f"Created new collection '{collection_name}'")
        print("ChromaDB collection created: " + str(collection))
    except Exception:
        try:
            collection = client.get_collection(name=collection_name)
            print(f"Using existing collection '{collection_name}'")
        except Exception:
            # if get_collection also fails, re-raise
            print(f"Failed to get existing collection '{collection_name}'")
            raise

    ids = []
    metadatas = []
    documents = []
    for chunk in iter_chunks(jsonl_path):
        ids.append(f"{chunk.get('doc_id')}-{chunk.get('chunk_id')}")
        metadatas.append({
            "source": chunk.get("source"),
            "page": chunk.get("page"),
            "chunk_length": chunk.get("chunk_length"),
        })
        documents.append(chunk.get("page_content"))
        print(chunk.get("source"), chunk.get("chunk_id"), "len:", len(chunk.get("page_content", "")))

    if emb is not None:
        try:
            vectors = emb.embed_documents(documents)
            print("Generated embeddings using OpenAIEmbeddings")
        except Exception:
            vectors = _mock_embed_documents(documents)
            print("Falling back to mock embeddings due to error in OpenAIEmbeddings")
    else:
        vectors = _mock_embed_documents(documents)
        print("Using mock embeddings (no OpenAI API key found)")

    # ChromaDB cannot accept None values in metadata; remove keys with None
    cleaned_metadatas = []
    for md in metadatas:
        if not isinstance(md, dict):
            cleaned_metadatas.append({})
            continue
        cleaned = {k: v for k, v in md.items() if v is not None}
        # ensure primitive types (strings/ints/floats) — convert Path-like to str
        for k, v in list(cleaned.items()):
            if isinstance(v, (list, dict)):
                # drop complex structures from metadata
                cleaned.pop(k, None)
            elif not isinstance(v, (str, int, float, bool)):
                try:
                    cleaned[k] = str(v)
                except Exception:
                    cleaned.pop(k, None)
        cleaned_metadatas.append(cleaned)

    collection.add(ids=ids, documents=documents, metadatas=cleaned_metadatas, embeddings=vectors)
    print(f"Added {len(ids)} documents to collection '{collection_name}'")
    print(collection.count())
    try:
        # Below line is not required for PersistentClient, but some versions may need it
        #client.persist()
        print("ChromaDB client persist() called successfully.")         
    except Exception:
        print("ChromaDB client persist() not supported in this version.")
        pass
    return collection


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--persist_dir", default=".chromadb")
    args = parser.parse_args()
    coll = build_index_from_jsonl(args.chunks, persist_dir=args.persist_dir)
    print("Index built.")
