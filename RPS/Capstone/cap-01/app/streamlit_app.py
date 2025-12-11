"""Streamlit UI wired to the local RAG pipeline.

This app will attempt to open a Chroma collection named `capstone` from
the project-root `.chromadb` persist directory and call `rag_pipeline.query.query_collection`.
If `OPENAI_API_KEY` is set, the pipeline will use real embeddings/LLM; otherwise
it uses deterministic mock fallbacks.
"""
from pathlib import Path
import streamlit as st
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Enterprise RAG Assistant", layout="wide")
st.title("Enterprise RAG Assistant (MVP)")

with st.sidebar:
    st.markdown("**RAG Settings**")
    persist_dir = st.text_input("Chroma persist dir (relative to project root)", value=".chromadb")
    top_k = st.slider("Top K", 1, 10, 4)
    model_name = st.text_input("LLM model name (optional)", value="gpt-4o")

question = st.text_input("Ask a question about the ingested documents:")


def make_chroma_client(persist_dir: str):
    import chromadb
    # prefer passing an absolute persist dir if supported
    try:
        return chromadb.PersistentClient(path=persist_dir)
    except TypeError:
        try:
            client = chromadb.Client()
            if hasattr(client, "persist_directory"):
                try:
                    client.persist_directory = persist_dir
                except Exception:
                    pass
            return client
        except Exception:
            try:
                from chromadb.config import Settings
                settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
                return chromadb.Client(settings=settings)
            except Exception as e:
                raise RuntimeError(f"Unable to construct chromadb client: {e}")


def get_collection(client, name="capstone"):
    try:
        return client.get_collection(name=name)
    except Exception:
        try:
            return client.create_collection(name=name)
        except Exception as e:
            raise RuntimeError(f"Unable to get or create collection '{name}': {e}")


if st.button("Query"):
    if not question or not question.strip():
        st.error("Please enter a question.")
    else:
        abs_persist = (ROOT / persist_dir).resolve()
        st.info(f"Using persist dir: {abs_persist}")
        try:
            abs_persist.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            client = make_chroma_client(str(abs_persist))
        except Exception as e:
            st.error(f"Failed to create Chroma client: {e}")
            raise

        try:
            collection = get_collection(client, name="capstone")
        except Exception as e:
            st.error(str(e))
            raise

        with st.spinner("Retrieving and generating answer..."):
            try:
                from rag_pipeline.query import query_collection

                res = query_collection(collection, question, k=top_k)
            except Exception as e:
                st.error(f"Query failed: {e}")
                raise

        st.subheader("Answer")
        st.write(res.get("answer") or res.get("answer") == "")

        st.subheader("Citations")
        citations = res.get("citations") or [r.get("source") for r in res.get("retrieved", [])]
        if citations:
            for c in citations:
                st.markdown(f"- `{c}`")
        else:
            st.write("No citations returned.")

        st.subheader("Retrieved Chunks")
        for i, r in enumerate(res.get("retrieved", []) or []):
            with st.expander(f"Chunk {i} — {r.get('source')}"):
                st.write(r.get("page_content"))

