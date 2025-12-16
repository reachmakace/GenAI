# Multi-Modal RAG with Redis

Minimal scaffold for a multimodal Retrieval-Augmented Generation (RAG) system using Redis as vector store.

Quick start
1. Create a virtual environment and activate it.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Create a `.env` with your API keys (e.g. `OPENAI_API_KEY`, `REDIS_URL`).

3. Run the sample ingest script (fill in config first):

```powershell
python -m src.ingest --input-dir Input/
```

What this scaffold contains
- `requirements.txt`: top-level dependencies (unconstrained to fetch latest pip versions).
- `src/`: small helper modules for ingestion, embeddings, and Redis client.
- `notebooks/demo.md`: placeholder describing an interactive demo workflow.

Next steps
- Add your API keys and test a single-document ingest.
- Implement embeddings and Redis indexing (see `src/embeddings.py`).
