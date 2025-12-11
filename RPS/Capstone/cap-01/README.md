# Enterprise RAG Assistant (Capstone)

Quick scaffold for the Enterprise RAG Assistant project (OpenAI + ChromaDB + LangChain).

Quickstart

1. Create a virtual environment and activate (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Set `OPENAI_API_KEY` in your environment (PowerShell):
```powershell
$env:OPENAI_API_KEY = 'sk-...'
```

3. Ingest documents (example):
```powershell
python -m ingestion.run_ingest --input_dir sample_data --output_dir data/chunks
```

4. Build index (example):
```powershell
python -m rag_pipeline.build_index --chunks data/chunks/chunks.jsonl --persist_dir .chromadb
```

5. Run Streamlit UI:
```powershell
streamlit run app/streamlit_app.py
```

Project layout
- `ingestion/` – loaders, preprocessing, chunking
- `rag_pipeline/` – embedding, index, query
- `models/` – Pydantic schemas
- `app/` – Streamlit or FastAPI app
- `notebooks/` – experiments and evaluation
- `tests/` – unit tests

Next actions
- Wire in OpenAI embeddings and LLM calls (requires API key)
- Add sample documents in `sample_data/` for smoke tests
- Run `pytest` and iterate on pipeline
