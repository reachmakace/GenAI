# Demo: End-to-end workflow (placeholder)

This file describes steps for a demo notebook.

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables: `OPENAI_API_KEY`, `REDIS_URL`
3. Run `python -m src.ingest --input-dir Input/` to confirm ingestion works.
4. Implement `src.embeddings.embed_texts`, then run a small script to index one document into Redis.
5. Create a small notebook showing retrieval and LLM-driven answer using retrieved context.
