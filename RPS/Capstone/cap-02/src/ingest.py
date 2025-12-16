"""Ingestion script that extracts text and writes overlapping chunks.

Produces a JSONL file of chunks ready for embedding.
"""
import argparse
import json
from pathlib import Path
from typing import List

from pdfminer.high_level import extract_text

from .chunker import chunk_text


def extract_text_from_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text(str(path))
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


def ingest_and_chunk(
    input_dir: str, output_path: str, chunk_size: int = 1000, overlap: int = 200
) -> List[dict]:
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    files_processed = 0

    with out_p.open("w", encoding="utf-8") as fh:
        for file in p.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix.lower() not in {".txt", ".pdf"}:
                continue

            files_processed += 1
            text = extract_text_from_file(file)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for idx, c in enumerate(chunks):
                record = {
                    "id": c["id"],
                    "source": str(file),
                    "chunk_index": idx,
                    "text": c["text"],
                    "start": c["start"],
                    "end": c["end"],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    return [{"files_processed": files_processed, "chunks_written": written, "output": str(out_p)}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with input documents")
    parser.add_argument("--output", default="data/chunks.jsonl", help="Output JSONL file for chunks")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Approximate chunk size in chars")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks in chars")
    args = parser.parse_args()

    result = ingest_and_chunk(args.input_dir, args.output, args.chunk_size, args.overlap)
    for r in result:
        print(f"Files processed: {r['files_processed']}")
        print(f"Chunks written: {r['chunks_written']}")
        print(f"Output: {r['output']}")


if __name__ == "__main__":
    main()
