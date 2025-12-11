"""Command-line script to run ingestion and write chunks as JSONL."""
import argparse
import json
from pathlib import Path
from ingestion.loader import find_files, load_file, chunk_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    files = find_files(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.jsonl"

    with out_path.open("w", encoding="utf-8") as fw:
        for f in files:
            doc = load_file(str(f))
            chunks = chunk_text(doc["text"], chunk_size=args.chunk_size, overlap=args.overlap)
            for i, c in enumerate(chunks):
                item = {
                    "doc_id": f.stem,
                    "chunk_id": i,
                    "chunk_length": len(c),
                    "source": str(f),
                    "page": None,
                    "page_content": c,
                }
                fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote chunks to {out_path}")


if __name__ == "__main__":
    main()
