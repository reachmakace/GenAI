from pathlib import Path
import sys
import os

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def make_chroma_client(persist_dir: str = ".chromadb"):
    import chromadb
    # Try a few client constructors to support multiple chromadb versions
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
                raise RuntimeError("Unable to construct chromadb client: %s" % e)


def main():
    persist = str(ROOT / ".chromadb")
    print("Using persist path:", persist)
    client = make_chroma_client(persist)

    # show collections
    try:
        cols = client.list_collections()
        print("Collections:", cols)
    except Exception:
        try:
            # older chroma API
            cols = client.list_collections()
            print("Collections:", cols)
        except Exception:
            pass

    try:
        coll = client.get_collection(name="capstone")
    except Exception:
        try:
            coll = client.create_collection(name="capstone")
        except Exception as e:
            print("Failed to get or create collection:", e)
            return

    from rag_pipeline.query import query_collection

    resp = query_collection(coll, "What is the total revenue of nvidia for the FY24.", k=3)
    print("\nPROMPT:\n", resp.get("prompt")[:10])
    print("\nANSWER:\n", resp.get("answer"))
    print("\nCITATIONS:\n", resp.get("citations") or [r.get("source") for r in resp.get("retrieved", [])])


if __name__ == "__main__":
    main()
