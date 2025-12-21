import os
import json
from langchain_community.vectorstores.redis import Redis as RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY, REDIS_URL


def get_image_path(slide_number, pdf_file="nvidia.pdf"):
    base = os.path.splitext(os.path.basename(pdf_file))[0]
    img_dir = os.path.join(os.path.dirname(__file__), "slide_images")
    return os.path.join(img_dir, f"{base}_slide_{slide_number}.png")


def store_summaries(json_path, pdf_file="nvidia.pdf"):
    with open(json_path, "r", encoding="utf-8") as f:
        slides = json.load(f)

    texts = [slide["summary"] for slide in slides]

    metadatas = [
        {
            "slide_number": str(slide["slide_number"]),
            "pdf_file": os.path.basename(pdf_file),
            "image_path": get_image_path(slide["slide_number"], pdf_file),
        }
        for slide in slides
    ]

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vectorstore = RedisVectorStore.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        redis_url=REDIS_URL,
        index_name="slides-multimodal-index"
    )

    print(f"Stored {len(texts)} multi-modal summaries in Redis vectorstore.")


if __name__ == "__main__":
    store_summaries("slide_summaries.json", "nvidia.pdf")
