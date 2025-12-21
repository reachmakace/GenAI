import os
import base64
import json
from openai import OpenAI
from pdf_loader import extract_slides_from_pdf
from langchain_community.vectorstores.redis import Redis as RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from config import OPENAI_API_KEY as CONFIG_OPENAI_API_KEY, REDIS_URL

# Set your OpenAI API key (or use dotenv/config.py)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or CONFIG_OPENAI_API_KEY

# Vision model name (e.g., 'gpt-4-vision-preview')
#VISION_MODEL = "gpt-4-vision-preview"
VISION_MODEL = "gpt-4o"

# Prompt template for multi-modal summary
def build_prompt(text):
    return f"""
You are an expert summarizer. Given the slide's text and image, write a concise summary (2-3 sentences) that captures the key points, including any visual information (charts, tables, etc.).

Slide text:
{text}

Slide image: [see attached image]
"""

def generate_multimodal_summaries(pdf_path, summary_cache_file="slide_summaries.json"):
    slides = extract_slides_from_pdf(pdf_path)
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Load or create summary cache
    if os.path.exists(summary_cache_file):
        with open(summary_cache_file, "r", encoding="utf-8") as f:
            summary_cache = json.load(f)
    else:
        summary_cache = {}

    for slide in slides:
        slide_id = f"{slide['pdf_file']}_slide_{slide['slide_number']}"
        if slide_id in summary_cache:
            summary = summary_cache[slide_id]
            print(f"[CACHE] Slide {slide['slide_number']} summary:\n{summary}\n")
        else:
            prompt = build_prompt(slide['text'])
            image_path = slide['image_path']
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]}
                ],
                max_tokens=256
            )
            summary = response.choices[0].message.content
            summary_cache[slide_id] = summary
            print(f"[LLM] Slide {slide['slide_number']} summary:\n{summary}\n")
    # Save updated cache
    with open(summary_cache_file, "w", encoding="utf-8") as f:
        json.dump(summary_cache, f, ensure_ascii=False, indent=2)

    # Store summaries and image refs in Redis
    texts = [summary_cache[f"{slide['pdf_file']}_slide_{slide['slide_number']}"] for slide in slides]
    metadatas = [
        {
            'slide_number': slide['slide_number'],
            'pdf_file': slide['pdf_file'],
            'image_path': slide['image_path']
        } for slide in slides
    ]
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = RedisVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        redis_url=REDIS_URL,
        index_name="slides-multimodal-index"
    )
    print(f"Stored {len(texts)} multi-modal summaries in Redis vectorstore.")

if __name__ == "__main__":
    generate_multimodal_summaries("nvidia.pdf")
