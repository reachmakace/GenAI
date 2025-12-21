import os
import base64
from openai import OpenAI
from langchain_community.vectorstores.redis import Redis as RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from config import (
    OPENAI_API_KEY, REDIS_URL,
    DEFAULT_RETRIEVAL_K, DEFAULT_CACHE_SIMILARITY_THRESHOLD,
    DEFAULT_SUMMARY_PROMPT, DEFAULT_ANSWER_PROMPT
)

VISION_MODEL = "gpt-4o"

# Helper to load image and encode as base64
def load_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Build multi-modal prompt for the LLM
def build_multimodal_prompt(question, slides, custom_template=None):
    if custom_template:
        prompt = custom_template.format(question=question)
    else:
        prompt = DEFAULT_ANSWER_PROMPT.format(question=question)
    for i, slide in enumerate(slides, 1):
        prompt += f"Slide {slide['slide_number']} summary:\n{slide['summary']}\n\n"
    prompt += "Use the images to support your answer."
    return prompt

# Retrieve top-k slides using MultiVectorRetriever
def retrieve_top_slides(query, k=3):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = RedisVectorStore(
        embedding=embeddings,
        redis_url=REDIS_URL,
        index_name="slides-multimodal-index"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    slides = []
    for i, doc in enumerate(docs):
        print(f"[DEBUG] Retrieved doc {i} metadata: {doc.metadata}")
        meta = doc.metadata
        slides.append({
            "slide_number": meta.get("slide_number"),
            "summary": doc.page_content,
            "image_path": meta.get("image_path")
        })
    return slides


# Main function to answer a user query

def answer_query_with_vision_llm(question, k=DEFAULT_RETRIEVAL_K, cache_threshold=DEFAULT_CACHE_SIMILARITY_THRESHOLD, custom_prompt=None):
    """
    Returns: dict with keys: answer (str), slides (list), cache_hit (bool)
    """
    cache_index = "semantic-cache-index"
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    cache_store = RedisVectorStore(
        embedding=embeddings,
        redis_url=REDIS_URL,
        index_name=cache_index
    )
    cache_retriever = cache_store.as_retriever(search_kwargs={"k": 1})
    cache_hit = False
    answer = ""
    slides = []
    # --- Semantic Cache: Check for similar question ---
    try:
        cache_docs = cache_retriever.get_relevant_documents(question)
        if cache_docs:
            cached_q = cache_docs[0].metadata.get("question_text")
            cached_a = cache_docs[0].page_content
            # Only treat as cache hit if the question matches exactly (case-insensitive, stripped)
            if cached_q and cached_q.strip().lower() == question.strip().lower():
                answer = cached_a
                slides_used = cache_docs[0].metadata.get("slides_used", [])
                # Optionally, retrieve slide details if needed
                cache_hit = True
                return {"answer": answer, "slides": [], "cache_hit": True}
    except Exception as e:
        pass  # For UI, just proceed to normal retrieval

    # --- Usual Retrieval + LLM ---
    slides = retrieve_top_slides(question, k)
    prompt = build_multimodal_prompt(question, slides, custom_template=custom_prompt)
    client = OpenAI(api_key=OPENAI_API_KEY)
    images_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{load_image_base64(slide['image_path'])}"}}
        for slide in slides if slide["image_path"] and os.path.exists(slide["image_path"])
    ]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}] + images_content}
    ]
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=messages,
        max_tokens=512
    )
    answer = response.choices[0].message.content

    # --- Store Q&A in semantic cache ---
    cache_metadata = {
        "question_text": str(question) if question else "",
        "slides_used": [str(slide["slide_number"]) for slide in slides],
    }
    cache_store.add_texts(
        texts=[answer],
        metadatas=[cache_metadata]
    )
    return {"answer": answer, "slides": slides, "cache_hit": False}

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Vision RAG Query")
    parser.add_argument("question", type=str, nargs="+", help="Your question for the RAG system")
    parser.add_argument("--k", type=int, default=DEFAULT_RETRIEVAL_K, help="Number of top slides to retrieve")
    parser.add_argument("--cache-threshold", type=float, default=DEFAULT_CACHE_SIMILARITY_THRESHOLD, help="Cache similarity threshold (0-1)")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt template (use {question} as placeholder)")
    args = parser.parse_args()
    question = " ".join(args.question)
    if args.prompt:
        os.environ["VISION_RAG_PROMPT_TEMPLATE"] = args.prompt
    answer_query_with_vision_llm(question, k=args.k, cache_threshold=args.cache_threshold)