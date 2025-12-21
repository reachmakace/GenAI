# This file will contain the main entry point for running the RAG pipeline in Phase 1
# It will orchestrate PDF loading, text extraction, embedding, and storage in Redis

import os
from langchain_community.vectorstores.redis import Redis as RedisVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA
from pdf_loader import extract_slides_from_pdf
from config import OPENAI_API_KEY, REDIS_URL

def embed_and_store_slides(pdf_path: str):
    # Extract slides
    slides = extract_slides_from_pdf(pdf_path)
    # Prepare texts and metadatas
    texts = [slide['text'] for slide in slides]
    metadatas = [{k: v for k, v in slide.items() if k != 'text'} for slide in slides]
    # Set up embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Store in Redis VectorStore
    vectorstore = RedisVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        redis_url=REDIS_URL,
        index_name="slides-index"
    )
    print(f"Stored {len(texts)} slides in Redis vectorstore.")

def retrieve_and_answer(query: str, k: int = 3):
    # Set up embeddings and vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = RedisVectorStore(
        embedding=embeddings,
        redis_url=REDIS_URL,
        index_name="slides-index"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # Set up LLM
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
    # Set up RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa({"query": query})
    print("\nQuestion:", query)
    print("Answer:", result["result"])
    print("\nTop source slides:")
    for doc in result["source_documents"]:
        meta = doc.metadata
        print(f"Metadata: {meta}")
        print(f"- Slide {meta.get('slide_number', '?')} from {meta.get('pdf_file', '?')}")
        print(f"  Content: {doc.page_content[:200]}...\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        embed_and_store_slides("nvidia.pdf")
    elif len(sys.argv) > 2 and sys.argv[1] == "query":
        question = " ".join(sys.argv[2:])
        retrieve_and_answer(question)
    else:
        print("Usage:")
        print("  python main.py ingest         # Ingest and store slides")
        print("  python main.py query <question>  # Query the RAG system")
