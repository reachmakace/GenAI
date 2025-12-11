"""Simple query helper that retrieves top-k chunks and constructs a prompt.

This module provides a minimal function to query a Chroma collection. Replace
the LLM call with a real call to OpenAI in production code.
"""
from typing import Dict, Any, List
import os


def build_prompt(question: str, retrieved: List[Dict[str, Any]]) -> str:
    ctx = "\n\n".join([f"Source: {r.get('source')}\n{r.get('page_content')}" for r in retrieved])
    prompt = f"You are given the following context:\n{ctx}\n\nAnswer the question: {question}\nProvide sources."
    return prompt


def generate_answer_with_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 512, model_name: str = "gpt-4o") -> str:
    """Generate answer using LangChain/OpenAI when available, otherwise return placeholder."""
    print("Generating answer with LLM...")
    OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
    print("OPENAI_API_KEY found, inside generate_answer_with_llm ", bool(OPENAI_KEY))
    
    if not OPENAI_KEY:
        return "[LLM disabled - set OPENAI_API_KEY to enable live responses]\n" + "Answer: (placeholder)"

    try:
        # Use LangChain LLM wrapper if available
        print("Trying to import LangChain ChatOpenAI...")
        from langchain_openai import ChatOpenAI
        print("Calling LLM via LangChain ChatOpenAI with model:", model_name)
        llm = ChatOpenAI(model_name=model_name, temperature=temperature, api_key=OPENAI_KEY, max_tokens=max_tokens)
        # ChatOpenAI is callable
        resp = llm.invoke(prompt)
        print("LangChain LLM response received.")
        # Some wrappers return an object with .content
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception:
        # fallback to OpenAI direct API via openai package
        print("Falling back to openai package for LLM call.")
        try:
            import openai
            print("Calling OpenAI ChatCompletion with model:", model_name)
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            print("OpenAI response received.")
            return resp.choices[0].message.content
        except Exception:
            return "[LLM call failed]"


def query_collection(collection, question: str, k: int = 4):
    """Embed the question, retrieve top-k docs and return prompt + generated answer.

    `collection` should be a Chroma collection or an adapter with a `query` method.
    """
    print("Querying collection with question:", question)
    import os

    docs = []
    use_openai = bool(os.environ.get("OPENAI_API_KEY"))
    print("OPENAI_API_KEY found:", use_openai)
    # Try to create a query embedding if OpenAI is enabled
    query_vec = None
    if use_openai:
        try:
            from langchain_openai import OpenAIEmbeddings
            print("Creating query embedding using OpenAIEmbeddings")
            emb = OpenAIEmbeddings(model="text-embedding-3-large")
            if hasattr(emb, "embed_query"):
                print("Using embed_query for question embedding")
                query_vec = emb.embed_query(question)
            else:
                print("Using embed_documents for question embedding")
                query_vec = emb.embed_documents([question])[0]
        except Exception:
            query_vec = None
        print("Query embedding created:", query_vec is not None)    
    # Prefer embedding-based query when available
    try:
        if query_vec is not None:
            res = collection.query(query_embeddings=[query_vec], n_results=k)
        else:
            res = collection.query(query_texts=[question], n_results=k)
        print("ChromaDB query executed.")
        if "documents" in res and len(res["documents"]) > 0:
            for i, d in enumerate(res["documents"][0]):
                src = None
                if res.get("metadatas") and len(res.get("metadatas")) > 0:
                    src_meta = res["metadatas"][0]
                    if i < len(src_meta):
                        src = src_meta[i].get("source")
                docs.append({"source": src, "page_content": d})
    except Exception:
        # If the collection doesn't support .query we try a more general interface
        try:
            # some adapters implement get that returns documents
            all_docs = collection.get()
            # naive top-k selection: take first k
            for item in all_docs.get("documents", [])[:k]:
                docs.append({"source": None, "page_content": item})
        except Exception:
            docs = []

    prompt = build_prompt(question, docs)
    print("Prompt built with retrieved documents.")
    print("Prompt length:", len(prompt))
    print(prompt[:5])  # print first 500 chars of prompt
    answer = generate_answer_with_llm(prompt)
    print("Answer generated by LLM.")
    return {"prompt": prompt, "retrieved": docs, "answer": answer}
