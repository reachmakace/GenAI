import streamlit as st
import os
from vision_rag_query import answer_query_with_vision_llm
from config import DEFAULT_RETRIEVAL_K, DEFAULT_CACHE_SIMILARITY_THRESHOLD, DEFAULT_ANSWER_PROMPT

st.set_page_config(page_title="Multi-Modal RAG Q&A", layout="wide")
st.title("📊 Multi-Modal Slide Q&A with Vision RAG")

with st.sidebar:
    st.header("Parameters")
    k = st.number_input("Top-k Slides", min_value=1, max_value=10, value=DEFAULT_RETRIEVAL_K)
    cache_threshold = st.slider("Cache Similarity Threshold", 0.0, 1.0, value=float(DEFAULT_CACHE_SIMILARITY_THRESHOLD), step=0.01)
    custom_prompt = st.text_area("Custom Prompt (optional)", value=DEFAULT_ANSWER_PROMPT, height=80)

question = st.text_input("Enter your question:")

if st.button("Get Answer") and question:
    with st.spinner("Retrieving answer..."):
        # Call backend directly
        # We'll need to refactor answer_query_with_vision_llm to return answer, slides, cache_hit
        result = answer_query_with_vision_llm(
            question=question,
            k=k,
            cache_threshold=cache_threshold,
            custom_prompt=custom_prompt
        )
        answer = result.get("answer", "")
        slides = result.get("slides", [])
        cache_hit = result.get("cache_hit", False)

    st.markdown(f"### {'✅ [CACHE HIT]' if cache_hit else '🟢 [LIVE ANSWER]'}\n**Answer:**\n{answer}")
    st.markdown("---")
    st.markdown("#### Referenced Slides:")
    for slide in slides:
        col1, col2 = st.columns([1, 4])
        with col1:
            if slide.get("image_path") and os.path.exists(slide["image_path"]):
                st.image(slide["image_path"], width=120)
        with col2:
            st.markdown(f"**Slide {slide.get('slide_number', '?')}**\n{slide.get('summary', '')}")
