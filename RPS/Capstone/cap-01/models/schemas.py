from pydantic import BaseModel, Field
from typing import List, Optional


class RetrievedChunk(BaseModel):
    source: Optional[str]
    page: Optional[int]
    chunk_id: Optional[int]
    page_content: str


class RAGResponse(BaseModel):
    answer: str
    citations: List[str]
    metadata: Optional[dict]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = 4
