from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User query")
    thread_id: str = Field(default="default", max_length=100)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)

class Citation(BaseModel):
    id: str
    doc_id: str
    doc_title: str
    page: int
    chunk_text: str
    score: float

class ChunkOut(BaseModel):
    id: str
    doc_id: str
    doc_title: str
    page: int
    text: str

class ThreadCreate(BaseModel):
    title: Optional[str] = Field(default="New consultation", max_length=200)

class ThreadOut(BaseModel):
    id: str
    title: str
    created_at: str
