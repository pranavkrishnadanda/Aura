from pydantic import BaseModel, Field
from typing import Optional, List

from app.config import settings


class ChatRequest(BaseModel):
    # Bound comes from settings so raising MAX_MESSAGE_LENGTH actually takes
    # effect. It was hardcoded to 4000 here, so Pydantic rejected anything longer
    # before the endpoint's own MAX_MESSAGE_LENGTH check could ever run, making
    # that setting dead above 4000 and the endpoint check unreachable.
    message: str = Field(..., min_length=1, max_length=settings.MAX_MESSAGE_LENGTH, description="User query")
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
