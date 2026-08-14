"""
Production models: Postgres + pgvector
Fallback to in-memory if DB unreachable (keeps local dev without Postgres)
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, ForeignKey, Float, Index
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Thread(Base):
    __tablename__ = "threads"
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    user_id = Column(String, default="anonymous")  # for auth
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True)
    thread_id = Column(String, ForeignKey("threads.id", ondelete="CASCADE"), index=True)
    role = Column(String, nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    citations = Column(Text)  # JSON string
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    thread = relationship("Thread", back_populates="messages")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(String, primary_key=True)
    doc_id = Column(String, index=True)
    doc_title = Column(String, nullable=False)
    page = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    # 768 for Gemini text-embedding-004, 1536 for others — use 768
    embedding = Column(Vector(768))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Index for fast cosine search
# CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
