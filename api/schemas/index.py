from __future__ import annotations

from pydantic import BaseModel, Field


class IndexResponse(BaseModel):
    documents_indexed: int = Field(
        ..., description="Number of document chunks stored in the vector DB"
    )


class IndexSourceInfo(BaseModel):
    name: str
    chunks: int


class IndexStatusResponse(BaseModel):
    sources: list[IndexSourceInfo]


class ClearResponse(BaseModel):
    cleared: bool = True
