from pydantic import BaseModel, Field


class IndexResponse(BaseModel):
    status: str = "ok"
    documents_indexed: int = Field(
        ..., description="Number of document chunks stored in the vector DB"
    )


class IndexSourceInfo(BaseModel):
    name: str
    chunks: int


class IndexStatusResponse(BaseModel):
    sources: list[IndexSourceInfo]
    total_chunks: int
