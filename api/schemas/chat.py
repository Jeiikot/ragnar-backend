from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The natural language question",
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        description="Session ID for conversation memory",
        examples=["abc123"],
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(
        default_factory=list,
        description="Source references in format 'file:line'",
    )
