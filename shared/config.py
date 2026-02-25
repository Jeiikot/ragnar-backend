from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        validate_default=True,
    )

    # OpenAI
    chat_provider: Literal["auto", "openai", "ollama", "huggingface"] = "auto"
    embeddings_provider: Literal["auto", "openai", "ollama", "huggingface"] = "auto"
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="")
    chat_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_chat_model: str = Field(default="qwen2.5:7b-instruct")
    ollama_embedding_model: str = Field(default="nomic-embed-text")

    # Hugging Face
    huggingface_api_key: str = Field(default="")
    huggingface_chat_model: str = Field(default="HuggingFaceH4/zephyr-7b-beta")
    huggingface_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    huggingface_max_new_tokens: int = Field(default=512, ge=64, le=8192)
    chat_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # Chroma
    chroma_persist_dir: str = Field(default="./chroma_data")
    chroma_collection_name: str = Field(default="ragnar")

    # Chunking
    chunk_size: int = Field(default=1500, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)

    # Retriever
    retriever_search_type: Literal["mmr", "similarity"] = "mmr"
    retriever_k: int = Field(default=6, ge=1)
    retriever_fetch_k: int = Field(default=20, ge=1)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8765, ge=1, le=65535)
    log_level: str = Field(default="INFO")
    log_format: Literal["json", "text"] = "json"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info: Any) -> int:
        chunk_size = info.data.get("chunk_size", 1500)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str] | Any:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return ["*"]
            if raw.startswith("["):
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("cors_origins JSON value must be a list")
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in raw.split(",") if item.strip()]
        return value

    @field_validator("openai_base_url", "ollama_base_url", mode="before")
    @classmethod
    def normalize_base_url(cls, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip().rstrip("/")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Lazy singleton — reads env / .env on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Clear cached Settings instance (useful for tests/CLI)."""
    global _settings
    _settings = None
