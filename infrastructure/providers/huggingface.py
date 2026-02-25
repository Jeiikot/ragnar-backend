from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from shared.config import Settings


def _has_value(value: str) -> bool:
    return bool(value and value.strip())


def build_chat_model(settings: Settings) -> BaseChatModel:
    from langchain_community.chat_models import ChatHuggingFace
    from langchain_community.llms import HuggingFaceEndpoint

    if not _has_value(settings.huggingface_api_key):
        raise ValueError("HUGGINGFACE_API_KEY is required for chat_provider=huggingface")
    endpoint = HuggingFaceEndpoint(
        model=settings.huggingface_chat_model,
        repo_id=settings.huggingface_chat_model,
        huggingfacehub_api_token=settings.huggingface_api_key,
        task="text-generation",
        max_new_tokens=settings.huggingface_max_new_tokens,
        temperature=settings.chat_temperature,
    )
    return ChatHuggingFace(llm=endpoint)


def build_embeddings(settings: Settings) -> Embeddings:
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

    if not _has_value(settings.huggingface_api_key):
        raise ValueError("HUGGINGFACE_API_KEY is required for embeddings_provider=huggingface")
    return HuggingFaceInferenceAPIEmbeddings(
        model_name=settings.huggingface_embedding_model,
        api_key=settings.huggingface_api_key,
    )
