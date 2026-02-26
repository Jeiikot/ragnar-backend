# Ragnar Backend

FastAPI REST API for indexing source code and PDF documents into ChromaDB and answering natural-language questions via a LangChain RAG pipeline.

## Stack

- **FastAPI** — async REST API
- **LangChain** — RAG pipeline with conversational memory
- **ChromaDB** — vector store (persisted to disk)
- **LLM/Embeddings providers** — OpenAI, Ollama, HuggingFace (auto or explicit)
- **pydantic-settings** — typed configuration from env vars

## Configuration

```bash
cp .env.example .env
```

Auto-selection order: **Ollama** (if reachable) → **OpenAI** (if `OPENAI_API_KEY`) → **HuggingFace** (if `HUGGINGFACE_API_KEY`)

Recommended models in `.env`:
- Local: `OLLAMA_CHAT_MODEL=qwen2.5:14b`, `OLLAMA_EMBEDDING_MODEL=nomic-embed-text`
- Cloud: `CHAT_MODEL=gpt-4o-mini`, `EMBEDDING_MODEL=text-embedding-3-small`

## Run with uv

```bash
uv sync --extra dev
uv run uvicorn api.main:app --host 0.0.0.0 --port 8765 --reload
```

## Run with Docker Compose

```bash
docker compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/index/code` | Upload and index a source code ZIP |
| POST | `/api/v1/index/documents` | Upload and index PDFs or ZIP of PDFs |
| GET | `/api/v1/index/status` | Get indexed sources for a session |
| POST | `/api/v1/index/clear` | Clear indexed data for a session |
| POST | `/api/v1/chat` | Ask a question about indexed content |
| GET | `/api/v1/health` | Health check |

## Testing

```bash
uv run pytest -q                  # unit tests
uv run pytest -m integration      # integration tests
uv run pytest -m e2e              # e2e tests
uv run ruff check .               # linting
```

## Project Structure

```
backend/
├── api/              # Routers, schemas, dependencies (composition root)
├── application/      # Use cases — orchestrates domain ports
├── domain/           # Entities + Protocol-based ports
├── infrastructure/   # Adapters (providers, chat engine, indexing, retriever)
├── shared/           # Configuration (pydantic-settings)
└── tests/            # unit/, integration/, e2e/
```

## Architecture

DDD-lite (Ports & Adapters):
- `domain/` defines `Protocol`-based ports and frozen dataclass entities
- `infrastructure/` provides concrete adapters wired via `functools.partial`
- `application/` orchestrates use cases through ports only
- `api/dependencies.py` is the composition root — builds and injects port bundles via `Depends()`
