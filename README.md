# Ragnar Backend

Index source code and ask natural-language questions about it via REST API.

## Stack

- **FastAPI** — async REST API
- **LangChain** — RAG pipeline with conversational memory
- **ChromaDB** — vector store (persisted to disk)
- **LLM/Embeddings providers** — OpenAI, Ollama, Hugging Face (auto or explicit)
- **pydantic-settings** — typed configuration from env vars

## Configuration (all modes)

```bash
cp .env.example .env
# By default CHAT_PROVIDER=auto and EMBEDDINGS_PROVIDER=auto.
# Auto order: Ollama (if reachable) -> OpenAI (if OPENAI_API_KEY) -> Hugging Face (if HUGGINGFACE_API_KEY).
```

Recommended defaults in `.env.example`:
- Cloud quality: `CHAT_MODEL=gpt-4o-mini`, `EMBEDDING_MODEL=text-embedding-3-small`
- Local/free: `OLLAMA_CHAT_MODEL=qwen2.5:7b-instruct`, `OLLAMA_EMBEDDING_MODEL=nomic-embed-text`
- HF API: `HUGGINGFACE_CHAT_MODEL=HuggingFaceH4/zephyr-7b-beta`, `HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

## Run With pip

```bash
# from backend/
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn api.main:app --host 0.0.0.0 --port 8765 --reload
```

## Run With uv

```bash
# from backend/
uv sync --extra dev
uv run uvicorn api.main:app --host 0.0.0.0 --port 8765 --reload
```

## API Endpoints

| Method | Path              | Description                          |
|--------|-------------------|--------------------------------------|
| POST   | `/api/v1/index/upload` | Upload and index a zip archive |
| POST   | `/api/v1/chat`    | Ask a question about indexed code    |
| GET    | `/api/v1/health`  | Health check                         |

### Index (zip)

```bash
curl -X POST http://localhost:8765/api/v1/index/upload \
  -F "file=@./my-project.zip"
```

Behavior:
- Indexes all text files from the uploaded zip.
- Ignores only by backend policy (`backend/.ignore`).
- Always excludes sensitive files and binary artifacts.
- Uses backend-wide exclusions from required `backend/.ignore`.

### Chat

```bash
curl -X POST http://localhost:8765/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does the auth module work?", "session_id": "abc123"}'
```

## Testing (pip / uv)

```bash
# pip
source .venv/bin/activate
pytest tests/unit -v
pytest tests/integration -v -m integration
pytest tests/e2e -v -m e2e

# uv
uv run pytest tests/unit -v
uv run pytest tests/integration -v -m integration
uv run pytest tests/e2e -v -m e2e
```

## Run With Docker

```bash
# from backend/
docker build -t ragnar-backend:local .
docker run --rm -it \
  -p 8765:8765 \
  --add-host=host.docker.internal:host-gateway \
  --env-file .env \
  -e CHROMA_PERSIST_DIR=/app/chroma_data \
  -e OLLAMA_BASE_URL=${OLLAMA_BASE_URL_DOCKER:-http://host.docker.internal:11434} \
  -v ragnar_chroma_data:/app/chroma_data \
  ragnar-backend:local
```

## Run With Docker Compose

```bash
# from backend/
docker compose up --build
```

## Recommended Flow (Ollama + Docker Compose)

1. Start Ollama on host:

```bash
ollama serve
```

2. Download models once:

```bash
ollama pull qwen2.5:14b
ollama pull nomic-embed-text
```

3. Start backend:

```bash
cd backend
docker compose up --build
```

4. Verify API health:

```bash
curl http://localhost:8765/api/v1/health
```

5. Index a project (must run before chat):

```bash
zip -r /tmp/my-project.zip ./my-project
curl -X POST http://localhost:8765/api/v1/index/upload \
  -F "file=@/tmp/my-project.zip"
```

6. Ask a question:

```bash
curl -X POST http://localhost:8765/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How does auth work?", "session_id": "demo-1"}'
```

Notes:
- If you change embeddings model/provider, re-run `/api/v1/index/upload`.

## Shortcuts (Makefile)

```bash
make help
make setup-pip
make setup-uv
make run-pip
make run-uv
make docker-build
make docker-run
make compose-up
```

## Project Structure

```
core/
  application/
    indexing_contracts.py
    indexing_use_cases.py
    indexing_service.py
  config.py
  chat.py
  retriever.py
  indexer.py
  infrastructure/
    indexing/
      file_collector.py
      chunker.py
      vector_store_writer.py
      zip_extractor.py
      ports_factory.py
  providers/
    __init__.py       # public facade (backward compatible imports)
    contracts.py      # provider builder protocols
    selector.py       # provider auto-resolution policy
    openai.py         # OpenAI chat + embeddings builders
    ollama.py         # Ollama chat + embeddings builders
    huggingface.py    # Hugging Face chat + embeddings builders
    types.py          # provider name aliases
  indexing/
    service.py        # compatibility facade used by API
    contracts.py      # compatibility re-export (application layer owns contracts)
    use_cases.py      # compatibility re-export (application layer owns use cases)
    file_discovery.py # legacy adapter module
    chunking.py       # legacy adapter module
    storage.py        # legacy adapter module
    zip_utils.py      # legacy adapter module
api/
  main.py
  schemas/
  dependencies.py
  routers/
tests/
```

## Internal Dependency Flow (DDD-lite)

```
API routers
  -> core.indexing.service / core.chat / core.retriever
  -> application services + use cases (orchestrate)
  -> infrastructure adapters (indexing + provider SDK builders)
  -> external SDKs/services (LangChain, Chroma, OpenAI/Ollama/HF)
```

Rules:
- Keep API layer free of provider/indexing implementation details.
- Keep use cases focused on orchestration logic.
- Keep adapter modules focused on one external concern.
- Keep `core.providers.__init__` as stable import surface.

## Adding a New Provider

1. Create `core/providers/<provider>.py` with:
   - `build_chat_model(settings) -> BaseChatModel`
   - `build_embeddings(settings) -> Embeddings`
2. Register it in `core/providers/__init__.py` inside `_PROVIDER_BUILDERS`.
3. Update `core/providers/types.py` (`ProviderName`) and `core/config.py` provider literals.
4. Extend auto-resolution policy in `core/providers/selector.py` if needed.
5. Add/adjust tests in `tests/unit/core/test_providers.py`.

Compatibility note:
- Preserve public facade exports in `core/providers/__init__.py`.
- Do not change endpoint contracts when only adding a provider.
