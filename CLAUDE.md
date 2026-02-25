# Ragnar Backend

FastAPI REST API with DDD-lite architecture for RAG-based code analysis.

## Build & Run

```bash
# Install dependencies
uv sync --extra dev

# Run dev server
uv run uvicorn api.main:app --host 0.0.0.0 --port 8765 --reload

# Run tests
uv run pytest -q                    # unit
uv run pytest -m integration        # integration (needs local Chroma)
uv run pytest -m e2e                # e2e (needs API keys)

# Lint
uv run ruff check .

# Docker
make docker-build && make docker-run
```

## Architecture (DDD-Lite / Ports & Adapters)

### Layer Rules

| Layer | Directory | May Import | Must NOT Import |
|-------|-----------|------------|-----------------|
| Domain | `domain/` | stdlib only | api, application, infrastructure |
| Application | `application/` | domain, shared | api, infrastructure |
| Infrastructure | `infrastructure/` | domain, shared | api |
| API | `api/` | all layers | — |
| Shared | `shared/` | stdlib, pydantic | domain, application, infrastructure, api |

### Key Patterns

- **Ports:** `typing.Protocol` in `domain/indexing/ports/protocols.py` — never use `abc.ABC`
- **Entities:** `@dataclass(frozen=True)` in `domain/`
- **Wiring:** `functools.partial` in `infrastructure/indexing/adapters.py` binds settings to adapters
- **DI:** FastAPI `Depends()` in `api/dependencies.py` provides `Settings`, `ChatEngine`, `IndexingPorts`, and `DocumentIndexingPorts`; port bundles are built per-request at the API layer (composition root)
- **Providers:** Three LLM/embedding providers (OpenAI, Ollama, HuggingFace) with identical `ProviderBuilders` contract. Auto-selection in `infrastructure/providers/selector.py`
- **Sessions:** Each `session_id` maps to a separate Chroma collection for multi-tenant isolation

### Adding New Code

**New endpoint:**
1. Add Pydantic schema in `api/schemas/`
2. Add route in `api/routers/`
3. Add tests in `tests/unit/api/routers/`

**New domain port:**
1. Define `Protocol` in `domain/`
2. Implement adapter in `infrastructure/`
3. Wire in `infrastructure/indexing/adapters.py`

**New provider:**
1. Create `infrastructure/providers/<name>.py` with `build_chat_model()` + `build_embeddings()`
2. Register in `_PROVIDER_BUILDERS` dict in `infrastructure/providers/__init__.py`
3. Update `ProviderName` literal in `infrastructure/providers/types.py`

## Coding Conventions

- Every file starts with `from __future__ import annotations`
- Line length: 99 (ruff, target Python 3.11)
- Type hints everywhere: `str | None` (not `Optional[str]`), `list[str]` (not `List[str]`)
- Logging: `logger = logging.getLogger(__name__)` at module level
- Request schemas: `model_config = ConfigDict(extra="forbid")`
- Application services are plain functions, not classes
- Use `asyncio.to_thread()` for CPU-bound work in async routes

## Testing Conventions

- Fixtures in `tests/conftest.py`: `test_settings`, `mock_retriever`, `mock_chat_engine`, `client`
- Router tests use `SyncASGIClient` (sync httpx wrapper) with `dependency_overrides`
- Unit tests use `MagicMock`, `AsyncMock`, `patch` from `unittest.mock`
- Markers: `@pytest.mark.integration`, `@pytest.mark.e2e`
- `asyncio_mode = "auto"` — async test functions work without decorators

## Configuration

All config in `shared/config.py` `Settings` class (pydantic-settings). Env vars map to lowercase field names. Access via `get_settings()` lazy singleton. Reset with `reset_settings()` for tests.

## Key Files

- `api/main.py` — App factory, lifespan, exception handlers
- `api/dependencies.py` — FastAPI DI: `get_app_settings`, `get_chat_engine_dep`, `get_indexing_ports`, `get_document_ports`, `reset_singletons`
- `shared/config.py` — All settings with validators
- `domain/indexing/ports/` — Domain port protocols (`protocols.py`) and bundles (`bundles.py`)
- `infrastructure/indexing/adapters.py` — Port-to-adapter wiring
- `infrastructure/chat/engine.py` — RAG pipeline (ChatEngine class)
- `infrastructure/providers/__init__.py` — Provider facade
- `application/indexing/service.py` — Indexing use cases
