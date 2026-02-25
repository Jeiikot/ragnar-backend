# ── Builder stage ──────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml .
COPY api/ api/
COPY application/ application/
COPY domain/ domain/
COPY infrastructure/ infrastructure/
COPY shared/ shared/
RUN pip install --no-cache-dir --prefix=/install ".[dev]"

# ── Runtime stage ─────────────────────────────────────────────────────
FROM python:3.12-slim

RUN groupadd --system app && useradd --system --gid app app

WORKDIR /app

COPY --from=builder /install /usr/local
COPY .ignore .ignore
COPY api/ api/
COPY application/ application/
COPY domain/ domain/
COPY infrastructure/ infrastructure/
COPY shared/ shared/

RUN mkdir -p /app/chroma_data && chown -R app:app /app

USER app

EXPOSE 8765

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8765"]
