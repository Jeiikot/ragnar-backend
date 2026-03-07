# Changelog

All notable changes to the Ragnar backend are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## [0.2.0] - 2026-03-07

### CI

- **ci:** Add git-cliff config and generate initial changelog


### Documentation

- **api:** Rewrite README to reflect DDD-lite architecture

- **docs:** Add skills reference and simplify CLAUDE.md


### Features

- **api:** Add standardized response envelope and error handler

- **domain:** Add domain error hierarchy


### Styling

- **api:** Apply ruff formatting fixes

## [0.1.0] - 2026-02-25

### CI

- **ci:** Add pre-commit hooks, gitignore, and env example


### Dependencies

- **deps:** Add pyproject.toml and uv lockfile


### Docker

- **docker:** Add dockerfile and docker-compose configuration


### Documentation

- **readme:** Add project readme, CLAUDE context, and agents guide


### Features

- **config:** Add pydantic-settings configuration

- **domain:** Add domain entities and protocol-based ports

- **providers:** Add multi-provider LLM support

- **indexing:** Add chromadb indexing infrastructure

- **chat:** Add langchain rag chat engine and retriever

- **app:** Add indexing use-case service

- **api:** Add fastapi rest api with chat and indexing endpoints


### Tests

- Add unit, integration and e2e test suite


