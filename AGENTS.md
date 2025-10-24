# Repository Guidelines

## Project Structure & Module Organization
- Core implementation lives in `src/llmemory`; keep orchestration in `manager.py`, database work in `db.py`, and embedding adapters in `embedding_providers.py`.
- Integration and regression suites sit in `tests/`, grouped by feature with shared fixtures in `tests/conftest.py` and sample corpora under `tests/res/`.
- `docs/` holds reference guides, `examples/` provides runnable snippets, and `run_server.py` is the local API demo entrypoint.
- The `pgdbm` directory is vendored; adjust it in tandem with dependency pins in `pyproject.toml`.

## Build, Test, and Development Commands
- `uv sync --all-extras` installs runtime and dev dependencies, including the local `pgdbm` source.
- `uv run pytest` runs the asynchronous test suite; add `--cov=llmemory` to refresh coverage.
- `uv run pre-commit run --all-files` executes the Black, isort, Ruff, and mypy hooks used in CI.

## Coding Style & Naming Conventions
- Use four-space indentation and keep lines under 100 characters; format via `uv run black src tests` and `uv run isort src tests`.
- Favor descriptive snake_case for functions and fixtures, PascalCase for classes, and UPPER_SNAKE for constants.
- Maintain full type hints so the strict mypy configuration passes without ignores.

## Testing Guidelines
- Add tests under `tests/` following the `test_*.py` naming pattern and mirroring module structure.
- Mark asyncio tests with `pytest.mark.asyncio` and reuse fixtures from `tests/conftest.py` to avoid custom setup.
- When PostgreSQL is needed, set `DATABASE_URL` to a pgvector-enabled database and reuse the corpora in `tests/res/`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `docs:`) as in current history, keeping subjects imperative and succinct.
- Provide PRs with a change summary, test evidence (`pytest`, coverage notes), environment updates, and linked issues.
- Update `README.md`, `docs/`, or `CHANGELOG.md` when touching public APIs; include screenshots only for user-facing examples.

## Environment & Configuration Notes
- Target PostgreSQL 14+ with pgvector and supply embedding credentials (`LLMEMORY_OPENAI_API_KEY` or local model config) via environment variables.
- Use `.env` files only for local work—never commit secrets, and prefer runtime overrides through `uv run` or your orchestrator.
