# AGENTS.md - Guidelines for Agentic Coding in MediaServer2

## Build/Lint/Test Commands
- Install dependencies: `uv sync`
- Run app: `uv run streamlit run main.py` (or `uv run python main.py`)
- Lint: `uv run ruff check .` (fix: `uv run ruff check --fix .`)
- Format: `uv run ruff format .`
- Test all: `uv run pytest` (no tests yet; add to tests/ dir)
- Test single: `uv run pytest tests/test_file.py::test_function_name`
- Type check: `uv run mypy .` (mypy not installed; add if needed)

## Code Style Guidelines
- **Imports**: Standard library first, then third-party (langchain, etc.), then local. Use absolute imports.
- **Formatting**: Use ruff for auto-formatting (line length 88, black-compatible).
- **Types**: Add type hints to all functions/methods. Use `typing` for complex types.
- **Naming**: snake_case for variables/functions, CamelCase for classes, UPPER_CASE for constants.
- **Error Handling**: Use try/except with specific exceptions. Log with logging module.
- **Docstrings**: Use Google-style for functions/classes.
- **Security**: Never log secrets/keys. Use env vars for config (e.g., Ollama URLs).
- **Commits**: Conventional commits (feat:, fix:, etc.). No secrets in commits.

No Cursor or Copilot rules found.