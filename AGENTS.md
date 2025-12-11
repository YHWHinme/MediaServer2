# AGENTS.md - Guidelines for Agentic Coding in MediaServer2

## Build/Lint/Test Commands
- Install dependencies: `uv sync`
- Run app: `uv run streamlit run main.py` (or `uv run python main.py`)
- Lint: `uv run ruff check .` (fix: `uv run ruff check --fix .`)
- Format: `uv run ruff format .`
- Test all: `uv run pytest`
- Test single: `uv run pytest path/to/test_file.py::test_function_name`
- Type check: `uv run mypy .` (if mypy installed)

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local. Use absolute imports.
- **Formatting**: Use ruff for auto-formatting (line length 88, black-compatible).
- **Types**: Add type hints to all functions/methods. Use `typing` for complex types.
- **Naming**: snake_case for variables/functions, CamelCase for classes, UPPER_CASE for constants.
- **Error Handling**: Use try/except with specific exceptions. Log errors with logging module.
- **Docstrings**: Use Google-style docstrings for functions/classes.
- **Security**: Never log secrets/keys. Use environment variables for config.
- **Commits**: Use conventional commits (feat:, fix:, etc.). No secrets in commits.

No Cursor or Copilot rules found.