# AGENTS.md - Guidelines for Agentic Coding in MediaServer2

## Build/Lint/Test Commands
- Install dependencies: `uv sync`
- Run app: `uv run streamlit run main.py` (or `uv run python main.py`)
- Lint: `uv run ruff check .` (fix: `uv run ruff check --fix .`)
- Format: `uv run ruff format .`
- Test all: `uv run pytest` (no tests yet; add to tests/ dir)
- Test single: `uv run pytest tests/test_file.py::test_function_name`
- Type check: `uv run mypy .` (mypy not installed; add if needed)
- Set up OpenAI API key: Copy `.env.example` to `.env` and add your API key

## Code Style Guidelines
- **Imports**: Standard library first, then third-party (langchain, etc.), then local. Use absolute imports.
- **Formatting**: Use ruff for auto-formatting (line length 88, black-compatible).
- **Types**: Add type hints to all functions/methods. Use `typing` for complex types.
- **Naming**: snake_case for variables/functions, CamelCase for classes, UPPER_CASE for constants.
- **Error Handling**: Use try/except with specific exceptions. Log with logging module.
- **Docstrings**: Use Google-style for functions/classes.
- **Security**: Never log secrets/keys. Use env vars for config (e.g., OpenAI API keys).
- **Commits**: Conventional commits (feat:, fix:, etc.). No secrets in commits.

## Environment Variables
- OPENAI_API_KEY: Required for OpenAI API access
- OPENAI_MODEL: Chat model (default: gpt-4)
- OPENAI_EMBEDDING_MODEL: Embedding model (default: text-embedding-ada-002)

## Cost Monitoring
- Cost monitoring function added but unimplemented (returns placeholder data)
- Tracks estimated API costs for GPT-4 usage
- TODO: Implement actual cost calculation using OpenAI response metadata

## Context Access Issue and Fix

### Issue Description
The AI in the RAG system was not consistently accessing or utilizing relevant context from documents when answering questions. In the original implementation, context was always retrieved and injected into the prompt via a direct RAG chain, but the LLM often ignored this context, potentially due to:
- Irrelevant or excessive context being provided unconditionally.
- Weak prompt instructions not compelling the model to prioritize retrieved context over its training data.
- Lack of explicit decision-making by the LLM on when to use external information.

This led to responses that were generic or hallucinated rather than grounded in the provided documents.

### Root Cause Analysis
- **Direct RAG Limitation**: The standard RAG approach pre-retrieves context for every query, which may include noise or irrelevant information, confusing the LLM.
- **No Tool-Calling**: Without tool-calling, the LLM has no mechanism to explicitly request specific information; it must rely on the prompt's context, which it may not fully utilize.
- **Model Behavior**: LLMs like those used (e.g., Ollama models) can prioritize their internal knowledge over provided context if not strongly instructed or if the context is not directly relevant.

### Proposed Solution: Tool-Calling Agent
Implement a tool-calling agent (inspired by the SimpleAgent class) that allows the LLM to dynamically invoke tools for information retrieval. This shifts responsibility to the LLM for deciding when and what context to access.

#### Key Components:
1. **Tools**:
   - `vector_search_tool`: Queries the vector store for relevant document chunks based on the query.
   - `web_search_tool`: Uses Perplexity API for web-based information when local documents are insufficient.
   - (Optional) `image_recall_tool`: For image-based queries in multi-modal setups.

2. **System Prompt**: Instructs the LLM to use tools for questions requiring external knowledge, ensuring context is accessed only when needed.

3. **Agent Logic**: Follows the SimpleAgent pattern:
   - Start with system prompt and user question.
   - If the LLM calls tools, execute them and append results to messages.
   - Final response incorporates tool outputs as context.

#### Benefits:
- **Dynamic Context Retrieval**: LLM decides relevance, reducing noise.
- **Improved Accuracy**: Explicit tool use ensures responses are based on retrieved data.
- **Flexibility**: Supports local docs, web search, or combined modes seamlessly.
- **Multi-Modal Ready**: Extensible for images/videos via additional tools.

#### Implementation Steps:
1. Define tools using `@tool` decorator from `langchain_core.tools`.
2. Bind tools to the ChatOllama client.
3. Replace the RAG chain in the Agent class with tool-calling invoke logic.
4. Update UI modes to use the agent for local/web/combined searches.
5. Test with sample queries to verify context utilization.

#### Risks and Considerations:
- **Model Support**: Ensure the Ollama model (e.g., qwen3) supports tool-calling; may need to switch models if not.
- **Performance**: Tool calls add latency; optimize retrieval (e.g., limit docs returned).
- **Error Handling**: Handle cases where tools fail or return no results.
- **Security**: Ensure tools don't expose sensitive data; validate inputs.

This fix should resolve the context access issue by making information retrieval LLM-driven and explicit.

## Pending Improvements
- ✅ COMPLETED: Migrated from Ollama to OpenAI GPT-4 with tool calling support
- ✅ COMPLETED: Fix prompt template in Agent class: Replaced raw string template with ChatPromptTemplate.from_messages() for proper chat formatting, including system message with {context} and MessagesPlaceholder for {question} to resolve ValueError and align with working examples.
- Add MCP (Model Context Protocol) integration: Implement MCP Python SDK to enable Obsidian integration via Docker MCP server. Requires API key management (python-dotenv), MCP client setup, and Docker container orchestration for connecting to Obsidian Local REST API plugin.

No Cursor or Copilot rules found.